"""SNOMED attribute and reference index builder for pgvector.

Creates/populates two PostgreSQL tables in ``VOCAB_SCHEMA``:
    snomed_attribute   — attribute value concepts with embeddings.
    snomed_reference   — sampled SNOMED source terms with relationship-target rows
                         and source-term embeddings.

This module is the canonical package-level version of
``sandbox/build_pg_indexes_attributes.py``.
The sandbox script is kept as a standalone convenience but this module is what
the CLI (``python -m ariadne.hierarchy build-index``) calls.

Usage::

    python -m ariadne.hierarchy build-index                        # append mode
    python -m ariadne.hierarchy build-index --if-exists skip      # skip if populated
    python -m ariadne.hierarchy build-index --if-exists rebuild   # truncate and rebuild
    python -m ariadne.hierarchy build-index --attributes-only
    python -m ariadne.hierarchy build-index --reference-only

Environment variables (loaded from .env):
    VOCAB_CONNECTION_STRING_ADM   psycopg DSN with write/DDL access
    VOCAB_CONNECTION_STRING       SQLAlchemy URL for read-only data loading
    VOCAB_SCHEMA                  OMOP vocabulary schema name
    EMBEDDING_MODEL               Embedding model name (via GENAI_PROVIDER routing)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import psycopg
from psycopg import sql
from pgvector.psycopg import register_vector
from sqlalchemy import create_engine

from ariadne.utils.settings import HierarchySettings, _DEFAULT_SNOMED_RELATIONSHIPS
from ariadne.utils.gen_ai_api import get_embedding_vectors
from ariadne.utils.utils import get_environment_variable
from ariadne.utils.utils import get_project_root

logger = logging.getLogger(__name__)

# Re-use the canonical relationship list from settings (single source of truth).
SNOMED_RELATIONSHIPS: list[str] = list(_DEFAULT_SNOMED_RELATIONSHIPS)


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _pg_connect() -> psycopg.Connection:
    """Return a psycopg connection (with pgvector registered) using the admin DSN."""
    conn_str = get_environment_variable("VOCAB_CONNECTION_STRING")
    conn_str = conn_str.replace("+psycopg", "").replace("+psycopg2", "")
    conn = psycopg.connect(conn_str)
    register_vector(conn)
    return conn


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

def create_tables(conn: psycopg.Connection, dim: int = 3072) -> None:
    """Create ``snomed_attribute`` and ``snomed_reference`` tables.

    Safe to call when tables already exist (uses ``CREATE TABLE IF NOT EXISTS``).

    Args:
        conn: Admin psycopg connection.
        dim: Embedding dimension (default 3072 for text-embedding-3-large).
    """
    schema = get_environment_variable("VOCAB_SCHEMA")
    with conn.cursor() as cur:
        cur.execute(sql.SQL("""
            CREATE TABLE IF NOT EXISTS {schema}.snomed_attribute (
                id                 SERIAL PRIMARY KEY,
                concept_id         INTEGER       NOT NULL,
                concept_code       VARCHAR(255)  NOT NULL,
                concept_name       VARCHAR(255)  NOT NULL,
                attribute_category VARCHAR(255)  NOT NULL,
                embedding          halfvec({dim})
            )
        """).format(schema=sql.Identifier(schema), dim=sql.Literal(dim)))
        cur.execute(sql.SQL("""
            CREATE TABLE IF NOT EXISTS {schema}.snomed_reference (
                id                 SERIAL PRIMARY KEY,
                concept_id_1       INTEGER       NOT NULL,
                concept_name_1     VARCHAR(255)  NOT NULL,
                concept_id_2       INTEGER       NOT NULL,
                concept_code_2     VARCHAR(255)  NOT NULL,
                concept_name_2     VARCHAR(255)  NOT NULL,
                attribute_category VARCHAR(255)  NOT NULL,
                embedding          halfvec({dim})
            )
        """).format(schema=sql.Identifier(schema), dim=sql.Literal(dim)))

    conn.commit()
    logger.info("Tables created/verified in schema '%s'.", schema)


def create_embedding_indexes(
    conn: psycopg.Connection,
    build_attribute: bool = True,
    build_reference: bool = True,
) -> None:
    """Create HNSW embedding indexes for populated target tables when requested."""
    schema = get_environment_variable("VOCAB_SCHEMA")
    with conn.cursor() as cur:
        if build_attribute:
            cur.execute(sql.SQL("""
                CREATE INDEX IF NOT EXISTS snomed_attribute_embedding_idx
                ON {schema}.snomed_attribute
                USING hnsw (embedding halfvec_cosine_ops)
            """).format(schema=sql.Identifier(schema)))

        if build_reference:
            cur.execute(sql.SQL("""
                CREATE INDEX IF NOT EXISTS snomed_reference_embedding_idx
                ON {schema}.snomed_reference
                USING hnsw (embedding halfvec_cosine_ops)
            """).format(schema=sql.Identifier(schema)))

    conn.commit()
    logger.info(
        "Embedding indexes created/verified (attribute=%s, reference=%s).",
        build_attribute,
        build_reference,
    )


# ---------------------------------------------------------------------------
# Population check
# ---------------------------------------------------------------------------

def check_populated(conn: psycopg.Connection, table: str = "snomed_attribute") -> bool:
    """Return True if *table* already contains at least one row.

    Args:
        conn: psycopg connection (read access is enough).
        table: Table name to check (``"snomed_attribute"`` or ``"snomed_reference"``).

    Returns:
        ``True`` if the table is non-empty.
    """
    schema = get_environment_variable("VOCAB_SCHEMA")
    table_path = f"{schema}.{table}"
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (table_path,))
        if cur.fetchone()[0] is None:
            return False
        cur.execute(
            sql.SQL("SELECT 1 FROM {schema}.{table} LIMIT 1").format(
                schema=sql.Identifier(schema),
                table=sql.Identifier(table),
            )
        )
        return cur.fetchone() is not None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_attributes_from_db() -> pd.DataFrame:
    """Load SNOMED attribute value rows from the OMOP vocabulary DB."""
    engine = create_engine(get_environment_variable("VOCAB_CONNECTION_STRING"))
    schema = get_environment_variable("VOCAB_SCHEMA")
    rel_list = ", ".join(f"'{r}'" for r in SNOMED_RELATIONSHIPS)
    df = pd.read_sql(f"""
        SELECT DISTINCT c2.concept_id, 
            c2.concept_code, 
            c2.concept_name,
            relationship_id AS attribute_category
        FROM {schema}.concept c
        INNER JOIN {schema}.concept_relationship cr
            ON  c.concept_id   = cr.concept_id_1
        INNER JOIN {schema}.concept c2 ON c2.concept_id = cr.concept_id_2
        WHERE c.vocabulary_id = 'SNOMED'
            AND c.standard_concept = 'S'
            AND cr.relationship_id IN ({rel_list})
            AND cr.invalid_reason IS NULL;
    """, engine)
    logger.info("Loaded %d attribute rows from DB.", len(df))
    return df


def load_reference_from_db(sample_size: int) -> pd.DataFrame:
    """Load and sample SNOMED reference rows from the OMOP vocabulary DB.

    Args:
        sample_size: Number of unique ``concept_id_1`` source concepts to include.

    Notes:
        The SQL query first pulls up to ``sample_size * 10`` rows, then applies a
        deterministic random sample (seed 42) over unique source concept IDs when
        more than ``sample_size`` IDs are present.
    """
    engine = create_engine(get_environment_variable("VOCAB_CONNECTION_STRING"))
    schema = get_environment_variable("VOCAB_SCHEMA")
    rel_list = ", ".join(f"'{r}'" for r in SNOMED_RELATIONSHIPS)
    df = pd.read_sql(f"""
        SELECT c.concept_id  AS concept_id_1,  
            c.concept_code  AS concept_code_1,
            c.concept_name AS concept_name_1,
            c2.concept_id  AS concept_id_2,  
            c2.concept_code AS concept_code_2,
            c2.concept_name AS concept_name_2,
            relationship_name AS attribute_category
        FROM {schema}.concept c
        JOIN {schema}.concept_relationship cr
            ON  c.concept_id    = cr.concept_id_1
            AND c.vocabulary_id = 'SNOMED'
            AND c.standard_concept = 'S'
            AND cr.relationship_id IN ({rel_list})
            AND cr.invalid_reason IS NULL
        JOIN {schema}.concept c2 ON c2.concept_id = cr.concept_id_2
        JOIN {schema}.relationship r ON cr.relationship_id = r.relationship_id
        LIMIT {sample_size * 10}
    """, engine)
    unique_ids = df["concept_id_1"].unique()
    if len(unique_ids) > sample_size:
        rng = np.random.default_rng(42)
        sampled_ids = rng.choice(unique_ids, size=sample_size, replace=False)
        df = df[df["concept_id_1"].isin(sampled_ids)]
    logger.info(
        "Loaded %d reference rows (%d unique source terms) from DB.",
        len(df),
        df["concept_id_1"].nunique(),
    )
    return df


def _batch_file(prefix: str, batch_index: int, folder: Path) -> Path:
    """Return deterministic parquet file path for one embedding batch."""
    return folder / f"{prefix}_batch_{batch_index:06d}.parquet"


def _write_parquet_atomic(df: pd.DataFrame, file_path: Path) -> None:
    """Write parquet atomically to avoid half-written checkpoints."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(file_path)


def _embedding_dim_from_batch_files(batch_files: list[Path]) -> int | None:
    """Infer embedding dimension from an existing parquet batch file."""
    for file_path in batch_files:
        if not file_path.exists():
            continue
        df = pd.read_parquet(file_path)
        if df.empty:
            continue
        return int(len(df.iloc[0]["embedding"]))
    return None


def _prepare_attribute_embedding_batches(
    df: pd.DataFrame,
    embedding_batch_size: int,
    embedding_cache_dir: Path,
) -> tuple[list[Path], int | None]:
    """Create one parquet file per embedding batch for attribute rows."""
    unique_terms = (
        df[["concept_id", "concept_name"]]
        .drop_duplicates()
        .sort_values("concept_id")
        .reset_index(drop=True)
    )
    total = len(unique_terms)
    total_batches = (total + embedding_batch_size - 1) // embedding_batch_size
    batch_files = [_batch_file("snomed_attribute", i, embedding_cache_dir) for i in range(total_batches)]

    total_cost = 0.0
    dim: int | None = None
    for batch_index in range(total_batches):
        file_path = batch_files[batch_index]
        if file_path.exists():
            continue

        start = batch_index * embedding_batch_size
        stop = start + embedding_batch_size
        unique_batch = unique_terms.iloc[start:stop].copy()
        result = get_embedding_vectors(unique_batch["concept_name"].astype(str).tolist())
        vectors = np.asarray(result["embeddings"], dtype=np.float32)
        total_cost += result["usage"]["total_cost_usd"]
        dim = int(vectors.shape[1])

        emb_by_id = {
            int(unique_batch.iloc[i]["concept_id"]): vectors[i].tolist()
            for i in range(len(unique_batch))
        }
        rows = df[df["concept_id"].astype(int).isin(emb_by_id.keys())].copy()
        rows["embedding"] = rows["concept_id"].astype(int).map(lambda concept_id: emb_by_id[int(concept_id)])
        rows = rows[["concept_id", "concept_code", "concept_name", "attribute_category", "embedding"]]
        _write_parquet_atomic(rows, file_path)
        logger.info("Cached attribute batch %d/%d: %s", batch_index + 1, total_batches, file_path.name)

    if dim is None:
        dim = _embedding_dim_from_batch_files(batch_files)
    logger.info("Attribute embedding batches ready (%d files). Cost: $%.4f", len(batch_files), total_cost)
    return batch_files, dim


def _prepare_reference_embedding_batches(
    df: pd.DataFrame,
    embedding_batch_size: int,
    embedding_cache_dir: Path,
) -> tuple[list[Path], int | None]:
    """Create one parquet file per embedding batch for reference rows."""
    unique_terms = (
        df[["concept_id_1", "concept_name_1"]]
        .drop_duplicates()
        .sort_values("concept_id_1")
        .reset_index(drop=True)
    )
    total = len(unique_terms)
    total_batches = (total + embedding_batch_size - 1) // embedding_batch_size
    batch_files = [_batch_file("snomed_reference", i, embedding_cache_dir) for i in range(total_batches)]

    total_cost = 0.0
    dim: int | None = None
    for batch_index in range(total_batches):
        file_path = batch_files[batch_index]
        if file_path.exists():
            continue

        start = batch_index * embedding_batch_size
        stop = start + embedding_batch_size
        unique_batch = unique_terms.iloc[start:stop].copy()
        result = get_embedding_vectors(unique_batch["concept_name_1"].astype(str).tolist())
        vectors = np.asarray(result["embeddings"], dtype=np.float32)
        total_cost += result["usage"]["total_cost_usd"]
        dim = int(vectors.shape[1])

        emb_by_id = {
            int(unique_batch.iloc[i]["concept_id_1"]): vectors[i].tolist()
            for i in range(len(unique_batch))
        }
        rows = df[df["concept_id_1"].astype(int).isin(emb_by_id.keys())].copy()
        rows["embedding"] = rows["concept_id_1"].astype(int).map(lambda concept_id: emb_by_id[int(concept_id)])
        rows = rows[
            [
                "concept_id_1",
                "concept_name_1",
                "concept_id_2",
                "concept_code_2",
                "concept_name_2",
                "attribute_category",
                "embedding",
            ]
        ]
        _write_parquet_atomic(rows, file_path)
        logger.info("Cached reference batch %d/%d: %s", batch_index + 1, total_batches, file_path.name)

    if dim is None:
        dim = _embedding_dim_from_batch_files(batch_files)
    logger.info("Reference embedding batches ready (%d files). Cost: $%.4f", len(batch_files), total_cost)
    return batch_files, dim


# ---------------------------------------------------------------------------
# Upsert functions
# ---------------------------------------------------------------------------

def upsert_attribute_index(
    conn: psycopg.Connection,
    batch_files: list[Path],
    upload_batch_size: int,
    rebuild: bool = False,
) -> psycopg.Connection:
    """Insert pre-embedded parquet batches into ``snomed_attribute``.

    Args:
        conn: Admin psycopg connection (returned — may be replaced on retry).
        batch_files: Parquet files containing pre-embedded attribute rows.
        upload_batch_size: Number of rows per INSERT chunk.
        rebuild: When True, truncate the table first.

    Returns:
        The (possibly reconnected) psycopg connection.
    """
    schema = get_environment_variable("VOCAB_SCHEMA")
    if rebuild:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("TRUNCATE TABLE {schema}.snomed_attribute RESTART IDENTITY")
                .format(schema=sql.Identifier(schema))
            )
        conn.commit()
        logger.info("Truncated snomed_attribute.")

    total_rows = 0
    insert_sql = sql.SQL(
        "INSERT INTO {schema}.snomed_attribute "
        "(concept_id, concept_code, concept_name, attribute_category, embedding) "
        "VALUES (%s, %s, %s, %s, %s)"
    ).format(schema=sql.Identifier(schema))

    for file_index, batch_file in enumerate(batch_files, start=1):
        df_batch = pd.read_parquet(batch_file)
        rows = [
            (int(r["concept_id"]), str(r["concept_code"]), str(r["concept_name"]),
             str(r["attribute_category"]), list(r["embedding"]))
            for _, r in df_batch.iterrows()
        ]
        total_rows += len(rows)

        for i in range(0, len(rows), upload_batch_size):
            chunk = rows[i: i + upload_batch_size]
            for attempt in range(3):
                try:
                    with conn.cursor() as cur:
                        for row in chunk:
                            cur.execute(insert_sql, row)
                    conn.commit()
                    break
                except Exception as exc:
                    logger.warning(
                        "File %d/%d chunk %d attempt %d failed (%s), reconnecting...",
                        file_index,
                        len(batch_files),
                        i // upload_batch_size + 1,
                        attempt + 1,
                        exc,
                    )
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = _pg_connect()
                    insert_sql = sql.SQL(
                        "INSERT INTO {schema}.snomed_attribute "
                        "(concept_id, concept_code, concept_name, attribute_category, embedding) "
                        "VALUES (%s, %s, %s, %s, %s)"
                    ).format(schema=sql.Identifier(schema))

        logger.info("Loaded attribute file %d/%d: %s", file_index, len(batch_files), batch_file.name)

    logger.info("Inserted %d rows into %s.snomed_attribute.", total_rows, schema)
    return conn


def upsert_reference_index(
    conn: psycopg.Connection,
    batch_files: list[Path],
    upload_batch_size: int,
    rebuild: bool = False,
) -> psycopg.Connection:
    """Insert pre-embedded parquet batches into ``snomed_reference``.

    Args:
        conn: Admin psycopg connection (returned — may be replaced on retry).
        batch_files: Parquet files containing pre-embedded reference rows.
        upload_batch_size: Number of rows per INSERT chunk.
        rebuild: When True, truncate the table first.

    Returns:
        The (possibly reconnected) psycopg connection.
    """
    schema = get_environment_variable("VOCAB_SCHEMA")
    if rebuild:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("TRUNCATE TABLE {schema}.snomed_reference RESTART IDENTITY")
                .format(schema=sql.Identifier(schema))
            )
        conn.commit()
        logger.info("Truncated snomed_reference.")

    total_rows = 0
    insert_sql = sql.SQL(
        "INSERT INTO {schema}.snomed_reference "
        "(concept_id_1, concept_name_1, concept_id_2, concept_code_2, "
        " concept_name_2, attribute_category, embedding) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)"
    ).format(schema=sql.Identifier(schema))

    for file_index, batch_file in enumerate(batch_files, start=1):
        df_batch = pd.read_parquet(batch_file)
        rows = [
            (int(r["concept_id_1"]), str(r["concept_name_1"]),
             int(r["concept_id_2"]), str(r["concept_code_2"]), str(r["concept_name_2"]),
             str(r["attribute_category"]), list(r["embedding"]))
            for _, r in df_batch.iterrows()
        ]
        total_rows += len(rows)

        for i in range(0, len(rows), upload_batch_size):
            chunk = rows[i: i + upload_batch_size]
            for attempt in range(3):
                try:
                    with conn.cursor() as cur:
                        for row in chunk:
                            cur.execute(insert_sql, row)
                    conn.commit()
                    break
                except Exception as exc:
                    logger.warning(
                        "File %d/%d chunk %d attempt %d failed (%s), reconnecting...",
                        file_index,
                        len(batch_files),
                        i // upload_batch_size + 1,
                        attempt + 1,
                        exc,
                    )
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = _pg_connect()
                    insert_sql = sql.SQL(
                        "INSERT INTO {schema}.snomed_reference "
                        "(concept_id_1, concept_name_1, concept_id_2, concept_code_2, "
                        " concept_name_2, attribute_category, embedding) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s)"
                    ).format(schema=sql.Identifier(schema))

        logger.info("Loaded reference file %d/%d: %s", file_index, len(batch_files), batch_file.name)

    logger.info("Inserted %d rows into %s.snomed_reference.", total_rows, schema)
    return conn


# ---------------------------------------------------------------------------
# High-level build entry point
# ---------------------------------------------------------------------------

def build_attribute_reference_tables(
    hierarchy_settings: HierarchySettings,
    if_exists: Literal["append", "skip", "rebuild"] = "append",
    build_attribute: bool = True,
    build_reference: bool = True,
) -> None:
    """Build (or rebuild) the pgvector SNOMED indexes.

    Args:
        hierarchy_settings: Hierarchy settings containing ``index_build`` defaults.
        if_exists: Behavior when target tables already contain rows:
            - ``"append"``: insert without truncating.
            - ``"skip"``: skip populated tables.
            - ``"rebuild"``: truncate first, then insert.
        build_attribute: Build ``snomed_attribute``?
        build_reference: Build ``snomed_reference``?
    """
    if if_exists not in {"append", "skip", "rebuild"}:
        raise ValueError(f"Unsupported if_exists mode: {if_exists}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    load_dotenv()

    embedding_cache_dir = Path(hierarchy_settings.index_build.embedding_cache_folder)
    rebuild = if_exists == "rebuild"

    embedding_cache_dir.mkdir(parents=True, exist_ok=True)

    if if_exists == "skip":
        conn = _pg_connect()
        try:
            if build_attribute and check_populated(conn, "snomed_attribute"):
                logger.info("snomed_attribute already populated — skipping.")
                build_attribute = False
            if build_reference and check_populated(conn, "snomed_reference"):
                logger.info("snomed_reference already populated — skipping.")
                build_reference = False
        finally:
            conn.close()

    if not build_attribute and not build_reference:
        logger.info("Nothing to build.")
        return

    attr_batch_files: list[Path] = []
    ref_batch_files: list[Path] = []
    dim: int | None = None

    # Build embeddings first (with parquet checkpoints), then create/upload tables.
    if build_attribute:
        df_attr = load_attributes_from_db()
        attr_batch_files, attr_dim = _prepare_attribute_embedding_batches(
            df_attr,
            embedding_batch_size=hierarchy_settings.index_build.embedding_batch_size,
            embedding_cache_dir=embedding_cache_dir,
        )
        dim = attr_dim if dim is None else dim

    if build_reference:
        df_ref = load_reference_from_db(sample_size=hierarchy_settings.index_build.reference_sample_size)
        ref_batch_files, ref_dim = _prepare_reference_embedding_batches(
            df_ref,
            embedding_batch_size=hierarchy_settings.index_build.embedding_batch_size,
            embedding_cache_dir=embedding_cache_dir,
        )
        if dim is None:
            dim = ref_dim

    if dim is None:
        probe = get_embedding_vectors(["probe"])
        dim = int(probe["embeddings"].shape[1])
    assert dim is not None
    logger.info("Embedding dimension: %d", dim)

    conn = _pg_connect()
    try:
        create_tables(conn, dim=dim)

        if build_attribute:
            conn = upsert_attribute_index(
                conn,
                attr_batch_files,
                upload_batch_size=hierarchy_settings.index_build.embedding_batch_size,
                rebuild=rebuild,
            )

        if build_reference:
            conn = upsert_reference_index(
                conn,
                ref_batch_files,
                upload_batch_size=hierarchy_settings.index_build.embedding_batch_size,
                rebuild=rebuild,
            )

        create_embedding_indexes(
            conn,
            build_attribute=build_attribute,
            build_reference=build_reference,
        )
    finally:
        conn.close()

    logger.info("Index build complete.")
