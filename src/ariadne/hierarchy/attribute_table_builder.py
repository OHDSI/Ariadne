"""SNOMED attribute table builder for pgvector.

Creates/populates PostgreSQL table ``{VOCAB_SCHEMA}.snomed_attribute`` with
attribute-value concept rows and their embeddings.`
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

logger = logging.getLogger(__name__)

def _pg_connect() -> psycopg.Connection:
    """Return a psycopg connection with pgvector adapters registered."""
    conn_str = get_environment_variable("VOCAB_CONNECTION_STRING")
    conn_str = conn_str.replace("+psycopg", "").replace("+psycopg2", "")
    conn = psycopg.connect(conn_str)
    register_vector(conn)
    return conn


def create_table(conn: psycopg.Connection, dim: int) -> None:
    """Create ``snomed_attribute`` table.

    Safe to call when tables already exist (uses ``CREATE TABLE IF NOT EXISTS``).

    Args:
        conn: Admin psycopg connection.
        dim: Embedding dimension
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
    conn.commit()
    logger.info("Tables created in schema '%s'.", schema)


def create_embedding_index(conn: psycopg.Connection) -> None:
    """Create the HNSW embedding index for ``snomed_attribute`` if missing."""
    schema = get_environment_variable("VOCAB_SCHEMA")
    with conn.cursor() as cur:
            cur.execute(sql.SQL("""
                CREATE INDEX IF NOT EXISTS snomed_attribute_embedding_idx
                ON {schema}.snomed_attribute
                USING hnsw (embedding halfvec_cosine_ops)
            """).format(schema=sql.Identifier(schema)))
    conn.commit()
    logger.info("Embedding indexes created for attribute table")


def check_populated(conn: psycopg.Connection, table: str) -> bool:
    """Return True if *table* already contains at least one row.

    Args:
        conn: psycopg connection (read access is enough).
        table: Table name to check.

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


def load_attributes_from_db() -> pd.DataFrame:
    """Load SNOMED attribute value rows from the OMOP vocabulary DB."""
    engine = create_engine(get_environment_variable("VOCAB_CONNECTION_STRING"))
    schema = get_environment_variable("VOCAB_SCHEMA")
    rel_list = ", ".join(f"'{r}'" for r in _DEFAULT_SNOMED_RELATIONSHIPS)
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
    """Create parquet checkpoints for attribute embeddings.

    Embeddings are generated once per unique ``(concept_id, concept_name)`` pair,
    then mapped back to all matching attribute rows in the input frame.
    """
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


def insert_attributes(
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


def build_attribute_table(
    hierarchy_settings: HierarchySettings,
    if_exists: Literal["append", "skip", "rebuild"] = "append"
) -> None:
    """Build the pgvector SNOMED attribute table from OMOP relationships.

    Args:
        hierarchy_settings: Hierarchy settings containing ``index_build`` defaults.
        if_exists: Behavior when target tables already contain rows:
            - ``"append"``: insert without truncating.
            - ``"skip"``: skip populated tables.
            - ``"rebuild"``: truncate first, then insert.
    """
    if if_exists not in {"append", "skip", "rebuild"}:
        raise ValueError(f"Unsupported if_exists mode: {if_exists}")

    embedding_cache_dir = Path(hierarchy_settings.index_build.embedding_cache_folder)
    rebuild = if_exists == "rebuild"

    embedding_cache_dir.mkdir(parents=True, exist_ok=True)

    if if_exists == "skip":
        conn = _pg_connect()
        try:
            if check_populated(conn, "snomed_attribute"):
                logger.info("snomed_attribute already populated — skipping.")
                return
        finally:
            conn.close()

    attr_batch_files: list[Path] = []
    dim: int | None = None

    # Build embeddings first (with parquet checkpoints), then create/upload tables.
    df_attr = load_attributes_from_db()
    attr_batch_files, attr_dim = _prepare_attribute_embedding_batches(
        df_attr,
        embedding_batch_size=hierarchy_settings.index_build.embedding_batch_size,
        embedding_cache_dir=embedding_cache_dir,
    )
    dim = attr_dim if dim is None else dim

    if dim is None:
        probe = get_embedding_vectors(["probe"])
        dim = int(probe["embeddings"].shape[1])
    assert dim is not None
    logger.info("Embedding dimension: %d", dim)

    conn = _pg_connect()
    try:
        create_table(conn, dim=dim)

        conn = insert_attributes(
            conn,
            attr_batch_files,
            upload_batch_size=hierarchy_settings.index_build.embedding_batch_size,
            rebuild=rebuild,
        )

        create_embedding_index(
            conn
        )
    finally:
        conn.close()

    logger.info("Index build complete.")
