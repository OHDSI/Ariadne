"""SNOMED attribute and reference index builder for pgvector.

Creates/populates two PostgreSQL tables in VOCAB_SCHEMA:
    snomed_attribute   — attribute value concepts with embeddings.
    snomed_reference   — 10 K sampled SNOMED disorder terms with embeddings.

This module is the canonical package-level version of sandbox/build_pg_indexes.py.
The sandbox script is kept as a standalone convenience but this module is what
the CLI (``python -m ariadne.hierarchy build-index``) calls.

Usage::

    python -m ariadne.hierarchy build-index            # build if tables empty
    python -m ariadne.hierarchy build-index --check    # skip if already populated
    python -m ariadne.hierarchy build-index --rebuild  # truncate and rebuild
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

import numpy as np
import pandas as pd
import psycopg
from psycopg import sql
from pgvector.psycopg import register_vector
from sqlalchemy import create_engine

from ariadne.utils.settings import _DEFAULT_SNOMED_RELATIONSHIPS
from ariadne.utils.gen_ai_api import get_embedding_vectors
from ariadne.utils.utils import get_environment_variable

logger = logging.getLogger(__name__)

# Re-use the canonical list from config.py (single source of truth)
SNOMED_RELATIONSHIPS: list[str] = list(_DEFAULT_SNOMED_RELATIONSHIPS)

REFERENCE_SAMPLE_SIZE = 10_000
EMBEDDING_BATCH_SIZE = 500


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _pg_connect() -> psycopg.Connection:
    """Return a psycopg connection (with pgvector registered) using the admin DSN."""
    conn_str = get_environment_variable("VOCAB_CONNECTION_STRING_ADM")
    conn_str = conn_str.replace("+psycopg", "").replace("+psycopg2", "")
    conn = psycopg.connect(conn_str)
    register_vector(conn)
    return conn


def _vocab_schema() -> str:
    return get_environment_variable("VOCAB_SCHEMA")


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

def create_tables(conn: psycopg.Connection, dim: int = 3072) -> None:
    """Create ``snomed_attribute`` and ``snomed_reference`` tables with HNSW indexes.

    Safe to call when tables already exist (uses ``CREATE TABLE IF NOT EXISTS``).

    Args:
        conn: Admin psycopg connection.
        dim: Embedding dimension (default 3072 for text-embedding-3-large).
    """
    schema = _vocab_schema()
    with conn.cursor() as cur:
        cur.execute(sql.SQL("""
            CREATE TABLE IF NOT EXISTS {schema}.snomed_attribute (
                id                 SERIAL PRIMARY KEY,
                concept_id         INTEGER       NOT NULL,
                concept_code       VARCHAR(255)  NOT NULL,
                concept_name       VARCHAR(255)  NOT NULL,
                attribute_category VARCHAR(255)  NOT NULL,
                embedding          vector({dim})
            )
        """).format(schema=sql.Identifier(schema), dim=sql.Literal(dim)))
        cur.execute(sql.SQL("""
            CREATE INDEX IF NOT EXISTS snomed_attribute_embedding_idx
            ON {schema}.snomed_attribute
            USING hnsw ((embedding::halfvec({dim})) halfvec_cosine_ops)
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
                embedding          vector({dim})
            )
        """).format(schema=sql.Identifier(schema), dim=sql.Literal(dim)))
        cur.execute(sql.SQL("""
            CREATE INDEX IF NOT EXISTS snomed_reference_embedding_idx
            ON {schema}.snomed_reference
            USING hnsw ((embedding::halfvec({dim})) halfvec_cosine_ops)
        """).format(schema=sql.Identifier(schema), dim=sql.Literal(dim)))

    conn.commit()
    logger.info("Tables created/verified in schema '%s'.", schema)


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
    schema = _vocab_schema()
    with conn.cursor() as cur:
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
        SELECT c2.concept_id, c2.concept_code, c2.concept_name,
               relationship_name AS attribute_category
        FROM {schema}.concept c
        JOIN {schema}.concept_relationship cr
            ON  c.concept_id   = cr.concept_id_1
            AND c.vocabulary_id = 'SNOMED'
            AND c.standard_concept = 'S'
            AND cr.relationship_id IN ({rel_list})
            AND cr.invalid_reason IS NULL
        JOIN {schema}.concept c2 ON c2.concept_id = cr.concept_id_2
        JOIN {schema}.relationship r ON cr.relationship_id = r.relationship_id
    """, engine)
    logger.info("Loaded %d attribute rows from DB.", len(df))
    return df


def load_reference_from_db(sample_size: int = REFERENCE_SAMPLE_SIZE) -> pd.DataFrame:
    """Load and sample SNOMED reference rows from the OMOP vocabulary DB.

    Args:
        sample_size: Number of unique source concepts to include (default 10 000).
    """
    engine = create_engine(get_environment_variable("VOCAB_CONNECTION_STRING"))
    schema = get_environment_variable("VOCAB_SCHEMA")
    rel_list = ", ".join(f"'{r}'" for r in SNOMED_RELATIONSHIPS)
    df = pd.read_sql(f"""
        SELECT c.concept_id  AS concept_id_1,  c.concept_code  AS concept_code_1,
               c.concept_name AS concept_name_1,
               c2.concept_id  AS concept_id_2,  c2.concept_code AS concept_code_2,
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


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def _embed_texts(texts: list[str]) -> tuple[np.ndarray, float]:
    """Embed *texts* in batches of EMBEDDING_BATCH_SIZE.

    Returns:
        ``(embeddings, total_cost_usd)`` where *embeddings* has shape ``[N, dim]``.
    """
    all_vecs: list[np.ndarray] = []
    total_cost = 0.0
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i: i + EMBEDDING_BATCH_SIZE]
        result = get_embedding_vectors(batch)
        all_vecs.append(result["embeddings"])
        total_cost += result["usage"]["total_cost_usd"]
        logger.info(
            "  Embedded %d/%d texts ($%.4f so far)",
            min(i + EMBEDDING_BATCH_SIZE, len(texts)),
            len(texts),
            total_cost,
        )
    return np.vstack(all_vecs), total_cost


# ---------------------------------------------------------------------------
# Upsert functions
# ---------------------------------------------------------------------------

def upsert_attribute_index(
    conn: psycopg.Connection,
    df: pd.DataFrame,
    rebuild: bool = False,
) -> psycopg.Connection:
    """Embed attribute concept names and insert rows into ``snomed_attribute``.

    Args:
        conn: Admin psycopg connection (returned — may be replaced on retry).
        df: DataFrame from :func:`load_attributes_from_db`.
        rebuild: When True, truncate the table first.

    Returns:
        The (possibly reconnected) psycopg connection.
    """
    schema = _vocab_schema()
    if rebuild:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("TRUNCATE TABLE {schema}.snomed_attribute RESTART IDENTITY")
                .format(schema=sql.Identifier(schema))
            )
        conn.commit()
        logger.info("Truncated snomed_attribute.")

    logger.info("Embedding %d attribute concept names…", len(df))
    embeddings, cost = _embed_texts(df["concept_name"].tolist())
    logger.info("Attribute embeddings done.  Cost: $%.4f", cost)

    rows = [
        (int(r["concept_id"]), str(r["concept_code"]), str(r["concept_name"]),
         str(r["attribute_category"]), embeddings[i].tolist())
        for i, (_, r) in enumerate(df.iterrows())
    ]
    _insert_sql_tpl = (
        "INSERT INTO {schema}.snomed_attribute "
        "(concept_id, concept_code, concept_name, attribute_category, embedding) "
        "VALUES (%s, %s, %s, %s, %s)"
    )
    insert_sql = sql.SQL(_insert_sql_tpl).format(schema=sql.Identifier(schema))

    chunk_size = 200
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i: i + chunk_size]
        for attempt in range(3):
            try:
                with conn.cursor() as cur:
                    cur.executemany(insert_sql, chunk)
                conn.commit()
                break
            except Exception as exc:
                logger.warning("Chunk %d attempt %d failed (%s), reconnecting…",
                               i // chunk_size + 1, attempt + 1, exc)
                try:
                    conn.close()
                except Exception:
                    pass
                conn = _pg_connect()
                insert_sql = sql.SQL(_insert_sql_tpl).format(schema=sql.Identifier(schema))
        logger.info("  Inserted rows %d–%d / %d", i + 1, min(i + chunk_size, len(rows)), len(rows))

    logger.info("Inserted %d rows into %s.snomed_attribute.", len(rows), schema)
    return conn


def upsert_reference_index(
    conn: psycopg.Connection,
    df: pd.DataFrame,
    rebuild: bool = False,
) -> psycopg.Connection:
    """Embed unique source-concept names and insert rows into ``snomed_reference``.

    Args:
        conn: Admin psycopg connection (returned — may be replaced on retry).
        df: DataFrame from :func:`load_reference_from_db`.
        rebuild: When True, truncate the table first.

    Returns:
        The (possibly reconnected) psycopg connection.
    """
    schema = _vocab_schema()
    if rebuild:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("TRUNCATE TABLE {schema}.snomed_reference RESTART IDENTITY")
                .format(schema=sql.Identifier(schema))
            )
        conn.commit()
        logger.info("Truncated snomed_reference.")

    unique_terms = df[["concept_id_1", "concept_name_1"]].drop_duplicates().reset_index(drop=True)
    logger.info("Embedding %d unique reference source term names…", len(unique_terms))
    embeddings, cost = _embed_texts(unique_terms["concept_name_1"].tolist())
    logger.info("Reference embeddings done.  Cost: $%.4f", cost)

    id_to_emb = {int(r["concept_id_1"]): embeddings[i] for i, (_, r) in enumerate(unique_terms.iterrows())}

    rows = [
        (int(r["concept_id_1"]), str(r["concept_name_1"]),
         int(r["concept_id_2"]), str(r["concept_code_2"]), str(r["concept_name_2"]),
         str(r["attribute_category"]), id_to_emb[int(r["concept_id_1"])].tolist())
        for _, r in df.iterrows()
    ]
    _insert_sql_tpl = (
        "INSERT INTO {schema}.snomed_reference "
        "(concept_id_1, concept_name_1, concept_id_2, concept_code_2, "
        " concept_name_2, attribute_category, embedding) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)"
    )
    insert_sql = sql.SQL(_insert_sql_tpl).format(schema=sql.Identifier(schema))

    chunk_size = 200
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i: i + chunk_size]
        for attempt in range(3):
            try:
                with conn.cursor() as cur:
                    cur.executemany(insert_sql, chunk)
                conn.commit()
                break
            except Exception as exc:
                logger.warning("Chunk %d attempt %d failed (%s), reconnecting…",
                               i // chunk_size + 1, attempt + 1, exc)
                try:
                    conn.close()
                except Exception:
                    pass
                conn = _pg_connect()
                insert_sql = sql.SQL(_insert_sql_tpl).format(schema=sql.Identifier(schema))
        logger.info("  Inserted rows %d–%d / %d", i + 1, min(i + chunk_size, len(rows)), len(rows))

    logger.info("Inserted %d rows into %s.snomed_reference.", len(rows), schema)
    return conn


# ---------------------------------------------------------------------------
# High-level build entry point
# ---------------------------------------------------------------------------

def build(
    rebuild: bool = False,
    attributes_only: bool = False,
    reference_only: bool = False,
    check: bool = False,
    reference_sample_size: int = REFERENCE_SAMPLE_SIZE,
) -> None:
    """Build (or rebuild) the pgvector SNOMED indexes.

    Args:
        rebuild: Truncate existing data before inserting.
        attributes_only: Only build ``snomed_attribute``.
        reference_only: Only build ``snomed_reference``.
        check: Skip any table that is already populated (takes precedence over
            *rebuild* only when the table is non-empty).
        reference_sample_size: Number of unique source concepts for the
            reference index (default 10 000).
    """
    from dotenv import load_dotenv
    from ariadne.utils.utils import get_project_root
    load_dotenv(get_project_root() / ".env")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    conn = _pg_connect()

    # Determine embedding dimension from a probe call
    probe = get_embedding_vectors(["probe"])
    dim = probe["embeddings"].shape[1]
    logger.info("Embedding dimension: %d", dim)

    create_tables(conn, dim=dim)

    if not reference_only:
        if check and check_populated(conn, "snomed_attribute"):
            logger.info("snomed_attribute already populated — skipping (--check).")
        else:
            df_attr = load_attributes_from_db()
            conn = upsert_attribute_index(conn, df_attr, rebuild=rebuild)

    if not attributes_only:
        if check and check_populated(conn, "snomed_reference"):
            logger.info("snomed_reference already populated — skipping (--check).")
        else:
            df_ref = load_reference_from_db(sample_size=reference_sample_size)
            conn = upsert_reference_index(conn, df_ref, rebuild=rebuild)

    conn.close()
    logger.info("Index build complete.")
