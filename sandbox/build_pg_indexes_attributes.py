"""
Build and store SNOMED attribute and reference indexes in PostgreSQL using pgvector.

Creates two tables in the VOCAB_SCHEMA:
  - snomed_attribute   : attribute vocabulary with embeddings
  - snomed_reference   : reference SNOMED terms with embeddings

Usage:
    python build_pg_indexes.py               # build from scratch
    python build_pg_indexes.py --rebuild     # truncate and rebuild

Environment variables (loaded from .env):
    VOCAB_CONNECTION_STRING_ADM   psycopg connection string (write access, for table creation)
    VOCAB_CONNECTION_STRING       SQLAlchemy connection string (read access, for data loading)
    VOCAB_SCHEMA                  OMOP vocabulary schema (tables will be created here)
    EMBEDDING_MODEL               Embedding model name (via GENAI_PROVIDER routing)
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from psycopg import sql
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sqlalchemy import create_engine

# Allow running from the sandbox directory (works in scripts and IPython/Jupyter)
_this_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
sys.path.insert(0, os.path.join(_this_dir, "..", "src"))

from ariadne.utils.gen_ai_api import get_embedding_vectors
from ariadne.utils.utils import get_environment_variable, get_project_root

load_dotenv(get_project_root() / ".env")

# SNOMED relationship IDs to include (must match working hierarchy.py)
SNOMED_RELATIONSHIPS = [
    "Has asso morph", "Has finding site", "Has causative agent", "Has clinical course",
    "Has finding context", "Has interpretation", "Has interprets", "Has occurrence",
    "Has pathology", "Has relat context", "Has severity", "Has temporal context",
    "Finding asso with",
]

REFERENCE_SAMPLE_SIZE = 10_000
EMBEDDING_BATCH_SIZE = 500
ATTR_EMBEDDINGS_CHECKPOINT = Path("/tmp/snomed_attr_embeddings.pkl")
REF_EMBEDDINGS_CHECKPOINT = Path("/tmp/snomed_ref_embeddings.pkl")


# ---------------------------------------------------------------------------
# Helper: raw psycopg connection (pgvector needs this, not SQLAlchemy)
# ---------------------------------------------------------------------------

def _pg_connect() -> psycopg.Connection:
    conn_str = get_environment_variable("VOCAB_CONNECTION_STRING_ADM")
    # Strip SQLAlchemy driver prefix if present
    conn_str = conn_str.replace("+psycopg", "").replace("+psycopg2", "")
    # Convert SQLAlchemy URL to psycopg conninfo if needed
    if conn_str.startswith("postgresql://") or conn_str.startswith("postgres://"):
        conn = psycopg.connect(conn_str)
    else:
        conn = psycopg.connect(conn_str)
    register_vector(conn)
    return conn


def _vocab_schema() -> str:
    return get_environment_variable("VOCAB_SCHEMA")


def _reader_user() -> str | None:
    """Extract the username from VOCAB_CONNECTION_STRING (the read-only connection)."""
    from urllib.parse import urlparse, unquote
    try:
        conn_str = get_environment_variable("VOCAB_CONNECTION_STRING")
        parsed = urlparse(conn_str)
        return unquote(parsed.username) if parsed.username else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

def create_tables(conn: psycopg.Connection, dim: int) -> None:
    """Create the snomed_attribute and snomed_reference tables (and HNSW indexes)."""
    schema = _vocab_schema()
    with conn.cursor() as cur:

        # snomed_attribute
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

        # snomed_reference
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
    print(f"Tables created/verified in schema '{schema}'.")


# ---------------------------------------------------------------------------
# Data loading from OMOP vocab
# ---------------------------------------------------------------------------

def load_attributes_from_db() -> pd.DataFrame:
    engine = create_engine(get_environment_variable("VOCAB_CONNECTION_STRING"))
    schema = get_environment_variable("VOCAB_SCHEMA")
    rel_list = ", ".join(f"'{r}'" for r in SNOMED_RELATIONSHIPS)
    df = pd.read_sql(f"""
        SELECT c2.concept_id, c2.concept_code, c2.concept_name,
               relationship_name AS attribute_category
        FROM {schema}.concept c
        JOIN {schema}.concept_relationship cr
            ON c.concept_id = cr.concept_id_1
            AND c.vocabulary_id = 'SNOMED' AND c.standard_concept = 'S'
            AND cr.relationship_id IN ({rel_list})
            AND cr.invalid_reason IS NULL
        JOIN {schema}.concept c2 ON c2.concept_id = cr.concept_id_2
        JOIN {schema}.relationship r ON cr.relationship_id = r.relationship_id
    """, engine)
    print(f"Loaded {len(df):,} attribute rows from DB.")
    return df


def load_reference_from_db(sample_size: int = REFERENCE_SAMPLE_SIZE) -> pd.DataFrame:
    engine = create_engine(get_environment_variable("VOCAB_CONNECTION_STRING"))
    schema = get_environment_variable("VOCAB_SCHEMA")
    rel_list = ", ".join(f"'{r}'" for r in SNOMED_RELATIONSHIPS)
    df = pd.read_sql(f"""
        SELECT c.concept_id AS concept_id_1, c.concept_code AS concept_code_1,
               c.concept_name AS concept_name_1,
               c2.concept_id AS concept_id_2, c2.concept_code AS concept_code_2,
               c2.concept_name AS concept_name_2,
               relationship_name AS attribute_category
        FROM {schema}.concept c
        JOIN {schema}.concept_relationship cr
            ON c.concept_id = cr.concept_id_1
            AND c.vocabulary_id = 'SNOMED' AND c.standard_concept = 'S'
            AND cr.relationship_id IN ({rel_list})
            AND cr.invalid_reason IS NULL
        JOIN {schema}.concept c2 ON c2.concept_id = cr.concept_id_2
        JOIN {schema}.relationship r ON cr.relationship_id = r.relationship_id
        LIMIT {sample_size * 10}
    """, engine)
    # Sample to desired number of unique concept_id_1 terms
    unique_ids = df["concept_id_1"].unique()
    if len(unique_ids) > sample_size:
        rng = np.random.default_rng(42)
        sampled_ids = rng.choice(unique_ids, size=sample_size, replace=False)
        df = df[df["concept_id_1"].isin(sampled_ids)]
    print(f"Loaded {len(df):,} reference rows ({df['concept_id_1'].nunique():,} unique terms) from DB.")
    return df


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _embed_texts(texts: list) -> tuple[np.ndarray, float]:
    """Embed a list of texts in batches. Returns (ndarray of shape [N, dim], total_cost)."""
    all_vecs = []
    total_cost = 0.0
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i: i + EMBEDDING_BATCH_SIZE]
        result = get_embedding_vectors(batch)
        all_vecs.append(result["embeddings"])
        total_cost += result["usage"]["total_cost_usd"]
        print(f"  Embedded {min(i + EMBEDDING_BATCH_SIZE, len(texts)):,}/{len(texts):,} texts "
              f"(${total_cost:.4f} so far)")
    return np.vstack(all_vecs), total_cost


def _save_embeddings_checkpoint(embeddings: np.ndarray, df: pd.DataFrame, checkpoint_path: Path) -> None:
    """Save embeddings and dataframe to checkpoint file for resuming."""
    with open(checkpoint_path, "wb") as f:
        pickle.dump({"embeddings": embeddings, "df": df}, f)
    print(f"  Saved embeddings checkpoint: {checkpoint_path}")


def _load_embeddings_checkpoint(checkpoint_path: Path) -> tuple[np.ndarray, pd.DataFrame] | None:
    """Load embeddings from checkpoint if it exists."""
    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            data = pickle.load(f)
        print(f"  Loaded embeddings from checkpoint: {checkpoint_path}")
        return data["embeddings"], data["df"]
    return None


# ---------------------------------------------------------------------------
# Upsert functions
# ---------------------------------------------------------------------------

def upsert_attribute_index(conn: psycopg.Connection, df: pd.DataFrame, rebuild: bool = False, skip_embedding: bool = False) -> psycopg.Connection:
    schema = _vocab_schema()
    if rebuild:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("TRUNCATE TABLE {schema}.snomed_attribute RESTART IDENTITY")
                        .format(schema=sql.Identifier(schema)))
        conn.commit()
        print("Truncated snomed_attribute.")
        # Clear checkpoint on rebuild
        if ATTR_EMBEDDINGS_CHECKPOINT.exists():
            ATTR_EMBEDDINGS_CHECKPOINT.unlink()

    # Try to load embeddings from checkpoint
    if skip_embedding or ATTR_EMBEDDINGS_CHECKPOINT.exists():
        result = _load_embeddings_checkpoint(ATTR_EMBEDDINGS_CHECKPOINT)
        if result:
            embeddings, df = result
        else:
            print(f"Embedding {len(df):,} attribute concept names...")
            embeddings, cost = _embed_texts(df["concept_name"].tolist())
            print(f"Attribute embeddings done. Cost: ${cost:.4f}")
            _save_embeddings_checkpoint(embeddings, df, ATTR_EMBEDDINGS_CHECKPOINT)
    else:
        print(f"Embedding {len(df):,} attribute concept names...")
        embeddings, cost = _embed_texts(df["concept_name"].tolist())
        print(f"Attribute embeddings done. Cost: ${cost:.4f}")
        _save_embeddings_checkpoint(embeddings, df, ATTR_EMBEDDINGS_CHECKPOINT)

    rows = [
        (
            int(row["concept_id"]),
            str(row["concept_code"]),
            str(row["concept_name"]),
            str(row["attribute_category"]),
            embeddings[i].tolist(),
        )
        for i, (_, row) in enumerate(df.iterrows())
    ]

    _attr_insert = (  # type: ignore[assignment]
        "INSERT INTO {schema}.snomed_attribute "
        "(concept_id, concept_code, concept_name, attribute_category, embedding) "
        "VALUES (%s, %s, %s, %s, %s)"
    )
    insert_sql = sql.SQL(_attr_insert).format(schema=sql.Identifier(schema))  # type: ignore[arg-type]

    insert_chunk_size = 200
    for i in range(0, len(rows), insert_chunk_size):
        chunk = rows[i: i + insert_chunk_size]
        for attempt in range(3):
            try:
                with conn.cursor() as cur:
                    cur.executemany(insert_sql, chunk)
                conn.commit()
                break
            except Exception as e:
                print(f"  Chunk {i // insert_chunk_size + 1}: attempt {attempt + 1} failed ({e}), reconnecting...")
                try:
                    conn.close()
                except Exception:
                    pass
                conn = _pg_connect()
                insert_sql = sql.SQL(_attr_insert).format(schema=sql.Identifier(schema))  # type: ignore[arg-type]
        print(f"  Inserted rows {i + 1}–{min(i + insert_chunk_size, len(rows)):,} / {len(rows):,}")
    print(f"Inserted {len(rows):,} rows into {schema}.snomed_attribute.")
    return conn


def upsert_reference_index(conn: psycopg.Connection, df: pd.DataFrame, rebuild: bool = False) -> psycopg.Connection:
    schema = _vocab_schema()
    if rebuild:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("TRUNCATE TABLE {schema}.snomed_reference RESTART IDENTITY")
                        .format(schema=sql.Identifier(schema)))
        conn.commit()
        print("Truncated snomed_reference.")

    # Embed unique concept_name_1 values, then broadcast to all rows
    unique_terms = df[["concept_id_1", "concept_name_1"]].drop_duplicates().reset_index(drop=True)
    print(f"Embedding {len(unique_terms):,} unique reference term names...")
    embeddings, cost = _embed_texts(unique_terms["concept_name_1"].tolist())
    print(f"Reference embeddings done. Cost: ${cost:.4f}")

    id_to_embedding = {
        int(row["concept_id_1"]): embeddings[i]
        for i, (_, row) in enumerate(unique_terms.iterrows())
    }

    rows = [
        (
            int(row["concept_id_1"]),
            str(row["concept_name_1"]),
            int(row["concept_id_2"]),
            str(row["concept_code_2"]),
            str(row["concept_name_2"]),
            str(row["attribute_category"]),
            id_to_embedding[int(row["concept_id_1"])].tolist(),
        )
        for _, row in df.iterrows()
    ]

    _ref_insert = (  # type: ignore[assignment]
        "INSERT INTO {schema}.snomed_reference "
        "(concept_id_1, concept_name_1, concept_id_2, concept_code_2, "
        " concept_name_2, attribute_category, embedding) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)"
    )
    insert_sql = sql.SQL(_ref_insert).format(schema=sql.Identifier(schema))  # type: ignore[arg-type]

    insert_chunk_size = 200
    for i in range(0, len(rows), insert_chunk_size):
        chunk = rows[i: i + insert_chunk_size]
        for attempt in range(3):
            try:
                with conn.cursor() as cur:
                    cur.executemany(insert_sql, chunk)
                conn.commit()
                break
            except Exception as e:
                print(f"  Chunk {i // insert_chunk_size + 1}: attempt {attempt + 1} failed ({e}), reconnecting...")
                try:
                    conn.close()
                except Exception:
                    pass
                conn = _pg_connect()
                insert_sql = sql.SQL(_ref_insert).format(schema=sql.Identifier(schema))  # type: ignore[arg-type]
        print(f"  Inserted rows {i + 1}–{min(i + insert_chunk_size, len(rows)):,} / {len(rows):,}")
    print(f"Inserted {len(rows):,} rows into {schema}.snomed_reference.")
    return conn


def load_attribute_index_from_pg(conn: psycopg.Connection) -> dict:
    """
    Load snomed_attribute table and return a dict matching the shape produced by
    build_attribute_index():
        { 'dataframe': pd.DataFrame, 'embeddings': np.ndarray }
    """
    schema = _vocab_schema()
    _attr_select = (  # type: ignore[assignment]
        "SELECT concept_id, concept_code, concept_name, attribute_category, embedding "
        "FROM {schema}.snomed_attribute ORDER BY id"
    )
    with conn.cursor() as cur:
        cur.execute(sql.SQL(_attr_select).format(schema=sql.Identifier(schema)))  # type: ignore[arg-type]
        rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=["concept_id", "concept_code", "concept_name",
                                     "attribute_category", "embedding"])
    embeddings = np.array(df["embedding"].tolist(), dtype=np.float32)
    df = df.drop(columns=["embedding"])
    print(f"Loaded {len(df):,} rows from {schema}.snomed_attribute.")
    return {"dataframe": df, "embeddings": embeddings}


def load_reference_index_from_pg(conn: psycopg.Connection) -> dict:
    """
    Load snomed_reference table and return a dict matching the shape produced by
    build_reference_index():
        { 'terms': pd.DataFrame, 'embeddings': np.ndarray, 'term_attributes': dict }
    """
    schema = _vocab_schema()
    _ref_select = (  # type: ignore[assignment]
        "SELECT concept_id_1, concept_name_1, concept_id_2, concept_code_2, "
        "       concept_name_2, attribute_category, embedding "
        "FROM {schema}.snomed_reference ORDER BY id"
    )
    with conn.cursor() as cur:
        cur.execute(sql.SQL(_ref_select).format(schema=sql.Identifier(schema)))  # type: ignore[arg-type]
        rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=["concept_id_1", "concept_name_1", "concept_id_2",
                                     "concept_code_2", "concept_name_2",
                                     "attribute_category", "embedding"])

    # Unique terms with one embedding per term (first occurrence)
    unique_terms = (
        df[["concept_id_1", "concept_name_1", "embedding"]]
        .drop_duplicates(subset=["concept_id_1"])
        .reset_index(drop=True)
    )
    embeddings = np.array(unique_terms["embedding"].tolist(), dtype=np.float32)
    unique_terms = unique_terms.drop(columns=["embedding"])

    # Group attribute rows by (concept_id_1, concept_name_1)
    term_attributes = (
        df.groupby(["concept_id_1", "concept_name_1"])
        .apply(
            lambda x: x[["concept_id_2", "concept_name_2", "attribute_category"]].to_dict("records"),
            include_groups=False,
        )
        .to_dict()
    )

    print(f"Loaded {len(unique_terms):,} unique reference terms from {schema}.snomed_reference.")
    return {"terms": unique_terms, "embeddings": embeddings, "term_attributes": term_attributes}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build SNOMED pgvector indexes in PostgreSQL.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Truncate existing tables before inserting.")
    parser.add_argument("--attributes-only", action="store_true",
                        help="Only rebuild the snomed_attribute table.")
    parser.add_argument("--reference-only", action="store_true",
                        help="Only rebuild the snomed_reference table.")
    parser.add_argument("--skip-embedding", action="store_true",
                        help="Skip embedding and use cached checkpoint (for resuming failed inserts).")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear cached embeddings before running.")
    args, _ = parser.parse_known_args()
    
    if args.clear_cache:
        for cp in (ATTR_EMBEDDINGS_CHECKPOINT, REF_EMBEDDINGS_CHECKPOINT):
            if cp.exists():
                cp.unlink()
                print(f"Cleared {cp}")

    print("Connecting to PostgreSQL...")
    conn = _pg_connect()

    try:
        if not args.reference_only:
            print("\n--- Building snomed_attribute ---")
            if not (args.skip_embedding and ATTR_EMBEDDINGS_CHECKPOINT.exists()):
                attr_df = load_attributes_from_db()
            else:
                attr_df = None  # Will be loaded from checkpoint
            # Determine embedding dim from a single probe or checkpoint
            if ATTR_EMBEDDINGS_CHECKPOINT.exists() and args.skip_embedding:
                checkpoint_data = _load_embeddings_checkpoint(ATTR_EMBEDDINGS_CHECKPOINT)
                if checkpoint_data:
                    dim = checkpoint_data[0].shape[1]
                else:
                    probe = get_embedding_vectors([attr_df["concept_name"].iloc[0]])
                    dim = probe["embeddings"].shape[1]
            else:
                if attr_df is None:
                    attr_df = load_attributes_from_db()
                probe = get_embedding_vectors([attr_df["concept_name"].iloc[0]])
                dim = probe["embeddings"].shape[1]
            create_tables(conn, dim)
            if attr_df is None:
                _, attr_df = _load_embeddings_checkpoint(ATTR_EMBEDDINGS_CHECKPOINT)
            conn = upsert_attribute_index(conn, attr_df, rebuild=args.rebuild, skip_embedding=args.skip_embedding)

        if not args.attributes_only:
            print("\n--- Building snomed_reference ---")
            ref_df = load_reference_from_db(sample_size=REFERENCE_SAMPLE_SIZE)
            if args.reference_only:
                # Tables might not exist yet — ensure they are created
                probe = get_embedding_vectors([ref_df["concept_name_1"].iloc[0]])
                dim = probe["embeddings"].shape[1]
                create_tables(conn, dim)
            conn = upsert_reference_index(conn, ref_df, rebuild=args.rebuild)

        # Grant SELECT to the reader role so VOCAB_CONNECTION_STRING works
        schema = _vocab_schema()
        reader_user = _reader_user()
        if reader_user:
            print(f"\n--- Granting SELECT to '{reader_user}' ---")
            with conn.cursor() as cur:
                for tbl in ("snomed_attribute", "snomed_reference"):
                    cur.execute(sql.SQL("GRANT SELECT ON {schema}.{tbl} TO {user}").format(
                        schema=sql.Identifier(schema),
                        tbl=sql.Identifier(tbl),
                        user=sql.Identifier(reader_user),
                    ))
            conn.commit()
            print("  Grants applied.")

        # Verify
        print("\n--- Verification ---")
        with conn.cursor() as cur:
            for tbl in ("snomed_attribute", "snomed_reference"):
                cur.execute(sql.SQL("SELECT COUNT(*) FROM {schema}.{tbl}")
                            .format(schema=sql.Identifier(schema), tbl=sql.Identifier(tbl)))
                count = cur.fetchone()[0]
                print(f"  {schema}.{tbl}: {count:,} rows")

    finally:
        conn.close()

    print("\nDone.")


if __name__ == "__main__":
    main()

