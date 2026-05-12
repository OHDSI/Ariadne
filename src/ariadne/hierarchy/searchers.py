"""pgvector-backed SNOMED searchers for attribute and reference retrieval.

Classes:
    AbstractSnomedSearcher — ABC enforcing lifecycle (close / cost / context manager).
    SnomedAttributeSearcher — queries ``snomed_attribute`` for candidate values.
    SnomedReferenceSearcher — queries ``snomed_reference`` for few-shot examples.
    SnomedReferenceConceptVectorSearcher — builds few-shot examples from
        ``concept`` + ``concept_relationship`` + ``relationship`` +
        configured vector table (e.g., ``concept_vector``) without requiring
        precomputed ``snomed_reference``.
"""

import logging
from abc import ABC, abstractmethod

import pandas as pd
import psycopg
from psycopg import sql
from pgvector.psycopg import register_vector

from ariadne.hierarchy.types import (
    ReferenceSearchResult,
    SearchBatchResult,
    SearchResult,
)
from ariadne.utils.gen_ai_api import get_embedding_vectors
from ariadne.utils.settings import HierarchySettings
from ariadne.utils.utils import get_environment_variable
from ariadne.vector_search.pgvector_concept_searcher import PgvectorConceptSearcher

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attribute mappings (SNOMED category ↔ pipeline key)
# ---------------------------------------------------------------------------

ATTR_KEY_TO_SNOMED_CATEGORY: dict[str, str] = {
    "associated_morphology": "Has associated morphology (SNOMED)",
    "finding_site": "Has finding site (SNOMED)",
    "causative_agent": "Has causative agent (SNOMED)",
    "clinical_course": "Has clinical course (SNOMED)",
    "finding_context": "Has finding context (SNOMED)",
    "interpretation": "Has interpretation (SNOMED)",
    "interprets": "Has interprets (SNOMED)",
    "occurrence": "Has occurrence (SNOMED)",
    "pathological_process": "Has pathological process (SNOMED)",
    "severity": "Has severity (SNOMED)",
    "subject_relationship_context": "Has subject relationship context (SNOMED)",
    "temporal_context": "Has temporal context (SNOMED)",
    "finding_asso_with": "Finding asso with (SNOMED)",
    "associated_with": "Finding asso with (SNOMED)",
    "finding_associated_with": "Finding asso with (SNOMED)",
}

ATTR_KEY_TO_GS_CATEGORY: dict[str, str] = {
    "associated_morphology": "Has asso morph",
    "finding_site": "Has finding site",
    "causative_agent": "Has causative agent",
    "clinical_course": "Has clinical course",
    "finding_context": "Has finding context",
    "interpretation": "Has interpretation",
    "interprets": "Has interprets",
    "occurrence": "Has occurrence",
    "pathological_process": "Has pathology",
    "severity": "Has severity",
    "subject_relationship_context": "Has relat context",
    "temporal_context": "Has temporal context",
    "finding_asso_with": "Finding asso with",
    "associated_with": "Finding asso with",
    "finding_associated_with": "Finding asso with",
}


SNOMED_CATEGORY_TO_ATTR_KEY: dict[str, str] = {v: k for k, v in ATTR_KEY_TO_SNOMED_CATEGORY.items()}


# ---------------------------------------------------------------------------
# Abstract base searcher
# ---------------------------------------------------------------------------

class AbstractSnomedSearcher(ABC):
    """ABC for pgvector-backed SNOMED searchers.

    Enforces lifecycle management (``close`` / context-manager) and cost
    tracking — matching the patterns used by ``PgvectorConceptSearcher`` and
    ``LlmMapper``.

    Subclasses must implement :meth:`search`.
    """

    def __init__(self, hierarchy_settings: HierarchySettings | None = None):
        self.hierarchy_settings = hierarchy_settings
        conn_str = get_environment_variable("VOCAB_CONNECTION_STRING")
        conn_str = conn_str.replace("+psycopg", "").replace("+psycopg2", "")
        self.connection = psycopg.connect(conn_str, autocommit=True)
        try:
            register_vector(self.connection)
            with self.connection.cursor() as cur:
                cur.execute(f"SET hnsw.ef_search = {self.hierarchy_settings.retrieval.hnsw_ef_search}")
            self.schema = get_environment_variable("VOCAB_SCHEMA")
        except Exception:
            self.connection.close()
            raise
        self._cost = 0.0

    # -- abstract contract --------------------------------------------------

    @abstractmethod
    def search(self, text: str, *args, **kwargs):
        """Run a similarity search.  Signature varies by subclass."""

    # -- lifecycle ----------------------------------------------------------

    def get_total_cost(self) -> float:
        """Return accumulated embedding cost (USD)."""
        return self._cost

    def close(self):
        """Close the database connection."""
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# Attribute searcher
# ---------------------------------------------------------------------------

class SnomedAttributeSearcher(AbstractSnomedSearcher):

    def search(self, text: str, category_name: str, top_k: int | None = None) -> SearchResult:
        """Embed *text* and return the closest concepts in the given attribute category.

        Args:
            text: Free-text value to embed and search for.
            category_name: SNOMED attribute category filter.
            top_k: Maximum number of results (defaults to ``hierarchy_settings.retrieval.top_k_per_category``).

        Returns:
            SearchResult(dataframe, cost).
        """
        top_k = top_k if top_k is not None else self.hierarchy_settings.retrieval.top_k_per_category
        result = get_embedding_vectors([text])
        vector = result["embeddings"][0]
        cost = result["usage"]["total_cost_usd"]
        self._cost += cost

        query = sql.SQL("""
            WITH q AS (SELECT %s::vector AS vec)
            SELECT concept_id, concept_code, concept_name, attribute_category,
                   1 - (embedding <=> q.vec) AS similarity
            FROM {schema}.{table}, q
            WHERE attribute_category = %s
            ORDER BY embedding <=> q.vec
            LIMIT %s
        """).format(schema=sql.Identifier(self.schema), table=sql.Identifier("snomed_attribute"))
        with self.connection.cursor() as cur:
            cur.execute(query, (vector.tolist(), category_name, top_k))
            rows = cur.fetchall()

        if not rows:
            return SearchResult(pd.DataFrame(columns=["concept_id", "concept_code", "concept_name", "attribute_category", "similarity"]), cost)

        return SearchResult(pd.DataFrame(rows, columns=["concept_id", "concept_code", "concept_name", "attribute_category", "similarity"]), cost)

    def search_batch(
        self,
        texts_and_categories: list[tuple[str, str, str]],
        top_k: int | None = None,
    ) -> SearchBatchResult:
        """Embed all *texts* in a single API call and run per-category queries.

        Args:
            texts_and_categories: List of ``(attr_key, text, category_name)`` tuples.
            top_k: Number of candidates per category (defaults to ``hierarchy_settings.retrieval.top_k_per_category``).

        Returns:
            SearchBatchResult(results, total_cost).
        """
        top_k = top_k if top_k is not None else self.hierarchy_settings.retrieval.top_k_per_category
        if not texts_and_categories:
            return SearchBatchResult({}, 0.0)

        texts = [t[1] for t in texts_and_categories]
        result = get_embedding_vectors(texts)
        vectors = result["embeddings"]
        cost = result["usage"]["total_cost_usd"]
        self._cost += cost

        results: dict[str, pd.DataFrame] = {}
        for (attr_key, _text, category_name), vector in zip(texts_and_categories, vectors):
            query = sql.SQL("""
                WITH q AS (SELECT %s::vector AS vec)
                SELECT concept_id, concept_code, concept_name, attribute_category,
                       1 - (embedding <=> q.vec) AS similarity
                FROM {schema}.{table}, q
                WHERE attribute_category = %s
                ORDER BY embedding <=> q.vec
                LIMIT %s
            """).format(schema=sql.Identifier(self.schema), table=sql.Identifier("snomed_attribute"))
            with self.connection.cursor() as cur:
                cur.execute(query, (vector.tolist(), category_name, top_k))
                rows = cur.fetchall()

            if rows:
                results[attr_key] = pd.DataFrame(
                    rows, columns=["concept_id", "concept_code", "concept_name", "attribute_category", "similarity"]
                )
            else:
                results[attr_key] = pd.DataFrame(
                    columns=["concept_id", "concept_code", "concept_name", "attribute_category", "similarity"]
                )
        return SearchBatchResult(results, cost)

    def expand_via_hierarchy(
        self,
        concept_ids: list[int],
        attribute_category: str,
    ) -> pd.DataFrame:
        """Expand candidate concept IDs by 1-hop SNOMED hierarchy.

        For each *concept_id*, retrieves its immediate parents (``Is a``) and
        children (``Subsumes``) from ``concept_relationship``, then filters to
        those that exist in ``snomed_attribute`` under the same
        *attribute_category*.  No embedding API call is needed.

        Args:
            concept_ids: Seed concept IDs to expand.
            attribute_category: SNOMED attribute category filter.

        Returns:
            DataFrame with columns
            ``[concept_id, concept_name, attribute_category, similarity]``
            where *similarity* is set to ``hierarchy_settings.scoring.hierarchy_similarity``.
        """
        if not concept_ids:
            return pd.DataFrame(
                columns=["concept_id", "concept_name", "attribute_category", "similarity"]
            )

        query = sql.SQL("""
            WITH seeds AS (
                SELECT unnest(%s::int[]) AS cid
            ),
            neighbors AS (
                -- parents (seed "Is a" parent)
                SELECT DISTINCT cr.concept_id_2 AS cid
                FROM {schema}.{concept_rel} cr
                JOIN seeds s ON cr.concept_id_1 = s.cid
                WHERE cr.relationship_id = 'Is a'
                  AND cr.invalid_reason IS NULL
                UNION
                -- children (child "Is a" seed)
                SELECT DISTINCT cr.concept_id_1 AS cid
                FROM {schema}.{concept_rel} cr
                JOIN seeds s ON cr.concept_id_2 = s.cid
                WHERE cr.relationship_id = 'Is a'
                  AND cr.invalid_reason IS NULL
            )
            SELECT DISTINCT sa.concept_id, sa.concept_code, sa.concept_name, sa.attribute_category
            FROM {schema}.{snomed_attr} sa
            JOIN neighbors n ON sa.concept_id = n.cid
            WHERE sa.attribute_category = %s
              AND sa.concept_id NOT IN (SELECT cid FROM seeds)
        """).format(
            schema=sql.Identifier(self.schema),
            concept_rel=sql.Identifier("concept_relationship"),
            snomed_attr=sql.Identifier("snomed_attribute"),
        )
        params = [concept_ids, attribute_category]
        with self.connection.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        if not rows:
            return pd.DataFrame(
                columns=["concept_id", "concept_code", "concept_name", "attribute_category", "similarity"]
            )

        df = pd.DataFrame(rows, columns=["concept_id", "concept_code", "concept_name", "attribute_category"])
        df = df.drop_duplicates(subset=["concept_id"])
        df["similarity"] = self.hierarchy_settings.scoring.hierarchy_similarity
        return df


# ---------------------------------------------------------------------------
# Reference searcher
# ---------------------------------------------------------------------------

class SnomedReferenceSearcher(AbstractSnomedSearcher):

    def __init__(self, hierarchy_settings: HierarchySettings | None = None,
                 exclude_concept_ids: set[int] | None = None):
        super().__init__(hierarchy_settings)
        self._exclude_concept_ids: list[int] = sorted(exclude_concept_ids) if exclude_concept_ids else []
        if self._exclude_concept_ids:
            logger.debug(
                "SnomedReferenceSearcher: excluding %d concept IDs from results.",
                len(self._exclude_concept_ids),
            )

    def search(
        self,
        text: str,
        top_k: int | None = None,
        embedding: "np.ndarray | None" = None,
    ) -> ReferenceSearchResult:
        """Embed *text* and return similar reference terms with their attributes.

        Args:
            text: Free-text term to embed (ignored when *embedding* is supplied).
            top_k: Maximum number of reference concepts to return
                (defaults to ``hierarchy_settings.retrieval.num_reference_examples``).
            embedding: Optional precomputed embedding vector (shape ``[dim]``).
                When provided the API call is skipped and cost is recorded as
                zero.  Pass the vector produced by
                ``PgvectorConceptSearcher.search_terms(..., return_embeddings=True)``
                to avoid re-embedding the same term in Step 2.

        Returns:
            ReferenceSearchResult(examples, cost).
        """
        top_k = top_k if top_k is not None else self.hierarchy_settings.retrieval.num_reference_examples
        if embedding is not None:
            import numpy as np  # local import — only needed here
            vector = np.asarray(embedding, dtype=float)
            cost = 0.0
        else:
            result = get_embedding_vectors([text])
            vector = result["embeddings"][0]
            cost = result["usage"]["total_cost_usd"]
            self._cost += cost

        inner_limit = top_k * 10
        if self._exclude_concept_ids:
            query = sql.SQL("""
                WITH q AS (SELECT %s::vector AS vec),
                     ranked AS (
                    SELECT concept_id_1, concept_name_1,
                           1 - (embedding <=> q.vec) AS similarity,
                           ROW_NUMBER() OVER (
                               PARTITION BY concept_id_1
                               ORDER BY embedding <=> q.vec
                           ) AS rn
                    FROM {schema}.{table}, q
                    WHERE concept_id_1 != ALL(%s)
                    ORDER BY embedding <=> q.vec
                    LIMIT %s
                )
                SELECT concept_id_1, concept_name_1, similarity
                FROM ranked
                WHERE rn = 1
                ORDER BY similarity DESC
                LIMIT %s
            """).format(schema=sql.Identifier(self.schema), table=sql.Identifier("snomed_reference"))
            with self.connection.cursor() as cur:
                cur.execute(query, (vector.tolist(), self._exclude_concept_ids, inner_limit, top_k))
                top_terms = cur.fetchall()
        else:
            query = sql.SQL("""
                WITH q AS (SELECT %s::vector AS vec),
                     ranked AS (
                    SELECT concept_id_1, concept_name_1,
                           1 - (embedding <=> q.vec) AS similarity,
                           ROW_NUMBER() OVER (
                               PARTITION BY concept_id_1
                               ORDER BY embedding <=> q.vec
                           ) AS rn
                    FROM {schema}.{table}, q
                    ORDER BY embedding <=> q.vec
                    LIMIT %s
                )
                SELECT concept_id_1, concept_name_1, similarity
                FROM ranked
                WHERE rn = 1
                ORDER BY similarity DESC
                LIMIT %s
            """).format(schema=sql.Identifier(self.schema), table=sql.Identifier("snomed_reference"))
            with self.connection.cursor() as cur:
                cur.execute(query, (vector.tolist(), inner_limit, top_k))
                top_terms = cur.fetchall()

        if not top_terms:
            return ReferenceSearchResult([], cost)

        # Fetch all attribute rows for those concept_id_1 values
        ids = [r[0] for r in top_terms]
        attr_query = sql.SQL("""
            SELECT concept_id_1, concept_id_2, concept_code_2, concept_name_2, attribute_category
            FROM {schema}.{table}
            WHERE concept_id_1 = ANY(%s)
        """).format(schema=sql.Identifier(self.schema), table=sql.Identifier("snomed_reference"))
        with self.connection.cursor() as cur:
            cur.execute(attr_query, (ids,))
            attr_rows = cur.fetchall()

        # Group attributes by concept_id_1
        attrs_by_id: dict[int, list] = {}
        for concept_id_1, concept_id_2, concept_code_2, concept_name_2, attribute_category in attr_rows:
            attrs_by_id.setdefault(concept_id_1, []).append({
                "concept_id_2": concept_id_2,
                "concept_code_2": concept_code_2,
                "concept_name_2": concept_name_2,
                "attribute_category": attribute_category,
            })

        return ReferenceSearchResult([
            {
                "concept_id": concept_id_1,
                "concept_name": concept_name_1,
                "similarity": similarity,
                "attributes": attrs_by_id.get(concept_id_1, []),
            }
            for concept_id_1, concept_name_1, similarity in top_terms
        ], cost)


class SnomedReferenceConceptVectorSearcher(AbstractSnomedSearcher):
    """Reference searcher that queries live OMOP vocab tables plus concept vectors.

    This class mirrors :class:`SnomedReferenceSearcher` output semantics but
    does not depend on the precomputed ``snomed_reference`` table.
    """

    def __init__(
        self,
        hierarchy_settings: HierarchySettings | None = None,
        exclude_concept_ids: set[int] | None = None,
        include_synonyms: bool = False,
    ):
        super().__init__(hierarchy_settings)
        self._exclude_concept_ids: list[int] = sorted(exclude_concept_ids) if exclude_concept_ids else []
        self._relationship_ids: list[str] = list(self.hierarchy_settings.snomed_relationships)
        self.include_synonyms = include_synonyms
        self._concept_searcher = PgvectorConceptSearcher(
            for_evaluation=False,
            include_synonyms=include_synonyms,
            include_mapped_terms=True,
        )

    def _search_reference_terms(self, text: str, top_k: int) -> list[tuple[int, str, float]]:
        fetch_limit = top_k * 2
        result_df = self._concept_searcher.search_term(
            term=text,
            limit=fetch_limit,
            vocabulary_id="SNOMED",
        )
        if result_df is None or result_df.empty:
            return []

        if self._exclude_concept_ids:
            result_df = result_df[~result_df["concept_id"].isin(self._exclude_concept_ids)]
        if result_df.empty:
            return []

        top_df = result_df.head(top_k)
        return [
            (int(row.concept_id), str(row.concept_name), float(row.score))
            for row in top_df.itertuples(index=False)
        ]

    def _fetch_reference_attributes(self, source_concept_ids: list[int]) -> dict[int, list[dict]]:
        if not source_concept_ids:
            return {}

        query = sql.SQL("""
            SELECT cr.concept_id_1,
                   c2.concept_id,
                   c2.concept_code,
                   c2.concept_name,
                   cr.relationship_id
            FROM {schema}.concept_relationship cr
            INNER JOIN {schema}.concept c2 
                ON c2.concept_id = cr.concept_id_2
            WHERE cr.concept_id_1 = ANY(%s)
              AND cr.relationship_id = ANY(%s)
              AND cr.invalid_reason IS NULL
        """).format(schema=sql.Identifier(self.schema))

        with self.connection.cursor() as cur:
            cur.execute(query, (source_concept_ids, self._relationship_ids))
            rows = cur.fetchall()

        attrs_by_id: dict[int, list[dict]] = {}
        for concept_id_1, concept_id_2, concept_code_2, concept_name_2, attribute_category in rows:
            attrs_by_id.setdefault(concept_id_1, []).append(
                {
                    "concept_id_2": concept_id_2,
                    "concept_code_2": concept_code_2,
                    "concept_name_2": concept_name_2,
                    "attribute_category": attribute_category,
                }
            )
        return attrs_by_id

    def search(
        self,
        text: str,
        top_k: int | None = None,
    ) -> ReferenceSearchResult:
        top_terms = self._search_reference_terms(text, top_k=top_k)
        total_cost = self.get_total_cost()
        if not top_terms:
            return ReferenceSearchResult([], total_cost)

        attrs_by_id = self._fetch_reference_attributes([row[0] for row in top_terms])
        examples = [
            {
                "concept_id": concept_id_1,
                "concept_name": concept_name_1,
                "similarity": similarity,
                "attributes": attrs_by_id.get(concept_id_1, []),
            }
            for concept_id_1, concept_name_1, similarity in top_terms
        ]
        return ReferenceSearchResult(examples, total_cost)

    def get_total_cost(self) -> float:
        return self._cost + self._concept_searcher.get_total_cost()

    def close(self):
        self._concept_searcher.close()
        super().close()



