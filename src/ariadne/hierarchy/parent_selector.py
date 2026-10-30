"""Stated parent selection for new SNOMED CT clinical concepts.

The RF2 exporter writes new concepts as sufficiently defined, so the classifier can infer fully
defined supertypes from the attributes but never primitive ones. The stated parent therefore has to
be a primitive concept, ideally the proximal primitive parent.

Allowed parents are the primitive SNOMED descendants of ``root_concept_id`` (Clinical finding); the
root itself is the fallback.

Algorithm, per source term
--------------------------
1. Neighbours: the ``num_neighbours`` most similar standard SNOMED concepts (embedding search) with
   ``similarity >= min_similarity``. With ``exclude_source_descendants`` the source concept and its
   descendants are dropped first, because they would not exist for a genuinely new concept.
2. Candidates: the primitive ancestors of each neighbour (the neighbour itself excluded).
3. Support: the similarity-weighted share of neighbours below each candidate.
4. Selection: candidates with ``support >= min_support``, minus any candidate that is an ancestor of
   another qualifying candidate, deepest first (``min_levels_of_separation`` from the root), capped
   at ``max_parents``. When neighbours agree this is their proximal primitive parent; when they
   disagree it is an intermediate primitive.
5. Fallback: the root when no candidate qualifies.

Usage::

    with psycopg.connect(conn_str) as conn:
        stated_parents, details = build_stated_parents_map(
            raw_results, conn, schema, config.hierarchy.parent_selection
        )
    # stated_parents: {source_concept_id (or source_key): [parent_sctid, ...]}
"""
from __future__ import annotations

import json
import logging
import os
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import pandas as pd
import psycopg
from psycopg import sql

from ariadne.utils.concurrency import run_concurrently
from ariadne.utils.settings import ConceptFilterSettings, ParentSelectionConfig, PgvectorSearchSettings
from ariadne.vector_search.pgvector_concept_searcher import PgvectorConceptSearcher

logger = logging.getLogger(__name__)

_SUPPORT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Neighbour:
    concept_id: int
    concept_name: str
    similarity: float


@dataclass(frozen=True)
class PrimitiveConcept:
    concept_id: int
    concept_code: str
    concept_name: str
    # min_levels_of_separation from the root concept.
    depth: int


@dataclass(frozen=True)
class ParentCandidate:
    concept: PrimitiveConcept
    support: float


@dataclass
class PrimitiveAncestry:
    """Strict primitive ancestors per concept, restricted to the allowed parent set."""

    concepts: dict[int, PrimitiveConcept] = field(default_factory=dict)
    ancestors: dict[int, set[int]] = field(default_factory=dict)

    def is_ancestor(self, ancestor_id: int, concept_id: int) -> bool:
        return ancestor_id in self.ancestors.get(concept_id, ())


# ---------------------------------------------------------------------------
# Selection (pure)
# ---------------------------------------------------------------------------


def most_specific(concept_ids: Iterable[int], ancestry: PrimitiveAncestry) -> set[int]:
    """Drop every concept that is an ancestor of another concept in the set."""
    ids = set(concept_ids)
    return {a for a in ids if not any(ancestry.is_ancestor(a, b) for b in ids if b != a)}


def select_primitive_parents(
    neighbours: Sequence[Neighbour],
    ancestry: PrimitiveAncestry,
    *,
    min_support: float,
    max_parents: int,
) -> list[ParentCandidate]:
    """Pick the deepest primitive ancestors shared by enough of the neighbours.

    Args:
        neighbours: Usable neighbours (already filtered for exclusions and similarity).
        ancestry: Primitive ancestors of the neighbours and of every candidate parent.
        min_support: Minimum similarity-weighted share of neighbours below a candidate.
        max_parents: Maximum number of parents returned.

    Returns:
        Parents ordered deepest first, then by support. Empty when nothing qualifies.
    """
    total = sum(n.similarity for n in neighbours)
    if total <= 0:
        return []

    weight: dict[int, float] = defaultdict(float)
    for neighbour in neighbours:
        for parent_id in ancestry.ancestors.get(neighbour.concept_id, ()):
            weight[parent_id] += neighbour.similarity

    qualifying = {pid for pid, w in weight.items() if w / total >= min_support - _SUPPORT_TOLERANCE}
    ranked = sorted(
        most_specific(qualifying, ancestry),
        key=lambda pid: (-ancestry.concepts[pid].depth, -weight[pid], pid),
    )
    return [ParentCandidate(ancestry.concepts[pid], weight[pid] / total) for pid in ranked[:max_parents]]


def proximal_primitive_parents(concept_id: int, ancestry: PrimitiveAncestry) -> list[PrimitiveConcept]:
    """Proximal primitive parents of an existing concept, deepest first."""
    parent_ids = most_specific(ancestry.ancestors.get(concept_id, ()), ancestry)
    return sorted((ancestry.concepts[pid] for pid in parent_ids), key=lambda c: (-c.depth, c.concept_id))


# ---------------------------------------------------------------------------
# Vocabulary queries
# ---------------------------------------------------------------------------


def fetch_primitive_ancestry(
    concept_ids: Iterable[int],
    conn: psycopg.Connection,
    schema: str,
    settings: ParentSelectionConfig,
) -> PrimitiveAncestry:
    """Fetch the strict primitive ancestors of *concept_ids* and, recursively, of those ancestors.

    The ancestors of the ancestors are needed to tell which candidate parents subsume each other.
    """
    query = sql.SQL(
        """
        WITH primitive_concept AS (
            SELECT c.concept_id,
                   c.concept_code,
                   c.concept_name,
                   root.min_levels_of_separation AS depth
            FROM {schema}.concept c
            JOIN {schema}.concept_ancestor root
              ON root.descendant_concept_id = c.concept_id
             AND root.ancestor_concept_id = %s
            JOIN {schema}.concept_relationship cr
              ON cr.concept_id_1 = c.concept_id
             AND cr.concept_id_2 = %s
             AND cr.relationship_id = 'Has status'
             AND cr.invalid_reason IS NULL
            WHERE c.vocabulary_id = 'SNOMED'
        )
        SELECT ca.descendant_concept_id,
               p.concept_id,
               p.concept_code,
               p.concept_name,
               p.depth
        FROM {schema}.concept_ancestor ca
        JOIN primitive_concept p
          ON p.concept_id = ca.ancestor_concept_id
        WHERE ca.descendant_concept_id = ANY(%s)
          AND ca.min_levels_of_separation >= 1
        """
    ).format(schema=sql.Identifier(schema))

    ancestry = PrimitiveAncestry()
    pending = {int(cid) for cid in concept_ids}
    with conn.cursor() as cur:
        while pending:
            cur.execute(
                query,
                (settings.root_concept_id, settings.primitive_status_concept_id, sorted(pending)),
            )
            for cid in pending:
                ancestry.ancestors.setdefault(cid, set())
            for descendant_id, parent_id, code, name, depth in cur.fetchall():
                ancestry.ancestors[descendant_id].add(parent_id)
                ancestry.concepts[parent_id] = PrimitiveConcept(parent_id, str(code), name, int(depth))
            pending = set(ancestry.concepts) - set(ancestry.ancestors)
    return ancestry


def fetch_descendants(
    ancestor_ids: Iterable[int],
    candidate_ids: Iterable[int],
    conn: psycopg.Connection,
    schema: str,
) -> dict[int, set[int]]:
    """Return ``{ancestor_id: {candidate_id, ...}}`` for candidates that descend from (or equal) each ancestor."""
    ancestor_ids = sorted({int(a) for a in ancestor_ids})
    candidate_ids = sorted({int(c) for c in candidate_ids})
    result: dict[int, set[int]] = defaultdict(set)
    if not ancestor_ids or not candidate_ids:
        return result
    query = sql.SQL(
        """
        SELECT ancestor_concept_id, descendant_concept_id
        FROM {schema}.concept_ancestor
        WHERE ancestor_concept_id = ANY(%s)
          AND descendant_concept_id = ANY(%s)
        """
    ).format(schema=sql.Identifier(schema))
    with conn.cursor() as cur:
        cur.execute(query, (ancestor_ids, candidate_ids))
        for ancestor_id, descendant_id in cur.fetchall():
            result[ancestor_id].add(descendant_id)
    return result


def fetch_root_concept(conn: psycopg.Connection, schema: str, settings: ParentSelectionConfig) -> PrimitiveConcept:
    query = sql.SQL("SELECT concept_code, concept_name FROM {schema}.concept WHERE concept_id = %s").format(
        schema=sql.Identifier(schema)
    )
    with conn.cursor() as cur:
        cur.execute(query, (settings.root_concept_id,))
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"Root concept {settings.root_concept_id} not found in {schema}.concept.")
    return PrimitiveConcept(settings.root_concept_id, str(row[0]), row[1], 0)


# ---------------------------------------------------------------------------
# Neighbour retrieval
# ---------------------------------------------------------------------------


def _load_neighbour_cache(settings: ParentSelectionConfig) -> dict[str, list[Neighbour]]:
    path = settings.neighbour_cache_file
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    # A smaller cached pool could be exhausted by exclusions, so it is not reused.
    if payload.get("candidate_pool_size") != settings.candidate_pool_size:
        return {}
    return {
        term: [Neighbour(int(cid), name, float(sim)) for cid, name, sim in hits]
        for term, hits in payload.get("neighbours", {}).items()
    }


def _save_neighbour_cache(cache: dict[str, list[Neighbour]], settings: ParentSelectionConfig) -> None:
    path = settings.neighbour_cache_file
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "candidate_pool_size": settings.candidate_pool_size,
        "neighbours": {
            term: [[n.concept_id, n.concept_name, n.similarity] for n in hits] for term, hits in cache.items()
        },
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def retrieve_neighbours(terms: Sequence[str], settings: ParentSelectionConfig) -> dict[str, list[Neighbour]]:
    """Return the ``candidate_pool_size`` most similar standard SNOMED concepts per term.

    Results are cached in ``settings.neighbour_cache_file``; only uncached terms are searched.
    """
    cache = _load_neighbour_cache(settings)
    missing = [term for term in dict.fromkeys(terms) if term not in cache]
    if not missing:
        return {term: cache[term] for term in terms}

    logger.info("Searching neighbours for %d terms (%d cached)…", len(missing), len(cache))
    search_settings = PgvectorSearchSettings(
        max_candidates=settings.candidate_pool_size,
        filter=ConceptFilterSettings(standard_concept=["S"], vocabulary_ids=["SNOMED"]),
        include_synonyms=False,
        include_mapped_terms=True,
    )
    local = threading.local()
    searchers: list[PgvectorConceptSearcher] = []
    searchers_lock = threading.Lock()

    def search(term: str) -> list[Neighbour]:
        if not hasattr(local, "searcher"):
            local.searcher = PgvectorConceptSearcher(settings=search_settings)
            with searchers_lock:
                searchers.append(local.searcher)
        result_df = local.searcher.search_term(term)
        if result_df is None:
            return []
        # PgvectorConceptSearcher returns cosine distance; convert to similarity.
        return [
            Neighbour(int(row.concept_id), str(row.concept_name), 1.0 - float(row.score))
            for row in result_df.itertuples(index=False)
        ]

    try:
        for done, (idx, hits) in enumerate(run_concurrently(missing, search, settings.max_workers), start=1):
            cache[missing[idx]] = hits
            if done % 50 == 0:
                logger.info("  %d / %d terms searched", done, len(missing))
    finally:
        cost = sum(s.get_total_cost() for s in searchers)
        for searcher in searchers:
            searcher.close()
        _save_neighbour_cache(cache, settings)
        logger.info("Neighbour search embedding cost: $%.6f", cost)

    return {term: cache[term] for term in terms}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _source_term(entry: dict) -> str:
    return str(entry.get("source_term") or "")


def _source_key(entry: dict) -> int | str | None:
    sid = entry.get("source_concept_id")
    if sid is not None and not (isinstance(sid, float) and pd.isna(sid)):
        return int(sid)
    return entry.get("source_key")


def build_stated_parents_map(
    raw_results: list[dict],
    conn: psycopg.Connection,
    schema: str,
    settings: ParentSelectionConfig,
    *,
    neighbours: dict[str, list[Neighbour]] | None = None,
) -> tuple[dict[int | str, list[str]], pd.DataFrame]:
    """Select stated parents for every entry of the hierarchy pipeline results.

    Args:
        raw_results: Result dicts from ``process_hierarchy`` (``hierarchy_results_raw.json``). Uses
            ``source_concept_id`` (or ``source_key``) and ``source_term``.
        conn: Open psycopg connection to the vocabulary database.
        schema: OHDSI Vocabularies schema name.
        settings: ``config.hierarchy.parent_selection``.
        neighbours: Pre-computed ``{source_term: [Neighbour, ...]}``; searched (and cached) when omitted.

    Returns:
        ``(stated_parents, details)``. *stated_parents* maps each source key to parent SCTIDs, as
        expected by ``export_to_rf2(stated_parent=...)``. *details* has one row per selected parent
        with its support, depth and whether it is the fallback.
    """
    entries = [(key, _source_term(e)) for e in raw_results if (key := _source_key(e)) is not None]
    if neighbours is None:
        neighbours = retrieve_neighbours([term for _, term in entries], settings)

    excluded: dict[int, set[int]] = defaultdict(set)
    if settings.exclude_source_descendants:
        source_ids = [key for key, _ in entries if isinstance(key, int)]
        pool_ids = {n.concept_id for hits in neighbours.values() for n in hits}
        excluded = fetch_descendants(source_ids, pool_ids, conn, schema)
        for sid in source_ids:
            excluded.setdefault(sid, set()).add(sid)

    usable: dict[int | str, list[Neighbour]] = {}
    for key, term in entries:
        skip = excluded.get(key, set()) if isinstance(key, int) else set()
        usable[key] = [
            n for n in neighbours.get(term, []) if n.concept_id not in skip and n.similarity >= settings.min_similarity
        ][: settings.num_neighbours]

    ancestry = fetch_primitive_ancestry(
        {n.concept_id for hits in usable.values() for n in hits}, conn, schema, settings
    )
    root = fetch_root_concept(conn, schema, settings)

    stated_parents: dict[int | str, list[str]] = {}
    rows = []
    for key, term in entries:
        parents = select_primitive_parents(
            usable[key], ancestry, min_support=settings.min_support, max_parents=settings.max_parents
        )
        if not parents:
            parents = [ParentCandidate(root, 0.0)]
        is_fallback = all(p.concept.concept_id == root.concept_id for p in parents)
        stated_parents[key] = [p.concept.concept_code for p in parents]
        for rank, parent in enumerate(parents, start=1):
            rows.append(
                {
                    "source_concept_id": key,
                    "source_term": term,
                    "parent_rank": rank,
                    "parent_concept_id": parent.concept.concept_id,
                    "parent_concept_code": parent.concept.concept_code,
                    "parent_concept_name": parent.concept.concept_name,
                    "parent_depth": parent.concept.depth,
                    "support": round(parent.support, 4),
                    "num_neighbours": len(usable[key]),
                    "is_fallback": is_fallback,
                }
            )

    details = pd.DataFrame(rows)
    fallback_count = int(details.drop_duplicates("source_concept_id")["is_fallback"].sum()) if rows else 0
    logger.info(
        "Stated parents: %d terms with a primitive parent below the root, %d fell back to %s.",
        len(stated_parents) - fallback_count,
        fallback_count,
        root.concept_name,
    )
    return stated_parents, details
