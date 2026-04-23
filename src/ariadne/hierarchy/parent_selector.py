"""Stated parent selection for SNOMED CT new concept classification.

Given the raw pipeline results (from ``hierarchy_results_raw.json``), this
module derives a per-concept stated "Is a" parent list by voting across the
SNOMED parents of the retrieved reference terms.

**No embeddings or LLM calls** — operates purely on OMOP ``concept_id``s
already stored in the cached JSON, plus SQL against ``concept_relationship``.

Algorithm
---------
For each source concept:

1. Collect the reference examples stored in ``raw_results[i]["reference_examples"]``.
2. Batch-query ``concept_relationship`` for the "Is a" parents of all unique
   reference term ``concept_id``s in a single SQL call.
3. For each reference term with ``similarity >= min_similarity``, add its
   parent SCTIDs to a weighted vote (weight = similarity score).
4. Filter out root/overly-generic parents (see ``_GENERIC_CONCEPT_IDS``).
5. Optionally apply an attribute subsumption filter: skip a reference term
   whose attribute ``concept_id_2`` set is a *strict subset* of the source's
   predicted attribute set (that reference term is less specific → its parents
   would be too deep in the hierarchy).
6. Return the top-k SCTIDs by weighted vote.  Fall back to ``Clinical finding``
   (404684003) if no candidates pass all filters.

Usage::

    import json, psycopg
    from ariadne.hierarchy.parent_selector import build_stated_parents_map

    with open("data/notebook_results/hierarchy_results_raw.json") as f:
        results = json.load(f)

    with psycopg.connect(conn_str) as conn:
        stated_parents = build_stated_parents_map(results, conn, schema)

    # {omop_concept_id: ["sctid1", "sctid2", ...]}
"""
from __future__ import annotations

import logging
from collections import defaultdict

import psycopg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Generic / root parents to exclude from candidates
# These are valid SNOMED parents but too broad to be useful as stated parents
# for a new clinical concept.
# ---------------------------------------------------------------------------
_GENERIC_CONCEPT_IDS: set[int] = {
    138875005,   # SNOMED CT Concept (root)
    404684003,   # Clinical finding
    64572001,    # Disease
    118234003,   # Finding
    49755003,    # Morphologic abnormality
    71388002,    # Procedure
    123037004,   # Body structure
    272379006,   # Event
    900000000000441003,  # SNOMED CT Model Component
}

_CLINICAL_FINDING_SCTID = "404684003"


# ---------------------------------------------------------------------------
# Step 1 – batch query: reference concept_id → their "Is a" parent SCTIDs
# ---------------------------------------------------------------------------

def get_reference_parents(
    ref_concept_ids: list[int],
    conn: psycopg.Connection,
    schema: str,
) -> dict[int, list[tuple[str, str]]]:
    """Return "Is a" parents for a batch of reference SNOMED concept IDs.

    Args:
        ref_concept_ids: OMOP ``concept_id`` values of the reference terms.
        conn: Open psycopg connection to the vocabulary database.
        schema: OMOP vocabulary schema name.

    Returns:
        ``{ref_concept_id: [(parent_concept_code, parent_concept_name), ...]}``.
        Generic parents (see ``_GENERIC_CONCEPT_IDS``) are excluded.
    """
    if not ref_concept_ids:
        return {}

    result: dict[int, list[tuple[str, str]]] = defaultdict(list)

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT cr.concept_id_1,
                   c2.concept_id    AS parent_concept_id,
                   c2.concept_code  AS parent_sctid,
                   c2.concept_name  AS parent_name
            FROM {schema}.concept_relationship cr
            JOIN {schema}.concept c2
              ON c2.concept_id = cr.concept_id_2
            WHERE cr.concept_id_1 = ANY(%s)
              AND cr.relationship_id = 'Is a'
              AND cr.invalid_reason IS NULL
              AND c2.vocabulary_id  = 'SNOMED'
            """,
            (ref_concept_ids,),
        )
        for ref_id, parent_cid, parent_code, parent_name in cur.fetchall():
            if parent_cid not in _GENERIC_CONCEPT_IDS:
                result[ref_id].append((parent_code, parent_name))

    return dict(result)


# ---------------------------------------------------------------------------
# Step 2 – vote: score parent candidates for one source concept
# ---------------------------------------------------------------------------

def score_parent_candidates(
    reference_examples: list[dict],
    reference_parents: dict[int, list[tuple[str, str]]],
    source_attr_concept_ids: set[int],
    *,
    min_similarity: float = 0.7,
    use_attr_filter: bool = True,
    top_k: int = 2,
) -> list[tuple[str, str, float]]:
    """Vote across reference term parents to find the best candidates.

    Args:
        reference_examples: The ``reference_examples`` list from one raw
            result entry.  Each item has ``concept_id``, ``similarity``,
            and ``attributes`` (list of ``{concept_id_2, ...}``).
        reference_parents: Output of ``get_reference_parents`` — maps
            reference concept_id to its "Is a" parent SCTIDs.
        source_attr_concept_ids: Set of ``concept_id_2`` values predicted for
            the source concept (used for the subsumption filter).
        min_similarity: Skip reference terms with similarity below this threshold.
        use_attr_filter: When True, skip reference terms whose attribute set is
            a *strict subset* of the source's predicted attributes — those
            terms are less specific and their parents would be too deep.
        top_k: Maximum number of parent candidates to return.

    Returns:
        List of ``(parent_sctid, parent_name, weighted_vote_score)`` tuples,
        sorted descending by score, capped at ``top_k``.
    """
    vote: dict[str, float] = defaultdict(float)
    vote_names: dict[str, str] = {}

    for ref in reference_examples:
        sim = ref.get("similarity", 0.0)
        if sim < min_similarity:
            continue

        ref_cid = ref.get("concept_id")
        if ref_cid is None:
            continue

        # Attribute subsumption filter: skip references that are less specific
        # than our source concept (their attribute set ⊊ source attribute set).
        if use_attr_filter and source_attr_concept_ids:
            ref_attr_ids = {
                a["concept_id_2"]
                for a in ref.get("attributes", [])
                if "concept_id_2" in a
            }
            if ref_attr_ids and ref_attr_ids < source_attr_concept_ids:
                # Reference is strictly less specific — skip
                continue

        for sctid, name in reference_parents.get(ref_cid, []):
            vote[sctid] += sim
            vote_names[sctid] = name

    if not vote:
        return []

    ranked = sorted(vote.items(), key=lambda x: x[1], reverse=True)
    return [(sctid, vote_names[sctid], score) for sctid, score in ranked[:top_k]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_stated_parents_map(
    raw_results: list[dict],
    conn: psycopg.Connection,
    schema: str,
    *,
    top_k: int = 2,
    min_similarity: float = 0.7,
    use_attr_filter: bool = True,
) -> dict[int, list[str]]:
    """Build a per-concept stated parent map from cached pipeline results.

    Reads ``source_concept_id`` and ``reference_examples`` from each entry in
    *raw_results*, queries ``concept_relationship`` for reference parents in a
    single batch call, then votes to select up to *top_k* parent SCTIDs per
    source concept.

    Args:
        raw_results: List of result dicts from ``hierarchy_results_raw.json``.
        conn: Open psycopg connection to the vocabulary database.
        schema: OMOP vocabulary schema name.
        top_k: Maximum stated parents per concept (default 2).
        min_similarity: Minimum reference similarity to be counted (default 0.7).
        use_attr_filter: Apply attribute-subsumption filter (default True).

    Returns:
        ``{omop_concept_id: [parent_sctid, ...]}``.  Falls back to
        ``["404684003"]`` (Clinical finding) when no candidates are found.
    """
    # Collect all unique reference concept_ids in one pass
    all_ref_ids: list[int] = []
    for entry in raw_results:
        for ref in entry.get("reference_examples", []):
            cid = ref.get("concept_id")
            if cid is not None:
                all_ref_ids.append(int(cid))

    unique_ref_ids = list(set(all_ref_ids))
    logger.info(
        "Querying parents for %d unique reference concepts…", len(unique_ref_ids)
    )

    reference_parents = get_reference_parents(unique_ref_ids, conn, schema)
    logger.info(
        "Found parents for %d / %d reference concepts.",
        len(reference_parents), len(unique_ref_ids),
    )

    stated_parents: dict[int, list[str]] = {}
    fallback_count = 0
    specific_count = 0

    for entry in raw_results:
        src_id = entry.get("source_concept_id")
        if src_id is None:
            continue
        src_id = int(src_id)

        # Build source attribute id set for subsumption filter.
        # attribute values may be a list[dict] OR a bare dict (single item not wrapped).
        source_attr_ids: set[int] = set()
        for attr_val in entry.get("attributes", {}).values():
            if not attr_val:
                continue
            items: list = attr_val if isinstance(attr_val, list) else [attr_val]
            for a in items:
                if isinstance(a, dict) and "concept_id" in a:
                    source_attr_ids.add(int(a["concept_id"]))

        ref_examples = entry.get("reference_examples", [])
        candidates = score_parent_candidates(
            ref_examples,
            reference_parents,
            source_attr_ids,
            min_similarity=min_similarity,
            use_attr_filter=use_attr_filter,
            top_k=top_k,
        )

        if candidates:
            stated_parents[src_id] = [c[0] for c in candidates]
            specific_count += 1
        else:
            stated_parents[src_id] = [_CLINICAL_FINDING_SCTID]
            fallback_count += 1

    logger.info(
        "Stated parents: %d concepts with specific parents, %d fell back to Clinical finding.",
        specific_count, fallback_count,
    )
    return stated_parents
