"""Four-step SNOMED CT attribute extraction pipeline.

Public API:
    find_attributes_two_stage(medical_term, attribute_searcher, ...) → dict

Helpers (prefixed with ``_``) handle individual steps:
    _retrieve_reference_examples  — Step 1
    extract_components            — Step 2
    _retrieve_candidates          — Step 3
    _build_selection_prompt       — Step 4a
"""

import json
import logging
from typing import cast

import pandas as pd

from ariadne.hierarchy.searchers import (
    ATTR_KEY_TO_SNOMED_CATEGORY,
    AbstractSnomedSearcher,
    SNOMED_CATEGORY_TO_ATTR_KEY,
)
from ariadne.hierarchy.types import (
    INTERPRETS_PAIRED_KEYS,
    ExtractionResult,
    LlmResult,
    ReferenceRetrievalResult,
    ReferenceSearchResult,
    SearchResult,
    merge_interprets_keys,
    split_interprets_pairs,
    validate_interprets_pairs,
)
from ariadne.utils.config import load_hierarchy_settings
from ariadne.utils.gen_ai_api import get_llm_response
from ariadne.utils.settings import HierarchySettings

logger = logging.getLogger(__name__)

# Type aliases
AttributeSearcher = AbstractSnomedSearcher
ReferenceSearcher = AbstractSnomedSearcher

# Normalize non-canonical keys the LLM may emit back to pipeline keys.
EXTRACTION_KEY_ALIASES: dict[str, str] = {
    "has_occurrence": "occurrence",
    "during": "occurrence",  # life-stage values (Congenital, Fetal period ...)
    "has_finding_context": "finding_context",
    "has_relat_context": "subject_relationship_context",
    "has_related_context": "subject_relationship_context",
    "has_related": "subject_relationship_context",
    "associated_with": "finding_asso_with",
    "finding_associated_with": "finding_asso_with",
}


class ContentFilterError(Exception):
    """Raised when the LLM content filter blocks a response."""


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def call_llm(system_prompt: str, user_prompt: str) -> LlmResult:
    """Call the LLM and return ``LlmResult(content, cost_usd)``.

    Args:
        system_prompt: System-level prompt text.
        user_prompt: User-level prompt text.

    Returns:
        LlmResult(content, cost).

    Raises:
        ContentFilterError: If the content filter blocks the response.
    """
    result = get_llm_response(user_prompt, system_prompt=system_prompt)
    if result["content"] is None:
        raise ContentFilterError(
            f"Content filter triggered for prompt: {user_prompt[:100]}..."
        )
    return LlmResult(result["content"], result["usage"]["total_cost_usd"])


def parse_json_response(response: str) -> dict:
    """Parse a JSON response from an LLM, stripping markdown fences if present.

    Args:
        response: Raw LLM response string.

    Returns:
        Parsed dict.

    Raises:
        ValueError: If the response cannot be parsed as JSON.
    """
    raw = response  # keep original for diagnostics
    response = response.strip()
    if response.startswith("```"):
        response = response.split("```")[1]
        if response.startswith("json"):
            response = response[4:]
    try:
        return json.loads(response)
    except json.JSONDecodeError as exc:
        logger.error(
            "Failed to parse LLM response as JSON. Raw response:\n%s", raw
        )
        raise ValueError(
            f"LLM returned malformed JSON: {exc}. "
            f"First 200 chars of response: {raw[:200]!r}"
        ) from exc


# ---------------------------------------------------------------------------
# Retrieval helpers (used by both pgvector and legacy paths)
# ---------------------------------------------------------------------------

def find_similar_reference_terms(
    query: str,
    reference_searcher: ReferenceSearcher,
    top_k: int,
    precomputed_embedding=None,
) -> ReferenceSearchResult:
    """Find similar reference terms for few-shot examples.

    Args:
        query: Medical term to search for.
        reference_searcher: Reference searcher (pgvector or legacy wrapper).
        top_k: Number of reference examples (from ``cfg.retrieval.num_reference_examples``).
        precomputed_embedding: Optional ``np.ndarray`` (shape ``[dim]``).  When
            supplied, passed straight through to
            ``SnomedReferenceSearcher.search`` so the embedding API call is
            skipped entirely.

    Returns:
        ReferenceSearchResult(examples, cost).
    """
    if precomputed_embedding is not None:
        return reference_searcher.search(query, top_k=top_k, embedding=precomputed_embedding)
    return reference_searcher.search(query, top_k=top_k)


def format_reference_examples(similar_terms: list[dict]) -> str:
    """Format reference examples into a human-readable block for the prompt.

    Args:
        similar_terms: List of reference dicts from ``find_similar_reference_terms``.

    Returns:
        Formatted string, or empty string if no terms.
    """
    if not similar_terms:
        return ""
    examples = []
    for term in similar_terms:
        attrs_text = [f"  - {a['attribute_category']}: {a['concept_name_2']} ({a['concept_id_2']})"
                      for a in term['attributes']]
        attrs_str = "\n".join(attrs_text) if attrs_text else "  (no attributes)"
        examples.append(f"Term: {term['concept_name']} ({term['concept_id']})\nAttributes:\n{attrs_str}")
    return "Similar SNOMED terms for reference:\n\n" + "\n\n".join(examples)


def _collect_reference_values(similar_terms: list[dict]) -> dict[str, list[dict]]:
    """Collect attribute values from reference examples, keyed by attr_key."""
    values_by_attr: dict[str, list[dict]] = {}
    seen_by_attr: dict[str, set[str]] = {}
    for term in similar_terms:
        for attr in term.get('attributes', []):
            attr_key = SNOMED_CATEGORY_TO_ATTR_KEY.get(attr['attribute_category'])
            if attr_key is None:
                continue
            concept_id_key = str(attr['concept_id_2'])
            seen_ids = seen_by_attr.setdefault(attr_key, set())
            if concept_id_key in seen_ids:
                continue
            seen_ids.add(concept_id_key)
            values_by_attr.setdefault(attr_key, []).append({
                'concept_id': attr['concept_id_2'],
                'concept_code': attr.get('concept_code_2'),
                'concept_name': attr['concept_name_2'],
            })
    return values_by_attr


# ---------------------------------------------------------------------------
# Step 1: Reference retrieval
# ---------------------------------------------------------------------------

def _retrieve_reference_examples(
    medical_term: str,
    reference_searcher: ReferenceSearcher | None,
    cfg: HierarchySettings,
    verbose: bool,
    precomputed_embedding=None,
) -> ReferenceRetrievalResult:
    """Step 1: Retrieve similar reference SNOMED terms for few-shot prompting.

    Args:
        medical_term: The medical term to find references for.
        reference_searcher: Reference searcher (or None to skip).
        cfg: Pipeline configuration.
        verbose: Whether to log progress.
        precomputed_embedding: Optional ``np.ndarray`` — when supplied the
            ``SnomedReferenceSearcher`` skips recomputing the embedding (saves
            one API call per term when the orchestrator passes the Step 1 vector
            through).

    Returns:
        ReferenceRetrievalResult(examples, prompt_text, cost).
    """
    if reference_searcher is None:
        return ReferenceRetrievalResult([], "", 0.0)

    if verbose:
        logger.debug("Step 1: Retrieving reference examples...")
    reference_examples, cost = find_similar_reference_terms(
        medical_term, reference_searcher, top_k=cfg.retrieval.num_reference_examples,
        precomputed_embedding=precomputed_embedding,
    )
    reference_text = format_reference_examples(reference_examples)
    if verbose:
        for term in reference_examples:
            logger.debug("  Reference: %s (similarity: %.3f)", term["concept_name"], term["similarity"])
    return ReferenceRetrievalResult(reference_examples, reference_text, cost)


# ---------------------------------------------------------------------------
# Step 2: Attribute extraction
# ---------------------------------------------------------------------------

def extract_components(
    medical_term: str,
    reference_text: str,
    cfg: HierarchySettings,
) -> ExtractionResult:
    """Step 2: Use the LLM to infer applicable SNOMED attributes.

    Args:
        medical_term: Term to decompose.
        reference_text: Formatted reference examples block.
        cfg: Pipeline configuration.

    Returns:
        ExtractionResult(components, cost).
    """
    if reference_text:
        reference_section = (
            "=== REFERENCE EXAMPLES ===\nStudy these carefully. "
            "They show how SNOMED assigns attributes to similar terms:\n\n" + reference_text
        )
    else:
        reference_section = ""
    system_prompt = cfg.prompts.extraction.format(reference_section=reference_section)
    user_prompt = f'Determine the attributes for: "{medical_term}"'
    response, cost = call_llm(system_prompt, user_prompt)
    return ExtractionResult(parse_json_response(response), cost)


# ---------------------------------------------------------------------------
# Step 3: Candidate retrieval
# ---------------------------------------------------------------------------

CANDIDATE_COLUMNS = [
    "concept_id",
    "concept_code",
    "concept_name",
    "attribute_category",
    "similarity",
    "extracted_mention",
    "attribute_key",
]


def _ensure_candidate_columns(candidates: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with all expected candidate columns present."""
    candidates = candidates.copy()
    for col in CANDIDATE_COLUMNS:
        if col not in candidates.columns:
            candidates[col] = None
    return candidates


def _normalize_and_dedupe_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Normalize candidate IDs and remove duplicates while preserving first rank."""
    if len(candidates) == 0:
        return candidates

    candidates = _ensure_candidate_columns(candidates)
    candidates["concept_id"] = candidates["concept_id"].astype(str)
    return candidates.drop_duplicates(subset=["concept_id"], keep="first")

def _unpack_mentions(components: dict[str, object]) -> list[tuple[str, str, str]]:
    """Unpack extraction output into ``(attr_key, mention, snomed_category)`` triples.

    Handles regular attributes, list-of-strings, and paired
    ``interprets_interpretation`` structures.

    Args:
        components: Parsed extraction dict ``{attr_key: value | list | None}``.

    Returns:
        List of (attr_key, mention_text, snomed_category) tuples.
    """
    mentions: list[tuple[str, str, str]] = []
    for raw_key, mention in components.items():
        if mention is None:
            continue

        attr_key = cast(str, EXTRACTION_KEY_ALIASES.get(raw_key, raw_key))

        # Handle paired interprets_interpretation tuples
        if attr_key == "interprets_interpretation":
            if isinstance(mention, list):
                for sub_key, val in split_interprets_pairs(mention):
                    sc = ATTR_KEY_TO_SNOMED_CATEGORY.get(sub_key)
                    if sc:
                        mentions.append((sub_key, str(val), sc))
            continue

        snomed_category = ATTR_KEY_TO_SNOMED_CATEGORY.get(attr_key)
        if snomed_category is None:
            continue
        # Support both single-string (legacy) and list-of-strings (new) extraction output
        if isinstance(mention, list):
            for item in mention:
                if item:
                    mentions.append((attr_key, str(item), snomed_category))
        else:
            mentions.append((attr_key, str(mention), snomed_category))
    return mentions


def _enrich_candidates(
    candidates: pd.DataFrame,
    attr_key: str,
    reference_values_by_attr: dict[str, list[dict]],
    attribute_searcher: AttributeSearcher,
    cfg: HierarchySettings,
    verbose: bool,
) -> pd.DataFrame:
    """Enrich candidates for a single attribute with reference values and hierarchy.

    Args:
        candidates: Initial candidates DataFrame for this attribute.
        attr_key: Attribute key (e.g. ``associated_morphology``).
        reference_values_by_attr: Reference values keyed by attr_key.
        attribute_searcher: Attribute searcher for hierarchy expansion.
        cfg: Pipeline configuration.
        verbose: Whether to log progress.

    Returns:
        Enriched candidates DataFrame.
    """
    candidates = _normalize_and_dedupe_candidates(candidates)
    snomed_category = ATTR_KEY_TO_SNOMED_CATEGORY.get(attr_key)

    # Enrich with values from reference examples not already in candidates
    if attr_key in reference_values_by_attr:
        existing_ids = set(candidates["concept_id"].tolist())
        mention = candidates["extracted_mention"].iloc[0]
        new_rows = [
            {"concept_id": str(rv["concept_id"]), "concept_code": rv.get("concept_code"),
             "concept_name": rv["concept_name"],
             "attribute_category": snomed_category, "similarity": cfg.scoring.reference_similarity,
             "extracted_mention": mention, "attribute_key": attr_key}
            for rv in reference_values_by_attr[attr_key]
            if str(rv["concept_id"]) not in existing_ids
        ]
        if new_rows:
            candidates = pd.concat([candidates, pd.DataFrame(new_rows)], ignore_index=True)
            if verbose:
                logger.debug("  %s: added %d values from reference examples", attr_key, len(new_rows))

    # Enrich with 1-hop hierarchy neighbors (parents + children)
    if snomed_category:
        existing_ids = set(candidates["concept_id"].tolist())
        hierarchy_df = attribute_searcher.expand_via_hierarchy(
            list(existing_ids), snomed_category
        )
        hierarchy_df = _ensure_candidate_columns(hierarchy_df)
        hierarchy_df["concept_id"] = hierarchy_df["concept_id"].astype(str)
        new_hier = hierarchy_df[~hierarchy_df["concept_id"].isin(existing_ids)]
        if len(new_hier) > 0:
            mention = candidates["extracted_mention"].iloc[0]
            new_hier = new_hier.copy()
            new_hier["extracted_mention"] = mention
            new_hier["attribute_key"] = attr_key
            candidates = pd.concat([candidates, new_hier], ignore_index=True)
            if verbose:
                logger.debug("  %s: added %d hierarchy neighbors", attr_key, len(new_hier))

    return _normalize_and_dedupe_candidates(candidates)


def _retrieve_candidates(
    extracted_components: dict,
    attribute_searcher: AttributeSearcher,
    reference_examples: list,
    verbose: bool,
    cfg: HierarchySettings,
) -> SearchResult:
    """Step 3: Embed each inferred attribute value and retrieve SNOMED candidates.

    Args:
        extracted_components: Parsed extraction output ``{attr_key: [free-text values]}``.
        attribute_searcher: Attribute searcher (pgvector or legacy dict).
        reference_examples: Reference examples (for enrichment).
        verbose: Whether to log progress.
        cfg: Pipeline configuration.

    Returns:
        SearchResult(candidates_df, total_embedding_cost).
    """
    reference_values_by_attr = _collect_reference_values(reference_examples)
    mentions = _unpack_mentions(extracted_components)

    if not mentions:
        return SearchResult(pd.DataFrame(columns=CANDIDATE_COLUMNS), 0.0)

    # Batch embed all mentions and search per-category
    indexed_mentions = [(f"{attr_key}_{i}", text, snomed_cat)
                        for i, (attr_key, text, snomed_cat) in enumerate(mentions)]
    results_by_idx, embedding_cost = attribute_searcher.search_batch(
        indexed_mentions, top_k=cfg.retrieval.top_k_per_category
    )

    # Group and deduplicate candidates per attr_key across multiple mentions
    candidates_by_attr: dict[str, pd.DataFrame] = {}
    for idx_key, (attr_key, mention, snomed_category) in zip(
        [m[0] for m in indexed_mentions], mentions
    ):
        candidates = results_by_idx.get(idx_key, pd.DataFrame(columns=CANDIDATE_COLUMNS))
        if len(candidates) == 0:
            continue

        candidates = _normalize_and_dedupe_candidates(candidates)
        candidates["extracted_mention"] = mention
        candidates["attribute_key"] = attr_key

        if attr_key in candidates_by_attr:
            candidates_by_attr[attr_key] = pd.concat(
                [candidates_by_attr[attr_key], candidates], ignore_index=True
            ).drop_duplicates(subset=["concept_id"], keep="first")
        else:
            candidates_by_attr[attr_key] = candidates

    all_candidates = []
    for attr_key, candidates in candidates_by_attr.items():
        candidates = _enrich_candidates(
            candidates, attr_key, reference_values_by_attr,
            attribute_searcher, cfg, verbose,
        )
        all_candidates.append(candidates)
        if verbose:
            top = candidates.iloc[0]
            logger.debug("  %s: top match = %s (score: %s)", attr_key, top["concept_name"],
                        top.get("similarity", "N/A"))

    candidates_df = (
        _ensure_candidate_columns(pd.concat(all_candidates, ignore_index=True))
        if all_candidates
        else pd.DataFrame(columns=CANDIDATE_COLUMNS)
    )
    if len(candidates_df) > 0:
        candidates_df["concept_id"] = candidates_df["concept_id"].astype(str)
        candidates_df = candidates_df.drop_duplicates(
            subset=["attribute_key", "concept_id"], keep="first"
        )
    return SearchResult(candidates_df, embedding_cost)


# ---------------------------------------------------------------------------
# Step 4: Selection prompt building
# ---------------------------------------------------------------------------

def _build_selection_prompt(candidates_df: pd.DataFrame) -> str:
    """Step 4a: Format the candidates DataFrame into the LLM selection prompt text.

    Args:
        candidates_df: DataFrame of candidates with columns
            ``[concept_id, concept_name, attribute_key, extracted_mention, similarity]``.

    Returns:
        Formatted prompt text listing candidates per attribute.
    """
    if candidates_df is None or len(candidates_df) == 0:
        return ""

    candidates_df = candidates_df.copy()
    if "attribute_key" not in candidates_df.columns:
        if "attribute_category" not in candidates_df.columns:
            return ""
        # Backward-compatible fallback for callers that only provide SNOMED categories.
        candidates_df["attribute_key"] = (
            candidates_df["attribute_category"]
            .map(SNOMED_CATEGORY_TO_ATTR_KEY)
            .fillna(candidates_df["attribute_category"])
        )

    if "concept_id" not in candidates_df.columns or "concept_name" not in candidates_df.columns:
        return ""

    if "extracted_mention" not in candidates_df.columns:
        candidates_df["extracted_mention"] = ""
    if "similarity" not in candidates_df.columns:
        candidates_df["similarity"] = None

    parts: list[str] = []
    paired_parts: dict[str, list[str]] = {}

    for group_key, group in candidates_df.groupby("attribute_key", sort=False):
        attr_key = cast(str, str(group_key))
        mentions = [str(m) for m in group["extracted_mention"].unique().tolist()]
        mention_str = "', '".join(mentions)
        lines = []
        for row in group.itertuples(index=False):
            similarity = getattr(row, "similarity", None)
            sim_str = f", similarity: {similarity:.3f}" if pd.notna(similarity) else ""
            lines.append(f"  - {row.concept_name} (concept_id: {row.concept_id}{sim_str})")

        if attr_key in INTERPRETS_PAIRED_KEYS:
            header = f"\n  {attr_key} candidates (inferred: '{mention_str}'):"
            paired_parts[attr_key] = [header] + lines
        else:
            header = f"\n{attr_key} (inferred: '{mention_str}'):"
            parts.append("\n".join([header] + lines))

    # Append grouped interprets_interpretation section
    if paired_parts:
        group_lines = ["\ninterprets_interpretation (PAIRED — select one interprets + one interpretation per role group):"]
        for key in ['interprets', 'interpretation']:
            if key in paired_parts:
                group_lines.extend(paired_parts[key])
        parts.append("\n".join(group_lines))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def _enforce_interprets_pairing(attrs: dict, *, verbose: bool = False) -> None:
    """Normalise interprets/interpretation keys into paired ``interprets_interpretation``.

    Operates **in-place** on *attrs*.  Handles:
    - Backward-compat merge of separate top-level keys.
    - Validation that each pair has both sides.
    - Cleanup of leftover separate keys.
    """
    if ("interprets" in attrs or "interpretation" in attrs) and "interprets_interpretation" not in attrs:
        merged = merge_interprets_keys(
            attrs.pop("interprets", None),
            attrs.pop("interpretation", None),
            verbose=verbose,
        )
        if merged:
            attrs["interprets_interpretation"] = merged

    if "interprets_interpretation" in attrs and attrs["interprets_interpretation"]:
        attrs["interprets_interpretation"] = validate_interprets_pairs(
            attrs["interprets_interpretation"], verbose=verbose,
        )

    # Clean up any leftover separate keys
    attrs.pop("interprets", None)
    attrs.pop("interpretation", None)


def _inject_selected_concept_codes(result: dict, candidates_df: pd.DataFrame) -> None:
    """Inject ``concept_code`` into selected attribute dicts in-place when missing."""
    if len(candidates_df) == 0 or "concept_code" not in candidates_df.columns:
        return

    code_lookup: dict[int, str] = (
        candidates_df.dropna(subset=["concept_code"])
        .drop_duplicates(subset=["concept_id"])
        .set_index("concept_id")["concept_code"]
        .to_dict()
    )

    def _inject_code(obj):
        if isinstance(obj, dict) and "concept_id" in obj:
            concept_id = obj.get("concept_id")
            if concept_id is not None and "concept_code" not in obj:
                obj["concept_code"] = code_lookup.get(int(concept_id))

    if "attributes" not in result or not isinstance(result["attributes"], dict):
        return

    for attr_key, attr_val in result["attributes"].items():
        if attr_val is None:
            continue
        if attr_key == "interprets_interpretation" and isinstance(attr_val, list):
            for pair in attr_val:
                if isinstance(pair, dict):
                    for value in pair.values():
                        _inject_code(value)
        elif isinstance(attr_val, list):
            for item in attr_val:
                _inject_code(item)
        else:
            _inject_code(attr_val)


def _attach_pipeline_metadata(
    result: dict,
    *,
    extracted_components: dict,
    candidates_df: pd.DataFrame,
    reference_examples: list[dict],
    extraction_cost: float,
    embedding_cost: float,
    selection_cost: float,
    total_cost: float,
) -> None:
    """Attach diagnostics and cost metadata to the pipeline result in-place."""
    result["extracted_components"] = extracted_components
    result["retrieved_candidates"] = (
        candidates_df.to_dict("records") if len(candidates_df) > 0 else []
    )
    if reference_examples:
        result["reference_examples"] = reference_examples
    result["cost"] = {
        "extraction_cost": extraction_cost,
        "embedding_cost": embedding_cost,
        "selection_cost": selection_cost,
        "total_cost": total_cost,
    }


def find_attributes_two_stage(
    medical_term: str,
    attribute_searcher: AttributeSearcher,
    reference_searcher: ReferenceSearcher | None = None,
    cfg: HierarchySettings | None = None,
    verbose: bool = True,
    precomputed_embedding=None,
) -> dict:
    """Run the 4-step SNOMED CT attribute extraction pipeline.

    Steps:
        1. Retrieve reference examples (pgvector or in-memory).
        2. LLM infers applicable attributes.
        3. Retrieve SNOMED candidate values per attribute.
        4. LLM selects exact SNOMED concepts from candidates.

    Args:
        medical_term: The clinical term to decompose.
        attribute_searcher: Attribute searcher (pgvector or legacy dict).
        reference_searcher: Reference searcher (pgvector, legacy dict, or None).
        cfg: Pipeline configuration.
        verbose: Whether to log progress.
        precomputed_embedding: Optional ``np.ndarray`` (shape ``[dim]``) — the
            embedding of *medical_term* computed upstream (e.g. by
            ``PgvectorConceptSearcher.search_terms``).  When supplied, Step 1's
            reference-retrieval embedding API call is skipped, saving cost.

    Returns:
        Dict with keys ``attributes``, ``extracted_components``,
        ``retrieved_candidates``, ``reference_examples``, ``cost``.
    """
    cfg_local: HierarchySettings = cfg if cfg is not None else load_hierarchy_settings()

    # Step 1
    reference_examples, reference_text, ref_cost = _retrieve_reference_examples(
        medical_term, reference_searcher, cfg_local, verbose,
        precomputed_embedding=precomputed_embedding,
    )

    # Step 2
    if verbose:
        logger.debug("Step 2: Inferring attributes...")
    components, extraction_cost = extract_components(medical_term, reference_text=reference_text,
                                                     cfg=cfg_local)
    if verbose:
        logger.debug("  Inferred: %s", json.dumps({k: v for k, v in components.items() if v}, indent=2))

    # Step 3
    if verbose:
        logger.debug("Step 3: Retrieving candidates...")
    candidates_df, embedding_cost = _retrieve_candidates(
        components, attribute_searcher, reference_examples,
        verbose, cfg=cfg_local
    )

    # Step 4
    if verbose:
        logger.debug("Step 4: Selecting best matches...")
    candidates_text = _build_selection_prompt(candidates_df)
    selection_cost = 0.0
    if candidates_text.strip():
        user_prompt = f"Medical term: {medical_term}\n\n{reference_text}\n\nCandidates:\n{candidates_text}"
        response, selection_cost = call_llm(cfg_local.prompts.selection, user_prompt)
        result = parse_json_response(response)
    else:
        if verbose:
            logger.debug("Step 4 skipped: no candidates available for selection.")
        result = {"attributes": {}}

    total_cost = ref_cost + extraction_cost + embedding_cost + selection_cost


    # --- Enforce interprets ↔ interpretation pairing ---
    if "attributes" in result:
        _enforce_interprets_pairing(result["attributes"], verbose=verbose)

    _inject_selected_concept_codes(result, candidates_df)
    _attach_pipeline_metadata(
        result,
        extracted_components=components,
        candidates_df=candidates_df,
        reference_examples=reference_examples,
        extraction_cost=extraction_cost,
        embedding_cost=embedding_cost,
        selection_cost=selection_cost,
        total_cost=total_cost,
    )
    if verbose:
        logger.debug("Total cost: $%.4f", total_cost)
    return result
