"""Four-step SNOMED CT attribute extraction pipeline.

Public API:
    find_attributes_two_stage(medical_term, attribute_index, ...) → dict

Helpers (prefixed with ``_``) handle individual steps:
    _retrieve_reference_examples  — Step 1
    extract_components            — Step 2
    _retrieve_candidates          — Step 3
    _build_selection_prompt       — Step 4a
"""

import json
import logging

import pandas as pd

from ariadne.hierarchy.config import HierarchyConfig
from ariadne.hierarchy.searchers import (
    ATTR_KEY_TO_SNOMED_CATEGORY,
    AbstractSnomedSearcher,
    SNOMED_CATEGORY_TO_ATTR_KEY,
    SnomedAttributeSearcher,
    SnomedReferenceSearcher,
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
from ariadne.utils.gen_ai_api import get_llm_response

logger = logging.getLogger(__name__)

# Type aliases
AttributeIndex = AbstractSnomedSearcher
ReferenceIndex = AbstractSnomedSearcher


class ContentFilterError(Exception):
    """Raised when the LLM content filter blocks a response."""


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def call_llm(system_prompt: str, user_prompt: str, model: str) -> LlmResult:
    """Call the LLM and return ``LlmResult(content, cost_usd)``.

    Args:
        system_prompt: System-level prompt text.
        user_prompt: User-level prompt text.
        model: Model identifier (from ``cfg.models``).

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
    reference_index: ReferenceIndex,
    top_k: int,
    precomputed_embedding=None,
) -> ReferenceSearchResult:
    """Find similar reference terms for few-shot examples.

    Args:
        query: Medical term to search for.
        reference_index: Reference searcher (pgvector or legacy wrapper).
        top_k: Number of reference examples (from ``cfg.retrieval.num_reference_examples``).
        precomputed_embedding: Optional ``np.ndarray`` (shape ``[dim]``).  When
            supplied, passed straight through to
            ``SnomedReferenceSearcher.search`` so the embedding API call is
            skipped entirely.

    Returns:
        ReferenceSearchResult(examples, cost).
    """
    if precomputed_embedding is not None:
        return reference_index.search(query, top_k=top_k, embedding=precomputed_embedding)
    return reference_index.search(query, top_k=top_k)


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
    for term in similar_terms:
        for attr in term.get('attributes', []):
            attr_key = SNOMED_CATEGORY_TO_ATTR_KEY.get(attr['attribute_category'])
            if attr_key is None:
                continue
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
    reference_index: ReferenceIndex | None,
    cfg: HierarchyConfig,
    verbose: bool,
    precomputed_embedding=None,
) -> ReferenceRetrievalResult:
    """Step 1: Retrieve similar reference SNOMED terms for few-shot prompting.

    Args:
        medical_term: The medical term to find references for.
        reference_index: Reference searcher (or None to skip).
        cfg: Pipeline configuration.
        verbose: Whether to log progress.
        precomputed_embedding: Optional ``np.ndarray`` — when supplied the
            ``SnomedReferenceSearcher`` skips recomputing the embedding (saves
            one API call per term when the orchestrator passes the Step 1 vector
            through).

    Returns:
        ReferenceRetrievalResult(examples, prompt_text, cost).
    """
    if reference_index is None:
        return ReferenceRetrievalResult([], "", 0.0)

    if verbose:
        logger.info("Step 1: Retrieving reference examples...")
    similar_terms, cost = find_similar_reference_terms(
        medical_term, reference_index, top_k=cfg.retrieval.num_reference_examples,
        precomputed_embedding=precomputed_embedding,
    )
    reference_text = format_reference_examples(similar_terms)
    if verbose:
        for t in similar_terms:
            logger.info("  Reference: %s (similarity: %.3f)", t['concept_name'], t['similarity'])
    return ReferenceRetrievalResult(similar_terms, reference_text, cost)


# ---------------------------------------------------------------------------
# Step 2: Attribute extraction
# ---------------------------------------------------------------------------

def extract_components(
    medical_term: str,
    reference_text: str,
    cfg: HierarchyConfig,
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
    response, cost = call_llm(system_prompt, user_prompt, model=cfg.models.extraction)
    return ExtractionResult(parse_json_response(response), cost)


# ---------------------------------------------------------------------------
# Step 3: Candidate retrieval
# ---------------------------------------------------------------------------

def _unpack_mentions(components: dict) -> list[tuple[str, str, str]]:
    """Unpack extraction output into ``(attr_key, mention, snomed_category)`` triples.

    Handles regular attributes, list-of-strings, and paired
    ``interprets_interpretation`` structures.

    Args:
        components: Parsed extraction dict ``{attr_key: value | list | None}``.

    Returns:
        List of (attr_key, mention_text, snomed_category) tuples.
    """
    # Normalise non-canonical key names the LLM sometimes emits back to the
    # canonical pipeline keys so they are not silently dropped.
    _ALIASES: dict[str, str] = {
        "has_occurrence": "occurrence",
        "during": "occurrence",          # life-stage values (Congenital, Fetal period …)
        "has_finding_context": "finding_context",
        "has_relat_context": "subject_relationship_context",
        "has_related_context": "subject_relationship_context",
        "has_related": "subject_relationship_context",
    }

    mentions: list[tuple[str, str, str]] = []
    for raw_key, mention in components.items():
        if mention is None:
            continue

        attr_key = _ALIASES.get(raw_key, raw_key)

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
    attribute_index: AttributeIndex,
    cfg: HierarchyConfig,
    verbose: bool,
) -> pd.DataFrame:
    """Enrich candidates for a single attribute with reference values and hierarchy.

    Args:
        candidates: Initial candidates DataFrame for this attribute.
        attr_key: Attribute key (e.g. ``associated_morphology``).
        reference_values_by_attr: Reference values keyed by attr_key.
        attribute_index: Attribute searcher for hierarchy expansion.
        cfg: Pipeline configuration.
        verbose: Whether to log progress.

    Returns:
        Enriched candidates DataFrame.
    """
    snomed_category = ATTR_KEY_TO_SNOMED_CATEGORY.get(attr_key)

    # Enrich with values from reference examples not already in candidates
    if attr_key in reference_values_by_attr:
        existing_ids = set(candidates["concept_id"].tolist())
        mention = candidates["extracted_mention"].iloc[0]
        new_rows = [
            {"concept_id": rv["concept_id"], "concept_code": rv.get("concept_code"),
             "concept_name": rv["concept_name"],
             "attribute_category": snomed_category, "similarity": cfg.scoring.reference_similarity,
             "extracted_mention": mention, "attribute_key": attr_key}
            for rv in reference_values_by_attr[attr_key]
            if rv["concept_id"] not in existing_ids
        ]
        if new_rows:
            candidates = pd.concat([candidates, pd.DataFrame(new_rows)], ignore_index=True)
            if verbose:
                logger.info("  %s: added %d values from reference examples", attr_key, len(new_rows))

    # Enrich with 1-hop hierarchy neighbors (parents + children)
    if snomed_category:
        existing_ids = set(candidates["concept_id"].tolist())
        hierarchy_df = attribute_index.expand_via_hierarchy(
            list(existing_ids), snomed_category
        )
        new_hier = hierarchy_df[~hierarchy_df["concept_id"].isin(existing_ids)]
        if len(new_hier) > 0:
            mention = candidates["extracted_mention"].iloc[0]
            new_hier = new_hier.copy()
            new_hier["extracted_mention"] = mention
            new_hier["attribute_key"] = attr_key
            candidates = pd.concat([candidates, new_hier], ignore_index=True)
            if verbose:
                logger.info("  %s: added %d hierarchy neighbors", attr_key, len(new_hier))

    return candidates


def _retrieve_candidates(
    components: dict,
    attribute_index: AttributeIndex,
    similar_terms: list,
    verbose: bool,
    cfg: HierarchyConfig,
) -> SearchResult:
    """Step 3: Embed each inferred attribute value and retrieve SNOMED candidates.

    Args:
        components: Parsed extraction output ``{attr_key: [free-text values]}``.
        attribute_index: Attribute searcher (pgvector or legacy dict).
        similar_terms: Reference examples (for enrichment).
        verbose: Whether to log progress.
        cfg: Pipeline configuration.

    Returns:
        SearchResult(candidates_df, total_embedding_cost).
    """
    reference_values_by_attr = _collect_reference_values(similar_terms)
    mentions = _unpack_mentions(components)

    if not mentions:
        return SearchResult(pd.DataFrame(), 0.0)

    # Batch embed all mentions and search per-category
    indexed_mentions = [(f"{attr_key}_{i}", text, snomed_cat)
                        for i, (attr_key, text, snomed_cat) in enumerate(mentions)]
    results_by_idx, embedding_cost = attribute_index.search_batch(
        indexed_mentions, top_k=cfg.retrieval.top_k_per_category
    )

    # Group and deduplicate candidates per attr_key across multiple mentions
    candidates_by_attr: dict[str, pd.DataFrame] = {}
    for idx_key, (attr_key, mention, snomed_category) in zip(
        [m[0] for m in indexed_mentions], mentions
    ):
        candidates = results_by_idx.get(idx_key, pd.DataFrame())
        if len(candidates) == 0:
            continue

        candidates = candidates.copy()
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
            attribute_index, cfg, verbose,
        )
        all_candidates.append(candidates)
        if verbose:
            top = candidates.iloc[0]
            logger.info("  %s: top match = %s (score: %s)", attr_key, top["concept_name"],
                        top.get("similarity", "N/A"))

    candidates_df = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
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
    parts: list[str] = []
    paired_parts: dict[str, list[str]] = {}

    for attr_key, group in candidates_df.groupby("attribute_key", sort=False):
        mentions = group["extracted_mention"].unique().tolist()
        mention_str = "', '".join(mentions)
        lines = []
        for row in group.itertuples(index=False):
            sim_str = f", similarity: {row.similarity:.3f}" if row.similarity is not None else ""
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


def find_attributes_two_stage(
    medical_term: str,
    attribute_index: AttributeIndex,
    reference_index: ReferenceIndex | None = None,
    cfg: HierarchyConfig | None = None,
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
        attribute_index: Attribute searcher (pgvector or legacy dict).
        reference_index: Reference searcher (pgvector, legacy dict, or None).
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
    cfg = cfg or HierarchyConfig.from_yaml()

    # Step 1
    similar_terms, reference_text, ref_cost = _retrieve_reference_examples(
        medical_term, reference_index, cfg, verbose,
        precomputed_embedding=precomputed_embedding,
    )

    # Step 2
    if verbose:
        logger.info("Step 2: Inferring attributes...")
    components, extraction_cost = extract_components(medical_term, reference_text=reference_text,
                                                     cfg=cfg)
    if verbose:
        logger.info("  Inferred: %s", json.dumps({k: v for k, v in components.items() if v}, indent=2))

    # Step 3
    if verbose:
        logger.info("Step 3: Retrieving candidates...")
    candidates_df, embedding_cost = _retrieve_candidates(
        components, attribute_index, similar_terms,
        verbose, cfg=cfg
    )

    # Step 4
    if verbose:
        logger.info("Step 4: Selecting best matches...")
    candidates_text = _build_selection_prompt(candidates_df)
    user_prompt = f"Medical term: {medical_term}\n\n{reference_text}\n\nCandidates:\n{candidates_text}"
    response, selection_cost = call_llm(cfg.prompts.selection, user_prompt, model=cfg.models.selection)

    total_cost = ref_cost + extraction_cost + embedding_cost + selection_cost

    result = parse_json_response(response)

    # --- Enforce interprets ↔ interpretation pairing ---
    if "attributes" in result:
        _enforce_interprets_pairing(result["attributes"], verbose=verbose)

    # Inject concept_code into each selected attribute concept dict
    if len(candidates_df) > 0 and "concept_code" in candidates_df.columns:
        code_lookup: dict[int, str] = (
            candidates_df.dropna(subset=["concept_code"])
            .drop_duplicates(subset=["concept_id"])
            .set_index("concept_id")["concept_code"]
            .to_dict()
        )
        def _inject_code(obj):
            if isinstance(obj, dict) and "concept_id" in obj:
                cid = obj.get("concept_id")
                if cid is not None and "concept_code" not in obj:
                    obj["concept_code"] = code_lookup.get(int(cid))
        if "attributes" in result and isinstance(result["attributes"], dict):
            for attr_key, attr_val in result["attributes"].items():
                if attr_val is None:
                    continue
                if attr_key == "interprets_interpretation" and isinstance(attr_val, list):
                    for pair in attr_val:
                        if isinstance(pair, dict):
                            for v in pair.values():
                                _inject_code(v)
                elif isinstance(attr_val, list):
                    for item in attr_val:
                        _inject_code(item)
                else:
                    _inject_code(attr_val)

    result['extracted_components'] = components
    result['retrieved_candidates'] = candidates_df.to_dict('records') if len(candidates_df) > 0 else []
    if similar_terms:
        result['reference_examples'] = similar_terms
    result['cost'] = {
        'extraction_cost': extraction_cost,
        'embedding_cost': embedding_cost,
        'selection_cost': selection_cost,
        'total_cost': total_cost,
    }
    if verbose:
        logger.info("Total cost: $%.4f", total_cost)
    return result
