"""Four-step SNOMED CT attribute extraction pipeline.

Public API:
    extract_attributes(source_term, attribute_searcher, ...) → dict

Helpers handle individual steps:
    infer_attributes        — Step 2 (LLM infers attribute mentions)
    _retrieve_candidates    — Step 3 (embed mentions, retrieve SNOMED candidates)
    _build_selection_prompt — Step 4 (build LLM selection prompt)
"""

import json
import logging
import threading
from typing import cast, Tuple, List, Dict

import pandas as pd

from ariadne.hierarchy.mrcm import MrcmRules, build_grouping_prompt_section
from ariadne.hierarchy.role_grouper import INTERPRETS_PAIR_KEY, flatten_role_groups, group_result
from ariadne.hierarchy.searchers import SnomedAttributeSearcher, SnomedReferenceConceptVectorSearcher
from ariadne.hierarchy.types import (
    INTERPRETS_PAIRED_KEYS,
    LlmResult,
    SearchResult,
    merge_interprets_keys,
    split_interprets_pairs,
    validate_interprets_pairs,
)
from ariadne.utils.gen_ai_api import get_llm_response
from ariadne.utils.settings import HierarchySettings

logger = logging.getLogger(__name__)

# Type aliases
AttributeSearcher = SnomedAttributeSearcher
ReferenceSearcher = SnomedReferenceConceptVectorSearcher

# Normalize non-canonical keys the LLM may emit back to pipeline keys.
EXTRACTION_KEY_ALIASES: dict[str, str] = {
    "has_occurrence": "occurrence",
    "has_finding_context": "finding_context",
    "has_relat_context": "subject_relationship_context",
    "has_related_context": "subject_relationship_context",
    "has_related": "subject_relationship_context",
    "associated_with": "finding_asso_with",
    "finding_associated_with": "finding_asso_with",
    "after": "occurs_after",
}

CANDIDATE_COLUMNS = [
    "concept_id",
    "concept_code",
    "concept_name",
    "attribute_category",
    "similarity",
    "extracted_mention",
    "attribute_key",
]


ROLE_GROUPS_KEY = "role_groups"


class ContentFilterError(Exception):
    """Raised when the LLM content filter blocks a response."""


# ---------------------------------------------------------------------------
# MRCM / role-group helpers
# ---------------------------------------------------------------------------

_MRCM_RULES_CACHE: dict[tuple, MrcmRules | None] = {}
_MRCM_RULES_LOCK = threading.Lock()


def _mrcm_settings(hierarchy_settings):
    return getattr(hierarchy_settings, "mrcm", None)


def _load_mrcm_rules(hierarchy_settings) -> MrcmRules | None:
    """Load (and cache) MRCM rules; returns ``None`` if the MRCM files are unavailable."""
    settings = _mrcm_settings(hierarchy_settings)
    if settings is None or not settings.domains:
        return None
    attributes = getattr(hierarchy_settings, "attributes", {}) or {}
    cache_key = (
        settings.folder,
        tuple(sorted(settings.domains.items())),
        tuple(settings.content_types),
        tuple(sorted((key, a.attribute_id) for key, a in attributes.items())),
    )
    with _MRCM_RULES_LOCK:
        if cache_key not in _MRCM_RULES_CACHE:
            try:
                _MRCM_RULES_CACHE[cache_key] = MrcmRules.load(settings, attributes)
            except (FileNotFoundError, ValueError) as exc:
                logger.warning("MRCM rules unavailable (%s); grouping without MRCM domain rules.", exc)
                _MRCM_RULES_CACHE[cache_key] = None
        return _MRCM_RULES_CACHE[cache_key]


def _fill_prompt(template: str, reference_text: str, hierarchy_settings) -> str:
    settings = _mrcm_settings(hierarchy_settings)
    grouping_text = build_grouping_prompt_section(settings) if settings is not None else ""
    return template.replace("{reference_section}", reference_text).replace("{grouping_section}", grouping_text)


def _normalize_role_groups(raw_groups: object) -> list[dict]:
    """Normalize LLM ``role_groups`` to ``[{attribute_key: value}]`` with canonical keys."""
    if not isinstance(raw_groups, list):
        return []
    groups = []
    for raw in raw_groups:
        if not isinstance(raw, dict):
            continue
        group = {
            EXTRACTION_KEY_ALIASES.get(key, key): value
            for key, value in raw.items()
            if value not in (None, "", [])
        }
        if group:
            groups.append(group)
    return groups


def _format_extracted_role_groups(components: dict) -> str:
    """Render the extraction-step role groups (mentions) for the selection prompt."""
    groups = _normalize_role_groups(components.get(ROLE_GROUPS_KEY))
    if not groups:
        return ""
    lines = ["Proposed role groups (from extraction):"]
    for i, group in enumerate(groups, start=1):
        parts = []
        for key, value in group.items():
            mentions = value if isinstance(value, list) else [value]
            parts.append(f"{key} = {' | '.join(str(m) for m in mentions)}")
        lines.append(f"  group {i}: " + "; ".join(parts))
    return "\n".join(lines)


def _merge_role_groups_into_attributes(result: dict) -> list[dict]:
    """Move selected ``role_groups`` out of the LLM result and merge them into ``attributes``.

    Accepts ``role_groups`` at the top level or inside ``attributes``. The flat
    ``attributes`` view (used by evaluation) is updated in place; the normalized groups
    are returned.
    """
    attributes = result.setdefault("attributes", {}) or {}
    result["attributes"] = attributes
    raw = result.pop(ROLE_GROUPS_KEY, None)
    nested = attributes.pop(ROLE_GROUPS_KEY, None)
    groups = _normalize_role_groups(raw) + _normalize_role_groups(nested)
    for key, concepts in flatten_role_groups(groups).items():
        existing = attributes.get(key)
        existing = [] if existing is None else (existing if isinstance(existing, list) else [existing])
        seen = {str(c.get("concept_id")) for c in existing if isinstance(c, dict)}
        attributes[key] = existing + [c for c in concepts if str(c["concept_id"]) not in seen]
    return groups


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


def format_reference_examples(
    similar_terms: list[dict], hierarchy_settings: HierarchySettings, include_concept_ids=False
) -> str:
    """Format reference examples into a human-readable block for the prompt.

    Args:
        similar_terms: List of reference dicts from the reference searcher.
        hierarchy_settings: Pipeline configuration; ``hierarchy.attributes`` names each relationship
            by its pipeline attribute key (e.g. ``"Has pathology"`` → ``pathological_process``).
            Relationships it doesn't list are shown by their OHDSI name.
        include_concept_ids: Append concept IDs to terms and attribute values.

    Returns:
        Formatted string, or empty string if no terms.
    """
    if not similar_terms:
        return ""
    attribute_key = hierarchy_settings.attribute_by_relationship()
    examples = []
    for term in similar_terms:
        attrs_text = []
        for a in term["attributes"]:
            line = f"  - {attribute_key.get(a['attribute_category'], a['attribute_category'])}: {a['target_concept_name']}"
            attrs_text.append(f"{line} ({a['target_concept_id']})" if include_concept_ids else line)
        attrs_str = "\n".join(attrs_text) if attrs_text else "  (no attributes)"
        if include_concept_ids:
            examples.append(f"Term: {term['concept_name']} ({term['concept_id']})\nAttributes:\n{attrs_str}")
        else:
            examples.append(f"Term: {term['concept_name']}\nAttributes:\n{attrs_str}")
    if not examples:
        return ""
    else:
        return "=== SIMILAR EXAMPLES ===\n" + "\n\n".join(examples)


def infer_attributes(
    source_term: str,
    reference_examples: List,
    hierarchy_settings: HierarchySettings,
) -> Tuple[Dict, float]:
    """Step 2: Use the LLM to infer applicable SNOMED attributes.

    Args:
        source_term: Term to decompose.
        reference_examples: Reference examples.
        hierarchy_settings: Pipeline configuration.

    Returns:
        ExtractionResult(components, cost).
    """
    reference_text = format_reference_examples(reference_examples, hierarchy_settings, include_concept_ids=False)
    system_prompt = _fill_prompt(hierarchy_settings.prompts.extraction, reference_text, hierarchy_settings)
    user_prompt = f'Determine the attributes for: "{source_term}"'
    response, cost = call_llm(system_prompt, user_prompt)
    return parse_json_response(response), cost




def _unpack_mentions(components: dict[str, object], hierarchy_settings: HierarchySettings) -> list[tuple[str, str, str]]:
    """Unpack extraction output into ``(attr_key, mention, snomed_category)`` triples.

    Handles regular attributes, list-of-strings, paired
    ``interprets_interpretation`` structures and ``role_groups`` (list of
    ``{attr_key: mention}`` dicts). Duplicate mentions are dropped.

    Args:
        components: Parsed extraction dict ``{attr_key: value | list | None}``.
        hierarchy_settings: Pipeline configuration; ``hierarchy.attributes`` maps each attribute key
            to its OHDSI relationship (``snomed_category``). Keys it doesn't list are skipped.

    Returns:
        List of (attr_key, mention_text, snomed_category) tuples.
    """
    relationship_for = hierarchy_settings.relationship_by_attribute()
    mentions: list[tuple[str, str, str]] = []
    items: list[tuple[str, object]] = []
    for raw_key, mention in components.items():
        if raw_key == ROLE_GROUPS_KEY:
            for group in _normalize_role_groups(mention):
                items.extend(group.items())
        else:
            items.append((raw_key, mention))

    for raw_key, mention in items:
        if mention is None:
            continue

        attr_key = cast(str, EXTRACTION_KEY_ALIASES.get(raw_key, raw_key))

        # Handle paired interprets_interpretation tuples
        if attr_key == "interprets_interpretation":
            if isinstance(mention, list):
                for sub_key, val in split_interprets_pairs(mention):
                    sc = relationship_for.get(sub_key)
                    if sc:
                        mentions.append((sub_key, str(val), sc))
            continue

        relationship_id = relationship_for.get(attr_key)
        if relationship_id is None:
            continue
        # Top-level attributes are lists; role_groups entries are single strings
        if isinstance(mention, list):
            for item in mention:
                if item:
                    mentions.append((attr_key, str(item), relationship_id))
        else:
            mentions.append((attr_key, str(mention), relationship_id))
    return list(dict.fromkeys(mentions))


def _retrieve_candidates(
    extracted_components: dict,
    attribute_searcher: AttributeSearcher,
    hierarchy_settings: HierarchySettings
) -> SearchResult:
    """Step 3: Embed each inferred attribute value and retrieve SNOMED candidates.

    Args:
        extracted_components: Parsed extraction output ``{attr_key: [free-text values]}``.
        attribute_searcher: ``SnomedAttributeSearcher`` over the ``snomed_attribute`` table.
        hierarchy_settings: Pipeline configuration.

    Returns:
        SearchResult(candidates_df, total_embedding_cost).
    """
    mentions = _unpack_mentions(extracted_components, hierarchy_settings)

    if not mentions:
        return SearchResult(pd.DataFrame(columns=CANDIDATE_COLUMNS), 0.0)

    # Batch embed all mentions and search per-category
    indexed_mentions = [(f"{attr_key}_{i}", text, snomed_cat)
                        for i, (attr_key, text, snomed_cat) in enumerate(mentions)]
    results_by_idx, embedding_cost = attribute_searcher.search_batch(
        indexed_mentions, top_k=hierarchy_settings.retrieval.top_k_per_category
    )

    # Group and deduplicate candidates per attr_key across multiple mentions
    candidates_by_attr: dict[str, pd.DataFrame] = {}
    for idx_key, (attr_key, mention, _) in zip(
        [m[0] for m in indexed_mentions], mentions
    ):
        candidates = results_by_idx.get(idx_key, pd.DataFrame(columns=CANDIDATE_COLUMNS))
        if len(candidates) == 0:
            continue

        candidates["extracted_mention"] = mention
        candidates["attribute_key"] = attr_key

        if attr_key in candidates_by_attr:
            candidates_by_attr[attr_key] = pd.concat(
                [candidates_by_attr[attr_key], candidates], ignore_index=True
            ).drop_duplicates(subset=["concept_id"], keep="first")
        else:
            candidates_by_attr[attr_key] = candidates

    all_candidates = list(candidates_by_attr.values())

    candidates_df = (
        pd.concat(all_candidates, ignore_index=True)
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

    parts: list[str] = []
    paired_parts: dict[str, list[str]] = {}

    for group_key, group in candidates_df.groupby("attribute_key", sort=False):
        attr_key = cast(str, str(group_key))
        mentions = [str(m) for m in group["extracted_mention"].unique().tolist()]
        mention_str = "', '".join(mentions)
        lines = []
        for row in group.itertuples(index=False):
            lines.append(f"  - {row.concept_name} (concept_id: {row.concept_id})")

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


def _enforce_interprets_pairing(attrs: dict) -> None:
    """Normalise interprets/interpretation keys into paired ``interprets_interpretation``.

    Operates **in-place** on *attrs*.  Handles:
    - Validation that each pair has both sides.
    - Cleanup of leftover separate keys.
    """
    if ("interprets" in attrs or "interpretation" in attrs) and "interprets_interpretation" not in attrs:
        merged = merge_interprets_keys(
            attrs.pop("interprets", None),
            attrs.pop("interpretation", None)
        )
        if merged:
            attrs["interprets_interpretation"] = merged

    if "interprets_interpretation" in attrs and attrs["interprets_interpretation"]:
        attrs["interprets_interpretation"] = validate_interprets_pairs(
            attrs["interprets_interpretation"]
        )

    # Clean up any leftover separate keys
    attrs.pop("interprets", None)
    attrs.pop("interpretation", None)


def _inject_selected_concept_codes(result: dict, candidates_df: pd.DataFrame) -> None:
    """Inject ``concept_code`` into selected attribute dicts in-place when missing or null."""
    if len(candidates_df) == 0 or "concept_code" not in candidates_df.columns:
        return

    # Key by string concept_id so lookups work whether the LLM/DataFrame ids are int or str.
    code_lookup: dict[str, str] = {
        str(cid): code
        for cid, code in candidates_df.dropna(subset=["concept_code"])
        .drop_duplicates(subset=["concept_id"])
        .set_index("concept_id")["concept_code"]
        .to_dict()
        .items()
    }

    def _inject_code(obj):
        if isinstance(obj, dict) and "concept_id" in obj:
            concept_id = obj.get("concept_id")
            # Fill when the LLM omitted the code or emitted an explicit null.
            if concept_id is not None and obj.get("concept_code") in (None, ""):
                obj["concept_code"] = code_lookup.get(str(concept_id))

    if "attributes" not in result or not isinstance(result["attributes"], dict):
        return

    for group in result.get("llm_role_groups") or []:
        for value in group.values():
            for item in value if isinstance(value, list) else [value]:
                _inject_code(item)

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


def drop_non_candidate_values(result: dict, candidates: pd.DataFrame | list[dict]) -> list[dict]:
    """Remove selected concepts that were not offered as candidates for their attribute (in place).

    The selection prompt only allows candidates, but the LLM can copy a concept from the reference
    examples instead. Such values are removed from ``attributes``, ``llm_role_groups`` and
    ``role_groups``, and recorded in ``result["dropped_selections"]``.

    Args:
        result: Pipeline result with the selected concepts.
        candidates: The retrieved candidates (``candidates_df`` or ``result["retrieved_candidates"]``).

    Returns:
        The dropped values as ``{"attribute", "concept_id", "concept_name"}`` dicts.
    """
    rows = candidates.to_dict("records") if isinstance(candidates, pd.DataFrame) else list(candidates or [])
    allowed: dict[str, set[str]] = {}
    for row in rows:
        allowed.setdefault(str(row.get("attribute_key")), set()).add(str(row.get("concept_id")))

    dropped: dict[tuple[str, str], dict] = {}

    def keep(key: str, concept) -> bool:
        if not isinstance(concept, dict) or concept.get("concept_id") in (None, ""):
            return True
        concept_id = str(concept["concept_id"])
        if concept_id in allowed.get(EXTRACTION_KEY_ALIASES.get(key, key), set()):
            return True
        entry = dropped.setdefault(
            (key, concept_id), {"attribute": key, "concept_id": concept_id, "concept_name": None}
        )
        entry["concept_name"] = entry["concept_name"] or concept.get("concept_name")
        return False

    attributes = result.get("attributes")
    if isinstance(attributes, dict):
        for key, value in list(attributes.items()):
            if value is None:
                continue
            if key == INTERPRETS_PAIR_KEY and isinstance(value, list):
                attributes[key] = [
                    pair for pair in value
                    if not isinstance(pair, dict) or all([keep(k, v) for k, v in pair.items()])
                ]
            elif isinstance(value, list):
                attributes[key] = [concept for concept in value if keep(key, concept)]
            elif not keep(key, value):
                attributes[key] = None

    if "llm_role_groups" in result:
        groups = []
        for group in result.get("llm_role_groups") or []:
            if not isinstance(group, dict):
                groups.append(group)
                continue
            kept = {}
            for key, value in group.items():
                if isinstance(value, list):
                    kept[key] = [concept for concept in value if keep(key, concept)]
                elif keep(key, value):
                    kept[key] = value
            if any(kept.values()):
                groups.append(kept)
        if groups:
            result["llm_role_groups"] = groups
        else:
            result.pop("llm_role_groups")

    if "role_groups" in result:
        role_groups = []
        for group in result.get("role_groups") or []:
            members = [m for m in group.get("members", []) if keep(str(m.get("attribute")), m)]
            if members:
                role_groups.append({**group, "members": members})
        result["role_groups"] = role_groups

    if dropped:
        result["dropped_selections"] = result.get("dropped_selections", []) + list(dropped.values())
    return list(dropped.values())


def assign_role_groups(result: dict, hierarchy_settings) -> None:
    """Group the selected attributes with the cached MRCM rules (in place); no-op when grouping is disabled."""
    mrcm_settings = _mrcm_settings(hierarchy_settings)
    if mrcm_settings is None or not mrcm_settings.grouping.enabled:
        return
    group_result(result, mrcm_settings, _load_mrcm_rules(hierarchy_settings))


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


def extract_attributes(
    source_term: str,
    attribute_searcher: AttributeSearcher,
    reference_searcher: ReferenceSearcher,
    hierarchy_settings: HierarchySettings | None = None
) -> dict:
    """Run the 4-step SNOMED CT attribute extraction pipeline.

    Steps:
        1. Retrieve reference examples (similar SNOMED concepts and their attributes).
        2. LLM infers applicable attributes.
        3. Retrieve SNOMED candidate values per attribute.
        4. LLM selects exact SNOMED concepts from candidates.

    Args:
        source_term: The clinical term to decompose.
        attribute_searcher: ``SnomedAttributeSearcher`` over the ``snomed_attribute`` table.
        reference_searcher: ``SnomedReferenceConceptVectorSearcher`` supplying few-shot
            reference examples.
        hierarchy_settings: Pipeline configuration.

    Returns:
        Dict with keys ``attributes``, ``role_groups``, ``grouping_method``,
        ``extracted_components``, ``retrieved_candidates``, ``reference_examples``,
        ``cost`` (plus ``llm_role_groups``/``mrcm_domain`` when available).
    """

    # Step 1
    logger.debug("Step 1: Finding reference examples")
    reference_examples, ref_cost = reference_searcher.search(source_term,
                                                             top_k=hierarchy_settings.retrieval.num_reference_examples)

    # Step 2
    logger.debug("Step 2: Inferring attributes")
    components, extraction_cost = infer_attributes(source_term,
                                                   reference_examples=reference_examples,
                                                   hierarchy_settings=hierarchy_settings)

    # Step 3
    logger.debug("Step 3: Retrieving candidates...")
    candidates_df, embedding_cost = _retrieve_candidates(
        components, attribute_searcher,
        hierarchy_settings=hierarchy_settings
    )

    # Step 4
    logger.debug("Step 4: Selecting best matches")
    candidates_text = _build_selection_prompt(candidates_df)
    selection_cost = 0.0
    if candidates_text.strip():
        reference_text = format_reference_examples(reference_examples, hierarchy_settings, include_concept_ids=True)
        system_prompt = _fill_prompt(hierarchy_settings.prompts.selection, reference_text, hierarchy_settings)
        user_prompt = f"Medical term: {source_term}\n\nCandidate concepts:\n{candidates_text}"
        extracted_groups_text = _format_extracted_role_groups(components)
        if extracted_groups_text:
            user_prompt += f"\n\n{extracted_groups_text}"
        response, selection_cost = call_llm(system_prompt, user_prompt)
        result = parse_json_response(response)
    else:
        logger.debug("Step 4 skipped: no candidates available for selection.")
        result = {"attributes": {}}

    total_cost = ref_cost + extraction_cost + embedding_cost + selection_cost

    llm_role_groups = _merge_role_groups_into_attributes(result)
    if llm_role_groups:
        result["llm_role_groups"] = llm_role_groups

    dropped = drop_non_candidate_values(result, candidates_df)
    if dropped:
        logger.warning(
            "%s: dropped selected value(s) not among the candidates: %s",
            source_term,
            "; ".join(f"{d['attribute']}={d['concept_name']} ({d['concept_id']})" for d in dropped),
        )

    # --- Enforce interprets ↔ interpretation pairing ---
    if "attributes" in result:
        _enforce_interprets_pairing(result["attributes"])

    _inject_selected_concept_codes(result, candidates_df)

    assign_role_groups(result, hierarchy_settings)
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
    logger.debug("Total cost: $%.4f", total_cost)
    return result
