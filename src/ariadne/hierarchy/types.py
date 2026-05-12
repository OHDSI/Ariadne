"""Shared types and helpers for the SNOMED CT hierarchy pipeline.

NamedTuple result types provide self-documenting returns while remaining
backward-compatible with existing ``a, b = func()`` unpacking.

interprets ↔ interpretation helpers centralise the paired-attribute logic
that was previously duplicated across pipeline.py and evaluator.py.
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NamedTuple result types
# ---------------------------------------------------------------------------

class LlmResult(NamedTuple):
    """Return type for :func:`~ariadne.hierarchy.pipeline.call_llm`."""

    content: str
    cost: float


class SearchResult(NamedTuple):
    """Return type for attribute ``search`` and ``_retrieve_candidates``."""

    dataframe: Any  # pd.DataFrame
    cost: float


class SearchBatchResult(NamedTuple):
    """Return type for ``search_batch``."""

    results: dict  # dict[str, pd.DataFrame]
    cost: float


class ReferenceSearchResult(NamedTuple):
    """Return type for reference ``search`` and ``find_similar_reference_terms``."""

    examples: list
    cost: float


# ---------------------------------------------------------------------------
# interprets ↔ interpretation helpers
# ---------------------------------------------------------------------------

INTERPRETS_PAIRED_KEYS = frozenset({"interprets", "interpretation"})


def split_interprets_pairs(pairs: list) -> list[tuple[str, Any]]:
    """Split ``interprets_interpretation`` pairs into ``(attr_key, value)`` items.

    Works for both mention-string pairs (pipeline retrieval) and concept-dict
    pairs (evaluator prediction rows).

    Args:
        pairs: List of dicts with optional ``interprets`` and ``interpretation`` keys.

    Returns:
        List of ``(attr_key, value)`` tuples (value is whatever the dict contained).
    """
    items: list[tuple[str, Any]] = []
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        for key in ("interprets", "interpretation"):
            val = pair.get(key)
            if val is not None:
                items.append((key, val))
    return items


def merge_interprets_keys(
    interprets_list: list | None,
    interpretation_list: list | None,
    *,
    verbose: bool = False,
) -> list[dict] | None:
    """Merge separate ``interprets`` / ``interpretation`` lists into paired dicts.

    Handles the backward-compatibility case where the LLM returns separate top-level
    keys instead of the expected ``interprets_interpretation`` list of pairs.

    Args:
        interprets_list: Values for the interprets side (may be a single value).
        interpretation_list: Values for the interpretation side.
        verbose: Log warnings for dropped orphans.

    Returns:
        List of ``{interprets: ..., interpretation: ...}`` dicts, or ``None``.
    """
    if interprets_list and not isinstance(interprets_list, list):
        interprets_list = [interprets_list]
    if interpretation_list and not isinstance(interpretation_list, list):
        interpretation_list = [interpretation_list]

    if not interprets_list or not interpretation_list:
        if verbose and (interprets_list or interpretation_list):
            logger.warning("  Dropped orphaned interprets/interpretation (missing pair)")
        return None

    pairs: list[dict] = []
    for i in range(max(len(interprets_list), len(interpretation_list))):
        pair: dict = {}
        if i < len(interprets_list) and interprets_list[i]:
            pair["interprets"] = interprets_list[i]
        if i < len(interpretation_list) and interpretation_list[i]:
            pair["interpretation"] = interpretation_list[i]
        if pair:
            pairs.append(pair)

    return pairs if pairs else None


def validate_interprets_pairs(
    pairs: list,
    *,
    verbose: bool = False,
) -> list[dict] | None:
    """Drop ``interprets_interpretation`` pairs missing either side.

    Args:
        pairs: List of pair dicts to validate.
        verbose: Log warnings for dropped incomplete pairs.

    Returns:
        List of valid pairs, or ``None`` if all dropped.
    """
    valid: list[dict] = []
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        has_interp = pair.get("interprets") is not None
        has_interpr = pair.get("interpretation") is not None
        if has_interp and has_interpr:
            valid.append(pair)
        elif verbose:
            logger.warning("  Dropped incomplete pair: %s", pair)
    return valid if valid else None
