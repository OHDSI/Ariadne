"""Evaluate selected stated parents against the proximal primitive parents of held-out SNOMED concepts.

For a held-out concept the reference is its proximal primitive parent set: the most specific
primitive ancestors (within the allowed parent set) in the inferred hierarchy. Outcomes:

- ``exact``: the selected parents equal the reference set.
- ``partial``: at least one reference parent selected; every other selected parent is an ancestor
  of a reference parent.
- ``too_general``: every selected parent is a proper ancestor of a reference parent. The classifier
  places the concept too high but not in the wrong branch.
- ``wrong``: at least one selected parent is not an ancestor-or-self of any reference parent, so the
  classifier can place the concept in the wrong branch.
- ``no_reference``: the source has no primitive ancestor in the allowed set (e.g. a situation) or
  no OMOP concept_id.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import psycopg

from ariadne.hierarchy.parent_selector import (
    PrimitiveAncestry,
    fetch_primitive_ancestry,
    proximal_primitive_parents,
)
from ariadne.utils.settings import ParentSelectionConfig

OUTCOMES = ["exact", "partial", "too_general", "wrong", "no_reference"]


def _concept_id(key) -> int | None:
    return int(key) if isinstance(key, (int, np.integer)) else None


def classify_parents(predicted: set[int], expected: set[int], ancestry: PrimitiveAncestry) -> str:
    """Classify a selected parent set against the reference proximal primitive parents."""
    if not expected:
        return "no_reference"
    if predicted == expected:
        return "exact"

    def covers(parent_id: int) -> bool:
        return any(parent_id == e or ancestry.is_ancestor(parent_id, e) for e in expected)

    if not all(covers(p) for p in predicted):
        return "wrong"
    return "partial" if predicted & expected else "too_general"


def evaluate_stated_parents(
    details: pd.DataFrame,
    conn: psycopg.Connection,
    schema: str,
    settings: ParentSelectionConfig,
) -> pd.DataFrame:
    """Compare ``build_stated_parents_map`` details with the held-out concepts' proximal primitive parents.

    Returns:
        One row per source with selected and reference parents and the outcome.
    """
    selected = details.groupby("source_concept_id", sort=False).agg(
        source_term=("source_term", "first"),
        parent_ids=("parent_concept_id", lambda s: set(int(v) for v in s)),
        parent_names=("parent_concept_name", lambda s: "; ".join(s)),
    )
    source_ids = [cid for key in selected.index if (cid := _concept_id(key)) is not None]
    ancestry = fetch_primitive_ancestry(source_ids, conn, schema, settings)

    rows = []
    for key, row in selected.iterrows():
        cid = _concept_id(key)
        expected = proximal_primitive_parents(cid, ancestry) if cid is not None else []
        rows.append(
            {
                "source_concept_id": key,
                "source_term": row["source_term"],
                "selected_parents": row["parent_names"],
                "reference_parents": "; ".join(c.concept_name for c in expected),
                "outcome": classify_parents(row["parent_ids"], {c.concept_id for c in expected}, ancestry),
            }
        )
    return pd.DataFrame(rows)
