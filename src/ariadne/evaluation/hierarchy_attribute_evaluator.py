"""Evaluation utilities for the SNOMED CT attribute extraction pipeline.

Public API:
    build_prediction_rows — flatten pipeline output into row-wise predictions.
    evaluate_results      — full outer join evaluation producing P/R/F1.
"""

import logging
import os
from pathlib import Path

import pandas as pd

from ariadne.hierarchy.llm_attribute_extractor import EXTRACTION_KEY_ALIASES
from ariadne.hierarchy.role_grouper import group_attributes
from ariadne.hierarchy.types import split_interprets_pairs
from ariadne.utils.settings import HierarchySettings

logger = logging.getLogger(__name__)


def _role_group_lookup(role_groups: list[dict] | None) -> dict[tuple[str, str], str]:
    """Map ``(attribute_key, concept_id)`` to its role group numbers, e.g. ``"1;2"``."""
    groups: dict[tuple[str, str], set[int]] = {}
    for group in role_groups or []:
        for member in group.get("members", []):
            key = (str(member.get("attribute")), str(member.get("concept_id")))
            groups.setdefault(key, set()).add(int(group.get("group", 0)))
    return {key: ";".join(str(g) for g in sorted(numbers)) for key, numbers in groups.items()}


def build_prediction_rows(results: list[dict], hierarchy_settings: HierarchySettings) -> list[dict]:
    """Extract prediction rows from pipeline results for evaluation.

    Handles regular attributes (single dict, list of dicts) and paired
    ``interprets_interpretation`` structures via :func:`split_interprets_pairs`.
    Each row carries ``relationship_groups``: the SNOMED role group number(s) of the
    attribute value, ``;``-separated (a value shared by several groups appears once).

    Args:
        results: Pipeline result dicts from ``process_hierarchy``.
        hierarchy_settings: Pipeline configuration. ``hierarchy.attributes`` gives each attribute
            key's OHDSI relationship (``attribute_category``). When ``hierarchy.mrcm.grouping`` is
            configured, results without ``role_groups`` (e.g. cached before grouping existed) are
            grouped on the fly; otherwise their ``relationship_groups`` is ``None``.

    Returns:
        List of flat dicts ready for ``pd.DataFrame``.
    """
    relationship_for = hierarchy_settings.relationship_by_attribute()
    mrcm_settings = hierarchy_settings.mrcm

    def _category(attr_key: str, default: str) -> str:
        return relationship_for.get(EXTRACTION_KEY_ALIASES.get(attr_key, attr_key), default)

    pred_rows: list[dict] = []
    for result in results:
        source_meta = {
            "source_key": result.get("source_key"),
            "source_concept_id": result.get("source_concept_id"),
            "source_concept_code": result.get("source_concept_code"),
            "source_vocabulary_id": result.get("source_vocabulary_id"),
            "source_term": result.get("source_term"),
        }
        if "attributes" not in result:
            continue
        role_groups = result.get("role_groups")
        if role_groups is None and mrcm_settings.grouping.enabled:
            role_groups = group_attributes(
                result["attributes"], mrcm_settings, result.get("llm_role_groups")
            ).role_groups
        group_lookup = _role_group_lookup(role_groups)

        def _groups(key: str, concept: dict) -> str | None:
            return group_lookup.get((key, str(concept.get("concept_id"))))
        for attr_key, attr_value in result["attributes"].items():
            if attr_value is None:
                continue

            # Handle paired interprets_interpretation tuples
            if attr_key == "interprets_interpretation":
                if not isinstance(attr_value, list):
                    attr_value = [attr_value]
                for sub_key, concept in split_interprets_pairs(attr_value):
                    if isinstance(concept, dict):
                        pred_rows.append({
                            **source_meta,
                            "predicted_target_concept_id": concept.get("concept_id"),
                            "predicted_target_concept_name": concept.get("concept_name"),
                            "predicted_target_concept_code": concept.get("concept_code"),
                            "attribute_category": _category(sub_key, f"Has {sub_key}"),
                            "relationship_groups": _groups(sub_key, concept),
                        })
                continue

            attr_type = _category(attr_key, attr_key)
            # Handle list of concept dicts (new multi-value format)
            if isinstance(attr_value, list):
                for item in attr_value:
                    if item and isinstance(item, dict):
                        pred_rows.append({
                            **source_meta,
                            "predicted_target_concept_id": item.get("concept_id"),
                            "predicted_target_concept_name": item.get("concept_name"),
                            "predicted_target_concept_code": item.get("concept_code"),
                            "attribute_category": attr_type,
                            "relationship_groups": _groups(attr_key, item),
                        })
            # Handle single concept dict
            elif isinstance(attr_value, dict):
                pred_rows.append({
                    **source_meta,
                    "predicted_target_concept_id": attr_value.get("concept_id"),
                    "predicted_target_concept_name": attr_value.get("concept_name"),
                    "predicted_target_concept_code": attr_value.get("concept_code"),
                    "attribute_category": attr_type,
                    "relationship_groups": _groups(attr_key, attr_value),
                })
    return pred_rows


def evaluate_results(
    results: list[dict],
    gs_path: str,
    output_dir: Path,
    hierarchy_settings: HierarchySettings,
) -> pd.DataFrame:
    """Produce a combined evaluation table (full outer join of GS and predictions).

    Columns:
        source_key, source_concept_id, source_concept_code, source_vocabulary_id,
        source_term, attribute_category,
        gs_target_concept_id, gs_target_concept_code, gs_target_concept_name,
        predicted_target_concept_id, predicted_target_concept_code, predicted_target_concept_name,
        matched, status (``match`` / ``missed`` / ``extra``).

    Summary statistics are printed and the combined table is saved to CSV.

    Args:
        results: List of pipeline result dicts from ``process_hierarchy``.
        gs_path: Path to the gold-standard CSV.
        output_dir: Path wherhe the output CSV should be saved.
        hierarchy_settings: Pipeline configuration (see :func:`build_prediction_rows`).

    Returns:
        Combined evaluation DataFrame.
    """
    # --- build predicted rows ---
    pred_rows = build_prediction_rows(results, hierarchy_settings)
    pred_df = pd.DataFrame(pred_rows)
    source_cols = ["source_key", "source_concept_id", "source_concept_code",
                   "source_vocabulary_id", "source_term"]
    if pred_df.empty:
        pred_df = pd.DataFrame(columns=source_cols + [
            "predicted_target_concept_id", "predicted_target_concept_name",
            "predicted_target_concept_code", "attribute_category", "relationship_groups"])

    # --- load gold standard ---
    gs_df = pd.read_csv(gs_path)
    gs_df = gs_df.rename(columns={
        'target_concept_id': 'gs_target_concept_id',
        'target_concept_code': 'gs_target_concept_code',
        'target_concept_name': 'gs_target_concept_name',
    })

    # --- composite source key + numeric target join key ---
    for df in (gs_df, pred_df):
        df['source_key'] = (
            df['source_vocabulary_id'].astype(str) + ':' + df['source_concept_code'].astype(str)
        )
    gs_df['_join_target'] = pd.to_numeric(gs_df['gs_target_concept_id'], errors='coerce').astype('Int64')
    pred_df['_join_target'] = pd.to_numeric(pred_df['predicted_target_concept_id'], errors='coerce').astype('Int64')

    combined = gs_df.merge(
        pred_df,
        on=['source_key', '_join_target', 'attribute_category'],
        how='outer',
        suffixes=('_gs', '_pred'),
    )

    # Reconcile source descriptor columns from both sides (GS wins, pred fills gaps)
    for col in ('source_concept_id', 'source_concept_code', 'source_vocabulary_id', 'source_term'):
        gs_col, pred_col = f'{col}_gs', f'{col}_pred'
        if gs_col in combined.columns:
            combined[col] = combined[gs_col].fillna(combined[pred_col])
            combined.drop(columns=[gs_col, pred_col], inplace=True)

    combined.drop(columns=['_join_target'], inplace=True)

    # --- flags ---
    has_gs = combined['gs_target_concept_id'].notna()
    has_pred = combined['predicted_target_concept_id'].notna()
    combined['matched'] = has_gs & has_pred
    combined['status'] = 'match'
    combined.loc[has_gs & ~has_pred, 'status'] = 'missed'
    combined.loc[~has_gs & has_pred, 'status'] = 'extra'

    # --- order columns nicely ---
    leading = ['source_key', 'source_concept_id', 'source_concept_code',
               'source_vocabulary_id', 'source_term', 'attribute_category',
               'gs_target_concept_id', 'gs_target_concept_code', 'gs_target_concept_name',
               'predicted_target_concept_id', 'predicted_target_concept_code', 'predicted_target_concept_name',
               'matched', 'status']
    extra_cols = [c for c in combined.columns if c not in leading]
    combined = combined[[c for c in leading if c in combined.columns] + extra_cols]

    # Sort for readability
    combined = combined.sort_values(['source_key', 'attribute_category', 'status']).reset_index(drop=True)

    # --- summary stats ---
    n_gs = int(has_gs.sum())
    n_pred = int(has_pred.sum())
    n_match = int(combined['matched'].sum())
    precision = n_match / n_pred * 100 if n_pred else 0.0
    recall = n_match / n_gs * 100 if n_gs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    logger.info("Gold standard rows: %d", n_gs)
    logger.info("Predicted rows:     %d", n_pred)
    logger.info("Matched:            %d", n_match)
    logger.info("Precision:          %.1f%%", precision)
    logger.info("Recall:             %.1f%%", recall)
    logger.info("F1:                 %.1f%%", f1)

    # --- save ---
    out_path = os.path.join(output_dir, "hierarchy_attribute_evaluation.csv")
    combined.to_csv(out_path, index=False)
    logger.info("Combined evaluation saved: %s (%d rows)", out_path, len(combined))
    return combined
