"""Evaluation utilities for the SNOMED CT attribute extraction pipeline.

Public API:
    build_prediction_rows — flatten pipeline output into row-wise predictions.
    evaluate_results      — full outer join evaluation producing P/R/F1.
"""

import logging
import os

import pandas as pd

from ariadne.hierarchy.searchers import ATTR_KEY_TO_GS_CATEGORY
from ariadne.hierarchy.types import split_interprets_pairs
from ariadne.utils.config import load_hierarchy_settings
from ariadne.utils.settings import HierarchySettings

logger = logging.getLogger(__name__)

def build_prediction_rows(results: list[dict]) -> list[dict]:
    """Extract prediction rows from pipeline results for evaluation.

    Handles regular attributes (single dict, list of dicts) and paired
    ``interprets_interpretation`` structures via :func:`split_interprets_pairs`.

    Args:
        results: Pipeline result dicts from ``process_hierarchy``.

    Returns:
        List of flat dicts ready for ``pd.DataFrame``.
    """
    pred_rows: list[dict] = []
    for result in results:
        concept_id_1 = result.get("source_concept_id")
        concept_name_1 = result.get("source_concept_name") or result.get("medical_term")
        if "attributes" not in result:
            continue
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
                            "concept_id_1": concept_id_1,
                            "concept_name_1": concept_name_1,
                            "predicted_concept_id_2": concept.get("concept_id"),
                            "predicted_concept_name_2": concept.get("concept_name"),
                            "predicted_concept_code_2": concept.get("concept_code"),
                            "attribute_category": ATTR_KEY_TO_GS_CATEGORY.get(sub_key, f"Has {sub_key}"),
                        })
                continue

            attr_type = ATTR_KEY_TO_GS_CATEGORY.get(attr_key, attr_key)
            # Handle list of concept dicts (new multi-value format)
            if isinstance(attr_value, list):
                for item in attr_value:
                    if item and isinstance(item, dict):
                        pred_rows.append({
                            "concept_id_1": concept_id_1,
                            "concept_name_1": concept_name_1,
                            "predicted_concept_id_2": item.get("concept_id"),
                            "predicted_concept_name_2": item.get("concept_name"),
                            "predicted_concept_code_2": item.get("concept_code"),
                            "attribute_category": attr_type,
                        })
            # Handle single concept dict (legacy format)
            elif isinstance(attr_value, dict):
                pred_rows.append({
                    "concept_id_1": concept_id_1,
                    "concept_name_1": concept_name_1,
                    "predicted_concept_id_2": attr_value.get("concept_id"),
                    "predicted_concept_name_2": attr_value.get("concept_name"),
                    "predicted_concept_code_2": attr_value.get("concept_code"),
                    "attribute_category": attr_type,
                })
    return pred_rows


def evaluate_results(
    results: list[dict],
    gs_path: str,
    cfg: HierarchySettings | None = None,
) -> pd.DataFrame:
    """Produce a combined evaluation table (full outer join of GS and predictions).

    Columns:
        concept_id_1, concept_name_1, attribute_category,
        gs_concept_id_2, gs_concept_name_2,
        predicted_concept_id_2, predicted_concept_name_2,
        matched, status (``match`` / ``missed`` / ``extra``).

    Summary statistics are printed and the combined table is saved to CSV.

    Args:
        results: List of pipeline result dicts from ``process_hierarchy``.
        gs_path: Path to the gold-standard CSV.
        cfg: Pipeline configuration (reads ``cfg.evaluation.output_dir``).

    Returns:
        Combined evaluation DataFrame.
    """
    cfg_local: HierarchySettings = cfg if cfg is not None else load_hierarchy_settings()
    output_dir = cfg_local.evaluation.output_dir
    # --- build predicted rows ---
    pred_rows = build_prediction_rows(results)
    pred_df = pd.DataFrame(pred_rows)

    # --- load gold standard ---
    gs_df = pd.read_csv(gs_path)
    gs_df = gs_df.rename(columns={'concept_id_2': 'gs_concept_id_2', 'concept_code_2': 'gs_concept_code_2', 'concept_name_2': 'gs_concept_name_2'})

    # --- full outer join on the matching key ---
    gs_df['_join_id2'] = gs_df['gs_concept_id_2']
    pred_df['_join_id2'] = pred_df['predicted_concept_id_2']

    combined = gs_df.merge(
        pred_df,
        on=['concept_id_1', '_join_id2', 'attribute_category'],
        how='outer',
        suffixes=('_gs', '_pred'),
    )

    # Reconcile concept_name_1 from both sides
    if 'concept_name_1_gs' in combined.columns:
        combined['concept_name_1'] = combined['concept_name_1_gs'].fillna(combined['concept_name_1_pred'])
        combined.drop(columns=['concept_name_1_gs', 'concept_name_1_pred'], inplace=True)

    combined.drop(columns=['_join_id2'], inplace=True)

    # --- flags ---
    has_gs = combined['gs_concept_id_2'].notna()
    has_pred = combined['predicted_concept_id_2'].notna()
    combined['matched'] = has_gs & has_pred
    combined['status'] = 'match'
    combined.loc[has_gs & ~has_pred, 'status'] = 'missed'
    combined.loc[~has_gs & has_pred, 'status'] = 'extra'

    # --- order columns nicely ---
    leading = ['concept_id_1', 'concept_name_1', 'attribute_category',
               'gs_concept_id_2', 'gs_concept_code_2', 'gs_concept_name_2',
               'predicted_concept_id_2', 'predicted_concept_code_2', 'predicted_concept_name_2',
               'matched', 'status']
    extra_cols = [c for c in combined.columns if c not in leading]
    combined = combined[[c for c in leading if c in combined.columns] + extra_cols]

    # Sort for readability
    combined = combined.sort_values(['concept_id_1', 'attribute_category', 'status']).reset_index(drop=True)

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
    out_path = os.path.join(output_dir, "attribute_evaluation.csv")
    combined.to_csv(out_path, index=False)
    logger.info("Combined evaluation saved: %s (%d rows)", out_path, len(combined))
    return combined
