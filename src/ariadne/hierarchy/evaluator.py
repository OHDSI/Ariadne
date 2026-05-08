"""Batch processing and evaluation for the SNOMED CT attribute extraction pipeline.

Public API:
    process_gold_standard — run the pipeline over a gold-standard CSV.
    evaluate_results      — full outer join evaluation producing P/R/F1.
"""

import logging
import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import psycopg

from ariadne.hierarchy.pipeline import AttributeIndex, ContentFilterError, ReferenceIndex, find_attributes_two_stage
from ariadne.hierarchy.searchers import ATTR_KEY_TO_GS_CATEGORY
from ariadne.hierarchy.types import split_interprets_pairs
from ariadne.utils.config import load_hierarchy_settings
from ariadne.utils.settings import HierarchySettings

logger = logging.getLogger(__name__)


def _has_db_connection(index) -> bool:
    """Return True if the index holds a live psycopg connection (not thread-safe)."""
    return hasattr(index, "connection")


def _process_term(
    concept_id: int,
    concept_name: str,
    attribute_index: AttributeIndex,
    reference_index: ReferenceIndex | None,
    cfg: HierarchySettings,
) -> dict:
    """Process a single term — used by both sequential and parallel paths."""
    try:
        result = find_attributes_two_stage(
            concept_name, attribute_index, reference_index=reference_index, cfg=cfg
        )
        result["source_concept_id"] = concept_id
        result["source_concept_name"] = concept_name
        return result
    except (ValueError, KeyError, ContentFilterError, psycopg.Error) as e:
        logger.exception("  Error processing term %s: %s", concept_name, e)
        return {
            "source_concept_id": concept_id,
            "source_concept_name": concept_name,
            "error": str(e),
        }


def process_gold_standard(
    gs_path: str,
    attribute_index: AttributeIndex,
    reference_index: ReferenceIndex | None = None,
    cfg: HierarchySettings | None = None,
    checkpoint_every: int = 5,
    max_workers: int = 1,
) -> list[dict]:
    """Run the pipeline over every unique term in a gold-standard CSV.

    Supports checkpointing and optional parallel execution.

    When *max_workers* > 1, each worker thread creates its own database
    connections (psycopg is not thread-safe).

    Args:
        gs_path: Path to the gold-standard CSV (must have ``concept_id_1``,
            ``concept_name_1`` columns).
        attribute_index: Attribute searcher.
        reference_index: Reference searcher (or None).
        cfg: Pipeline configuration.
        checkpoint_every: Save a checkpoint every N terms (default 5).
        max_workers: Number of parallel worker threads (default 1 = sequential).

    Returns:
        List of result dicts (one per term).
    """
    cfg_local: HierarchySettings = cfg if cfg is not None else load_hierarchy_settings()
    checkpoint_file = Path(cfg_local.evaluation.output_dir) / "hierarchy_checkpoint.pkl"

    gs_df = pd.read_csv(gs_path)
    unique_terms = gs_df[["concept_id_1", "concept_name_1"]].drop_duplicates()
    logger.info("Processing %d terms from %s", len(unique_terms), gs_path)
    logger.info(
        "Models: extraction=%s, selection=%s | workers=%d",
        cfg_local.extraction, cfg_local.selection, max_workers,
    )

    # --- resume from checkpoint if available ---
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "rb") as f:
                checkpoint = pickle.load(f)
            all_results: list[dict] = checkpoint["results"]
            processed_ids: set = checkpoint["processed_ids"]
            logger.info("Resuming from checkpoint: %d terms already done", len(all_results))
        except (pickle.UnpicklingError, EOFError, KeyError) as exc:
            logger.warning("Corrupted checkpoint %s — starting fresh: %s", checkpoint_file, exc)
            checkpoint_file.unlink(missing_ok=True)
            all_results = []
            processed_ids = set()
    else:
        all_results = []
        processed_ids = set()

    pending = [
        row for row in unique_terms.itertuples(index=False)
        if row.concept_id_1 not in processed_ids
    ]

    if not pending:
        logger.info("All terms already processed.")
        return all_results

    if max_workers <= 1:
        # ── Sequential path ────────────────────────────────────────────────
        total_cost = sum(r.get("cost", {}).get("total_cost", 0.0) for r in all_results)
        for row in pending:
            logger.info(
                "\n%s\n[%d/%d] %s",
                "=" * 60, len(all_results) + 1, len(unique_terms), row.concept_name_1,
            )
            result = _process_term(
                row.concept_id_1, row.concept_name_1,
                attribute_index, reference_index, cfg_local,
            )
            all_results.append(result)
            processed_ids.add(row.concept_id_1)
            if "cost" in result:
                total_cost += result["cost"]["total_cost"]
                logger.info("  cost: $%.4f | running total: $%.4f",
                            result["cost"]["total_cost"], total_cost)

            if len(all_results) % checkpoint_every == 0:
                _save_checkpoint(checkpoint_file, all_results, processed_ids, cfg_local)

    else:
        # ── Parallel path ──────────────────────────────────────────────────
        # psycopg connections are not thread-safe — each worker creates its
        # own searcher instances. In-memory (legacy) indexes are read-only
        # and can be shared directly.
        attr_needs_conn = _has_db_connection(attribute_index)
        ref_needs_conn = reference_index is not None and _has_db_connection(reference_index)
        attr_cls = type(attribute_index)
        ref_cls = type(reference_index) if reference_index is not None else None

        lock = threading.Lock()
        done_count = [len(all_results)]  # mutable counter shared across threads

        def _worker(concept_id: int, concept_name: str) -> dict:
            # Create thread-local DB connections if needed
            local_attr = attr_cls(cfg=cfg_local) if attr_needs_conn else attribute_index
            local_ref = None
            if reference_index is not None:
                local_ref = ref_cls(cfg=cfg_local) if ref_needs_conn else reference_index
            try:
                return _process_term(concept_id, concept_name, local_attr, local_ref, cfg_local)
            finally:
                if attr_needs_conn:
                    local_attr.close()
                if ref_needs_conn and local_ref is not None:
                    local_ref.close()

        total_cost = sum(r.get("cost", {}).get("total_cost", 0.0) for r in all_results)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_worker, row.concept_id_1, row.concept_name_1): row
                for row in pending
            }
            for future in as_completed(futures):
                row = futures[future]
                result = future.result()
                with lock:
                    all_results.append(result)
                    processed_ids.add(row.concept_id_1)
                    done_count[0] += 1
                    n_done = done_count[0]
                    if "cost" in result:
                        total_cost += result["cost"]["total_cost"]
                    logger.info(
                        "[%d/%d] %s — cost: $%.4f | total: $%.4f",
                        n_done, len(unique_terms), row.concept_name_1,
                        result.get("cost", {}).get("total_cost", 0.0),
                        total_cost,
                    )
                    if n_done % checkpoint_every == 0:
                        _save_checkpoint(checkpoint_file, all_results, processed_ids, cfg_local)

    logger.info(
        "\n%s\nCompleted: %d terms, Total cost: $%.4f",
        "=" * 60, len(all_results), total_cost,
    )

    # --- clean up checkpoint on success ---
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        logger.info("Checkpoint file cleaned up")

    return all_results


def _save_checkpoint(
    checkpoint_file: Path,
    results: list[dict],
    processed_ids: set,
    cfg: HierarchySettings,
) -> None:
    os.makedirs(cfg.evaluation.output_dir, exist_ok=True)
    with open(checkpoint_file, "wb") as f:
        pickle.dump({"results": results, "processed_ids": processed_ids}, f)
    logger.info("Checkpoint saved (%d terms)", len(results))


def _build_prediction_rows(results: list[dict]) -> list[dict]:
    """Extract prediction rows from pipeline results for evaluation.

    Handles regular attributes (single dict, list of dicts) and paired
    ``interprets_interpretation`` structures via :func:`split_interprets_pairs`.

    Args:
        results: Pipeline result dicts from ``process_gold_standard``.

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
        results: List of pipeline result dicts from ``process_gold_standard``.
        gs_path: Path to the gold-standard CSV.
        cfg: Pipeline configuration (reads ``cfg.evaluation.output_dir``).

    Returns:
        Combined evaluation DataFrame.
    """
    cfg_local: HierarchySettings = cfg if cfg is not None else load_hierarchy_settings()
    output_dir = cfg_local.evaluation.output_dir
    # --- build predicted rows ---
    pred_rows = _build_prediction_rows(results)
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
