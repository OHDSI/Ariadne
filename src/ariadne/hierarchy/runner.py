"""Batch execution helpers for the hierarchy extraction pipeline.

Public API:
    process_hierarchy — run the single-term pipeline over a gold-standard CSV.
"""

import logging
import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import psycopg

from ariadne.hierarchy.pipeline import (
    AttributeSearcher,
    ContentFilterError,
    ReferenceSearcher,
    find_attributes_two_stage,
)
from ariadne.utils.config import load_hierarchy_settings
from ariadne.utils.settings import HierarchySettings

logger = logging.getLogger(__name__)


def _has_db_connection(searcher) -> bool:
    """Return True if the searcher holds a live psycopg connection (not thread-safe)."""
    return hasattr(searcher, "connection")


def _process_term(
    concept_id: int,
    concept_name: str,
    attribute_searcher: AttributeSearcher,
    reference_searcher: ReferenceSearcher | None,
    cfg: HierarchySettings,
) -> dict:
    """Process a single term using ``find_attributes_two_stage``."""
    try:
        result = find_attributes_two_stage(
            concept_name,
            attribute_searcher,
            reference_searcher=reference_searcher,
            cfg=cfg,
        )
        result["source_concept_id"] = concept_id
        result["source_concept_name"] = concept_name
        return result
    except (ValueError, KeyError, ContentFilterError, psycopg.Error) as exc:
        logger.exception("  Error processing term %s: %s", concept_name, exc)
        return {
            "source_concept_id": concept_id,
            "source_concept_name": concept_name,
            "error": str(exc),
        }


def process_hierarchy(
    gs_path: str,
    attribute_searcher: AttributeSearcher,
    reference_searcher: ReferenceSearcher | None = None,
    cfg: HierarchySettings | None = None,
    checkpoint_every: int = 25,
    max_workers: int = 1,
) -> list[dict]:
    """Run hierarchy extraction over every unique term in a gold-standard CSV.

    Supports checkpointing and optional parallel execution.
    When *max_workers* > 1, each worker thread creates its own database
    connections (psycopg is not thread-safe).
    """
    cfg_local: HierarchySettings = cfg if cfg is not None else load_hierarchy_settings()
    checkpoint_file = Path(cfg_local.evaluation.output_dir) / "hierarchy_checkpoint.pkl"

    gs_df = pd.read_csv(gs_path)
    unique_terms = gs_df[["concept_id_1", "concept_name_1"]].drop_duplicates()
    logger.debug("Processing %d terms from %s", len(unique_terms), gs_path)
    logger.debug(
        "Models: extraction=%s, selection=%s | workers=%d",
        cfg_local.extraction,
        cfg_local.selection,
        max_workers,
    )

    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "rb") as f:
                checkpoint = pickle.load(f)
            all_results: list[dict] = checkpoint["results"]
            processed_ids: set = checkpoint["processed_ids"]
            logger.debug("Resuming from checkpoint: %d terms already done", len(all_results))
        except (pickle.UnpicklingError, EOFError, KeyError) as exc:
            logger.warning("Corrupted checkpoint %s - starting fresh: %s", checkpoint_file, exc)
            checkpoint_file.unlink(missing_ok=True)
            all_results = []
            processed_ids = set()
    else:
        all_results = []
        processed_ids = set()

    pending_terms = [
        row for row in unique_terms.itertuples(index=False) if row.concept_id_1 not in processed_ids
    ]

    if not pending_terms:
        logger.debug("All terms already processed.")
        return all_results

    total_cost = sum(r.get("cost", {}).get("total_cost", 0.0) for r in all_results)

    if max_workers <= 1:
        for row in pending_terms:
            concept_id = int(row.concept_id_1)
            concept_name = str(row.concept_name_1)
            logger.debug(
                "\n%s\n[%d/%d] %s",
                "=" * 60,
                len(all_results) + 1,
                len(unique_terms),
                concept_name,
            )
            result = _process_term(
                concept_id,
                concept_name,
                attribute_searcher,
                reference_searcher,
                cfg_local,
            )
            all_results.append(result)
            processed_ids.add(concept_id)
            if "cost" in result:
                total_cost += result["cost"]["total_cost"]
                logger.debug(
                    "  cost: $%.4f | running total: $%.4f",
                    result["cost"]["total_cost"],
                    total_cost,
                )

            if len(all_results) % checkpoint_every == 0:
                _save_checkpoint(checkpoint_file, all_results, processed_ids, cfg_local)
                logger.info(
                    "Processed %d/%d codes | Total API cost: $%.4f",
                    len(all_results),
                    len(unique_terms),
                    total_cost,
                )
    else:
        attr_needs_conn = _has_db_connection(attribute_searcher)
        ref_needs_conn = reference_searcher is not None and _has_db_connection(reference_searcher)
        attr_cls = type(attribute_searcher)
        ref_cls = type(reference_searcher) if reference_searcher is not None else None

        lock = threading.Lock()
        done_count = [len(all_results)]

        def _worker(concept_id: int, concept_name: str) -> dict:
            local_attribute_searcher = (
                attr_cls(cfg=cfg_local) if attr_needs_conn else attribute_searcher
            )
            local_reference_searcher = None
            if reference_searcher is not None:
                local_reference_searcher = (
                    ref_cls(cfg=cfg_local) if ref_needs_conn else reference_searcher
                )
            try:
                return _process_term(
                    concept_id,
                    concept_name,
                    local_attribute_searcher,
                    local_reference_searcher,
                    cfg_local,
                )
            finally:
                if attr_needs_conn:
                    local_attribute_searcher.close()
                if ref_needs_conn and local_reference_searcher is not None:
                    local_reference_searcher.close()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_worker, int(row.concept_id_1), str(row.concept_name_1)): row
                for row in pending_terms
            }
            for future in as_completed(futures):
                row = futures[future]
                result = future.result()
                concept_id = int(row.concept_id_1)
                concept_name = str(row.concept_name_1)
                with lock:
                    all_results.append(result)
                    processed_ids.add(concept_id)
                    done_count[0] += 1
                    n_done = done_count[0]
                    if "cost" in result:
                        total_cost += result["cost"]["total_cost"]
                    logger.debug(
                        "[%d/%d] %s - cost: $%.4f | total: $%.4f",
                        n_done,
                        len(unique_terms),
                        concept_name,
                        result.get("cost", {}).get("total_cost", 0.0),
                        total_cost,
                    )
                    if n_done % checkpoint_every == 0:
                        _save_checkpoint(checkpoint_file, all_results, processed_ids, cfg_local)
                        logger.info(
                            "Processed %d/%d codes | Total API cost: $%.4f",
                            n_done,
                            len(unique_terms),
                            total_cost,
                        )

    logger.debug(
        "\n%s\nCompleted: %d terms, Total cost: $%.4f",
        "=" * 60,
        len(all_results),
        total_cost,
    )

    if checkpoint_file.exists():
        checkpoint_file.unlink()
        logger.debug("Checkpoint file cleaned up")

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
    logger.debug("Checkpoint saved (%d terms)", len(results))


