"""Batch execution helpers for the hierarchy extraction pipeline.

Public API:
    process_hierarchy — run the single-term pipeline over a gold-standard CSV.
"""

import logging
import os
import pickle
from pathlib import Path

import psycopg

from ariadne.hierarchy.pipeline import (
    AttributeSearcher,
    ContentFilterError,
    ReferenceSearcher,
    find_attributes_two_stage,
)
from ariadne.utils.settings import HierarchySettings

logger = logging.getLogger(__name__)


def _process_term(
    concept_id: int,
    concept_name: str,
    attribute_searcher: AttributeSearcher,
    reference_searcher: ReferenceSearcher | None,
    hierarchy_settings: HierarchySettings,
) -> dict:
    """Process a single term using ``find_attributes_two_stage``."""
    try:
        result = find_attributes_two_stage(
            concept_name,
            attribute_searcher,
            reference_searcher=reference_searcher,
            hierarchy_settings=hierarchy_settings,
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
    terms: str,
    attribute_searcher: AttributeSearcher,
    reference_searcher: ReferenceSearcher | None = None,
    hierarchy_settings: HierarchySettings | None = None,
    checkpoint_every: int = 5,
) -> list[dict]:
    """Run hierarchy extraction over every unique term in a gold-standard CSV.

    Supports checkpointing during single-threaded execution.
    """
    checkpoint_file = Path(hierarchy_settings.evaluation.output_dir) / "hierarchy_checkpoint.pkl"

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
        row for row in terms.itertuples(index=False) if row.concept_id not in processed_ids
    ]

    if not pending_terms:
        logger.debug("All terms already processed.")
        return all_results

    total_cost = sum(r.get("cost", {}).get("total_cost", 0.0) for r in all_results)

    try:
        for row in pending_terms:
            concept_id = int(row.concept_id)
            concept_name = str(row.concept_name)
            logger.debug(
                "\n%s\n[%d/%d] %s",
                "=" * 60,
                len(all_results) + 1,
                len(terms),
                concept_name,
            )
            result = _process_term(
                concept_id,
                concept_name,
                attribute_searcher,
                reference_searcher,
                hierarchy_settings,
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

            if checkpoint_every > 0 and len(all_results) % checkpoint_every == 0:
                _save_checkpoint(checkpoint_file, all_results, processed_ids, hierarchy_settings)
                logger.info(
                    "Processed %d/%d codes | Total API cost: $%.4f",
                    len(all_results),
                    len(terms),
                    total_cost,
                )
    except KeyboardInterrupt:
        _save_checkpoint(checkpoint_file, all_results, processed_ids, hierarchy_settings)
        logger.warning(
            "Interrupted while running. Saved checkpoint with %d terms.",
            len(all_results),
        )
        raise

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
    hierarchy_settings: HierarchySettings,
) -> None:
    os.makedirs(hierarchy_settings.evaluation.output_dir, exist_ok=True)
    with open(checkpoint_file, "wb") as f:
        pickle.dump({"results": results, "processed_ids": processed_ids}, f)
    logger.debug("Checkpoint saved (%d terms)", len(results))


