"""Batch execution helpers for hierarchy attribute extraction.

The module exposes a single public entry point, :func:`process_hierarchy`,
which executes the term-level extraction pipeline over a tabular dataset.
Execution is intentionally single-threaded and checkpoint-aware so long runs
can resume after interruptions.
"""

import logging
import os
import pickle
from pathlib import Path

import pandas as pd
import psycopg

from ariadne.hierarchy.llm_attribute_extractor import (
    AttributeSearcher,
    ContentFilterError,
    ReferenceSearcher,
    extract_attributes,
)
from ariadne.utils.settings import HierarchySettings

logger = logging.getLogger(__name__)


def _process_term(
    source_code: str,
    source_term: str,
    attribute_searcher: AttributeSearcher,
    reference_searcher: ReferenceSearcher | None,
    hierarchy_settings: HierarchySettings,
) -> dict:
    """Run extraction for one source term and normalize failure handling.

    Parameters
    ----------
    source_code
        Stable source identifier for the term being processed.
    source_term
        Source concept name/text to send to the attribute extractor.
    attribute_searcher
        Search component used to retrieve candidate attributes.
    reference_searcher
        Optional search component used for reference context retrieval.
    hierarchy_settings
        Runtime settings controlling model/provider/evaluation behavior.

    Returns
    -------
    dict
        Extraction result enriched with source metadata. On known recoverable
        errors, returns an error payload instead of raising.
    """
    try:
        result = extract_attributes(
            source_term,
            attribute_searcher,
            reference_searcher=reference_searcher,
            hierarchy_settings=hierarchy_settings,
        )
        result["source_code"] = source_code
        result["source_concept_name"] = source_term
        return result
    except (ValueError, KeyError, ContentFilterError, psycopg.Error) as exc:
        logger.exception("  Error processing term %s: %s", source_term, exc)
        return {
            "source_code": source_code,
            "source_term": source_term,
            "error": str(exc),
        }


def process_hierarchy(
    terms: pd.DataFrame,
    attribute_searcher: AttributeSearcher,
    reference_searcher: ReferenceSearcher,
    hierarchy_settings: HierarchySettings | None,
    source_code_column: str = "source_code",
    source_term_column: str = "source_term",
    checkpoint_every: int = 5,
) -> list[dict]:
    """Execute hierarchy extraction over source terms with checkpoint resume.

    Parameters
    ----------
    terms
        Input rows containing source identifiers and term text.
    attribute_searcher
        Searcher used by the extraction pipeline for attribute candidates.
    reference_searcher
        Searcher used for optional reference concept lookups.
    hierarchy_settings
        Required runtime settings. ``evaluation.output_dir`` is used for
        checkpoint persistence.
    source_code_column
        Name of the column containing unique source identifiers.
    source_term_column
        Name of the column containing source term text.
    checkpoint_every
        Save progress every N processed terms. Set to ``0`` or a negative
        value to disable periodic checkpoint writes.

    Returns
    -------
    list[dict]
        One result dictionary per processed source code, including any
        previously restored checkpoint results.

    Raises
    ------
    ValueError
        If ``hierarchy_settings`` is missing.
    KeyboardInterrupt
        Re-raised after persisting an interruption checkpoint.
    """
    if not hierarchy_settings:
        raise ValueError("HierarchySettings must be provided")

    checkpoint_file = Path(hierarchy_settings.evaluation.output_dir) / "hierarchy_checkpoint.pkl"

    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "rb") as f:
                checkpoint = pickle.load(f)
            all_results: list[dict] = checkpoint["results"]
            processed_ids: set = checkpoint["processed_ids"]
            logger.info("Resuming from checkpoint: %d terms already done", len(all_results))
        except (pickle.UnpicklingError, EOFError, KeyError) as exc:
            logger.warning("Corrupted checkpoint %s - starting fresh: %s", checkpoint_file, exc)
            checkpoint_file.unlink(missing_ok=True)
            all_results = []
            processed_ids = set()
    else:
        all_results = []
        processed_ids = set()

    pending_terms = [
        row for row in terms.itertuples(index=False) if getattr(row, source_code_column) not in processed_ids
    ]

    if not pending_terms:
        logger.info("All terms already processed.")
        return all_results

    total_cost = sum(r.get("cost", {}).get("total_cost", 0.0) for r in all_results)

    try:
        for row in pending_terms:
            source_code = str(getattr(row, source_code_column))
            source_term = str(getattr(row, source_term_column))
            logger.debug(
                "\n%s\n[%d/%d] %s",
                "=" * 60,
                len(all_results) + 1,
                len(terms),
                source_term,
            )
            result = _process_term(
                source_code,
                source_term,
                attribute_searcher,
                reference_searcher,
                hierarchy_settings,
            )
            all_results.append(result)
            processed_ids.add(source_code)
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
    """Persist current batch progress to a pickle checkpoint file."""
    os.makedirs(hierarchy_settings.evaluation.output_dir, exist_ok=True)
    with open(checkpoint_file, "wb") as f:
        pickle.dump({"results": results, "processed_ids": processed_ids}, f)
    logger.debug("Checkpoint saved (%d terms)", len(results))
