"""Batch execution helpers for hierarchy attribute extraction.

The module exposes a single public entry point, :func:`process_hierarchy`,
which executes the term-level extraction pipeline over a tabular dataset.
Execution is intentionally single-threaded and resumable: each term's result
is written to its own JSON file as soon as it succeeds, so an interrupted run
resumes by reprocessing only the terms without a cached file. Failed terms are
left uncached and retried on the next run.
"""

import json
import logging
import re
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


def _make_source_key(vocabulary_id: str, concept_code: str) -> str:
    """Composite primary key for a source term (works without an OMOP concept_id)."""
    return f"{vocabulary_id}:{concept_code}"


def _process_term(
    source_meta: dict,
    source_term: str,
    attribute_searcher: AttributeSearcher,
    reference_searcher: ReferenceSearcher | None,
    hierarchy_settings: HierarchySettings,
) -> dict:
    """Run extraction for one source term and normalize failure handling.

    Parameters
    ----------
    source_meta
        Source identity fields (``source_key``, ``source_concept_id``,
        ``source_concept_code``, ``source_vocabulary_id``, ``source_term``)
        attached to the result verbatim.
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
        result.update(source_meta)
        return result
    except (ValueError, KeyError, ContentFilterError, psycopg.Error) as exc:
        logger.exception("  Error processing term %s: %s", source_term, exc)
        return {**source_meta, "error": str(exc)}


def process_hierarchy(
    terms: pd.DataFrame,
    attribute_searcher: AttributeSearcher,
    reference_searcher: ReferenceSearcher,
    hierarchy_settings: HierarchySettings | None,
    source_term_column: str = "source_term",
    source_code_column: str = "source_concept_code",
    source_vocab_column: str = "source_vocabulary_id",
    source_id_column: str | None = "source_concept_id",
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
        Required runtime settings. ``evaluation.llm_response_folder`` is used to
        store one result file per source term for resume.
    source_term_column
        Name of the column containing source term text.
    source_code_column
        Name of the column containing the source concept code.
    source_vocab_column
        Name of the column containing the source vocabulary id. Combined with
        ``source_code_column`` to form the composite primary key.
    source_id_column
        Optional column with the OMOP ``concept_id`` (may be absent for new
        vocabularies). Carried through when present.
    checkpoint_every
        Cadence (in terms) for emitting a progress log line. Results are
        persisted immediately per term regardless of this value.

    Returns
    -------
    list[dict]
        One result dictionary per processed source code, including any
        previously cached results loaded from disk.

    Raises
    ------
    ValueError
        If ``hierarchy_settings`` is missing.
    """
    if not hierarchy_settings:
        raise ValueError("HierarchySettings must be provided")

    results_dir = Path(hierarchy_settings.evaluation.llm_response_folder)

    all_results, processed_ids = _load_cached_results(results_dir)
    if all_results:
        logger.info(
            "Resuming: %d terms already cached in %s", len(all_results), results_dir
        )

    has_source_id = bool(source_id_column) and source_id_column in terms.columns

    def _row_source_key(row) -> str:
        return _make_source_key(
            getattr(row, source_vocab_column), getattr(row, source_code_column)
        )

    pending_terms = [
        row for row in terms.itertuples(index=False) if _row_source_key(row) not in processed_ids
    ]

    if not pending_terms:
        logger.info("All terms already processed.")
        return all_results

    total_cost = sum(r.get("cost", {}).get("total_cost", 0.0) for r in all_results)

    for row in pending_terms:
        source_concept_code = str(getattr(row, source_code_column))
        source_vocabulary_id = str(getattr(row, source_vocab_column))
        source_term = str(getattr(row, source_term_column))
        source_key = _make_source_key(source_vocabulary_id, source_concept_code)
        raw_id = getattr(row, source_id_column) if has_source_id else None
        source_concept_id = None if raw_id is None or pd.isna(raw_id) else raw_id
        source_meta = {
            "source_key": source_key,
            "source_concept_id": source_concept_id,
            "source_concept_code": source_concept_code,
            "source_vocabulary_id": source_vocabulary_id,
            "source_term": source_term,
        }
        logger.debug(
            "\n%s\n[%d/%d] %s",
            "=" * 60,
            len(all_results) + 1,
            len(terms),
            source_term,
        )
        result = _process_term(
            source_meta,
            source_term,
            attribute_searcher,
            reference_searcher,
            hierarchy_settings,
        )

        # Do not cache failures: leaving them unpersisted means they are retried
        # on the next run (mirrors the LlmMapper per-response file cache).
        if "error" in result:
            logger.warning(
                "Term %s failed; not caching so it retries next run.", source_key
            )
            all_results.append(result)
            continue

        _save_result(results_dir, result)
        all_results.append(result)
        processed_ids.add(source_key)
        if "cost" in result:
            total_cost += result["cost"]["total_cost"]
            logger.debug(
                "  cost: $%.4f | running total: $%.4f",
                result["cost"]["total_cost"],
                total_cost,
            )

        if checkpoint_every > 0 and len(processed_ids) % checkpoint_every == 0:
            logger.info(
                "Processed %d/%d codes | Total API cost: $%.4f",
                len(processed_ids),
                len(terms),
                total_cost,
            )

    logger.debug(
        "\n%s\nCompleted: %d terms, Total cost: $%.4f",
        "=" * 60,
        len(all_results),
        total_cost,
    )

    return all_results


def _safe_result_filename(source_key: str) -> str:
    """Filesystem-safe filename for a source term's cached result."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", source_key) + ".json"


def _load_cached_results(results_dir: Path) -> tuple[list[dict], set]:
    """Load previously persisted per-term results for resume."""
    all_results: list[dict] = []
    processed_ids: set = set()
    if not results_dir.exists():
        return all_results, processed_ids
    for path in sorted(results_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                result = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping unreadable result file %s: %s", path, exc)
            continue
        source_key = result.get("source_key")
        if source_key is None:
            logger.warning("Result file %s has no source_key; skipping.", path)
            continue
        all_results.append(result)
        processed_ids.add(source_key)
    return all_results, processed_ids


def _save_result(results_dir: Path, result: dict) -> None:
    """Persist a single term result atomically to its own JSON file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / _safe_result_filename(result["source_key"])
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    tmp_path.replace(path)
