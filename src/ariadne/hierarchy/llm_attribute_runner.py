"""Batch execution helpers for hierarchy attribute extraction.

The module exposes a single public entry point, :func:`process_hierarchy`,
which executes the term-level extraction pipeline over a tabular dataset.
Execution is resumable: each term's result is written to its own JSON file as
soon as it succeeds, so an interrupted run resumes by reprocessing only the
terms without a cached file. Failed terms are left uncached and retried on the
next run.

Terms are processed concurrently by ``max_workers`` threads (the work is
dominated by waiting on LLM API calls). Workers only run the extraction;
caching, cost tracking and logging happen in the calling thread.
"""

import json
import logging
import re
from pathlib import Path

import openai
import pandas as pd
import psycopg

from ariadne.hierarchy.llm_attribute_extractor import (
    AttributeSearcher,
    ContentFilterError,
    ReferenceSearcher,
    assign_role_groups,
    drop_non_candidate_values,
    extract_attributes,
)
from ariadne.utils.concurrency import run_concurrently
from ariadne.utils.settings import HierarchySettings

logger = logging.getLogger(__name__)


def _make_source_key(vocabulary_id: str, concept_code: str) -> str:
    """Composite primary key for a source term (works without an OMOP concept_id)."""
    return f"{vocabulary_id}:{concept_code}"


def _process_term(
    source_meta: dict,
    source_term: str,
    attribute_searcher: AttributeSearcher,
    reference_searcher: ReferenceSearcher,
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
        Search component used for reference context retrieval.
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
    except (
        ValueError,
        KeyError,
        ContentFilterError,
        psycopg.Error,
        # Transient API failures (raised after the OpenAI client's own retries);
        # more likely with concurrent workers, so fail the term, not the run.
        openai.RateLimitError,
        openai.APIConnectionError,
    ) as exc:
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
    max_workers: int | None = None,
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
    max_workers
        Number of terms processed concurrently. Defaults to
        ``hierarchy_settings.evaluation.max_workers``; ``1`` runs sequentially.

    Returns
    -------
    list[dict]
        One result dictionary per processed source code: previously cached
        results first, then newly processed terms in input order.

    Raises
    ------
    ValueError
        If ``hierarchy_settings`` is missing.
    """
    if not hierarchy_settings:
        raise ValueError("HierarchySettings must be provided")

    results_dir = Path(hierarchy_settings.evaluation.llm_response_folder)

    all_results = load_hierarchy_results(results_dir)
    processed_ids = {r["source_key"] for r in all_results}
    if all_results:
        logger.info(
            "Resuming: %d terms already cached in %s", len(all_results), results_dir
        )
        _drop_cached_non_candidate_values(all_results, hierarchy_settings)

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
    if max_workers is None:
        max_workers = getattr(hierarchy_settings.evaluation, "max_workers", 1)

    def _source_meta(row) -> dict:
        source_concept_code = str(getattr(row, source_code_column))
        source_vocabulary_id = str(getattr(row, source_vocab_column))
        raw_id = getattr(row, source_id_column) if has_source_id else None
        return {
            "source_key": _make_source_key(source_vocabulary_id, source_concept_code),
            "source_concept_id": None if raw_id is None or pd.isna(raw_id) else raw_id,
            "source_concept_code": source_concept_code,
            "source_vocabulary_id": source_vocabulary_id,
            "source_term": str(getattr(row, source_term_column)),
        }

    def _worker(source_meta: dict) -> dict:
        logger.debug("\n%s\n%s", "=" * 60, source_meta["source_term"])
        return _process_term(
            source_meta,
            source_meta["source_term"],
            attribute_searcher,
            reference_searcher,
            hierarchy_settings,
        )

    pending_meta = [_source_meta(row) for row in pending_terms]
    new_results: list[dict | None] = [None] * len(pending_meta)
    logger.info(
        "Processing %d pending terms with %d worker(s)", len(pending_meta), max(1, max_workers)
    )

    for idx, result in run_concurrently(pending_meta, _worker, max_workers):
        new_results[idx] = result
        source_key = result["source_key"]

        # Do not cache failures: leaving them unpersisted means they are retried
        # on the next run (mirrors the LlmMapper per-response file cache).
        if "error" in result:
            logger.warning(
                "Term %s failed; not caching so it retries next run.", source_key
            )
            continue

        _save_result(results_dir, result)
        processed_ids.add(source_key)
        if "cost" in result:
            total_cost += result["cost"]["total_cost"]
            logger.debug(
                "  %s cost: $%.4f | running total: $%.4f",
                source_key,
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

    all_results.extend(r for r in new_results if r is not None)

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


def load_hierarchy_results(path: str | Path) -> list[dict]:
    """Load pipeline results from a JSON list file or a folder of per-term JSON files.

    Unreadable per-term files and results without a ``source_key`` are skipped with a warning.
    """
    path = Path(path)
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        loaded = data if isinstance(data, list) else [data]
    else:
        loaded = []
        for file in sorted(path.glob("*.json")):
            try:
                with open(file, encoding="utf-8") as f:
                    loaded.append(json.load(f))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping unreadable result file %s: %s", file, exc)
    results = [r for r in loaded if isinstance(r, dict) and r.get("source_key")]
    if len(results) < len(loaded):
        logger.warning("Skipped %d result(s) without a source_key in %s", len(loaded) - len(results), path)
    return results


def _drop_cached_non_candidate_values(results: list[dict], hierarchy_settings) -> None:
    """Apply the candidates-only rule to cached results (in place), including ones cached before it existed.

    The cache files are not rewritten; the rule is re-applied on every load.
    """
    affected = []
    for result in results:
        if result.get("error") or "retrieved_candidates" not in result:
            continue
        dropped = drop_non_candidate_values(result, result["retrieved_candidates"])
        if dropped:
            assign_role_groups(result, hierarchy_settings)
            affected.append(
                f"{result.get('source_term')}: "
                + ", ".join(f"{d['attribute']}={d['concept_name']}" for d in dropped)
            )
    if affected:
        logger.info(
            "Dropped selected values not among the candidates in %d cached result(s): %s",
            len(affected), "; ".join(affected),
        )


def _save_result(results_dir: Path, result: dict) -> None:
    """Persist a single term result atomically to its own JSON file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / _safe_result_filename(result["source_key"])
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    tmp_path.replace(path)
