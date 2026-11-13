from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from ariadne.evaluation.concept_selection_evaluator import (
    BROAD_MATCH,
    EXACT_MATCH,
    PREDICATE,
    PREDICATE_B,
    SOURCE_ID,
    TARGET_CONCEPT_ID,
    TARGET_CONCEPT_ID_B,
)
from ariadne.utils.utils import resolve_path


def sweep_decision_thresholds(
    scores: pd.DataFrame,
    gold_standard_file: str | Path = "data/gold_standards/exact_matching_gs.csv",
    match_thresholds: Sequence[float] = (0.7, 0.8, 0.9, 0.95, 0.99),
    prune_thresholds: Sequence[float] = (0.01, 0.05, 0.1, 0.2),
    source_ids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Evaluates decision-model thresholds against the gold standard, using already-computed probabilities
    (no API calls).

    Args:
        scores: Output of :meth:`DecisionModelMapper.score_terms` (``source_id``, ``concept_id`` with -1 for
            no match, ``probability``).
        gold_standard_file: Gold standard CSV (as used by :func:`evaluate`).
        match_thresholds: Values for ``match_threshold`` to evaluate.
        prune_thresholds: Values for ``prune_threshold`` to evaluate.
        source_ids: Optional subset of source IDs to evaluate.

    Returns:
        One row per threshold combination with:

        - ``n_terms``: terms with decision model scores and a gold standard entry.
        - ``fraction_resolved``: fraction mapped without the LLM (confident match or all candidates pruned).
        - ``resolved_accuracy``: accuracy of those resolved mappings.
        - ``n_resolved_correct`` / ``n_resolved_incorrect``.
        - ``mean_llm_candidates``: average number of candidates left for the LLM.
        - ``gold_pruned_rate``: among unresolved terms whose exact-match gold concept is a candidate, the fraction
          where it was pruned (the LLM can then no longer find it).
    """
    gold_standard = pd.read_csv(resolve_path(gold_standard_file), dtype={SOURCE_ID: str})
    gold_by_id = gold_standard.drop_duplicates(subset=[SOURCE_ID]).set_index(SOURCE_ID)

    terms = []
    for source_id, group in scores.groupby("source_id", sort=False):
        source_id = str(source_id)
        if source_ids is not None and source_id not in source_ids:
            continue
        if source_id not in gold_by_id.index:
            continue
        gold = gold_by_id.loc[source_id]
        exact_targets = {
            int(gold[col])
            for col, pred in ((TARGET_CONCEPT_ID, PREDICATE), (TARGET_CONCEPT_ID_B, PREDICATE_B))
            if gold.get(pred) == EXACT_MATCH and pd.notna(gold.get(col))
        }
        no_match_correct = gold.get(PREDICATE) == BROAD_MATCH or gold.get(PREDICATE_B) == BROAD_MATCH
        probabilities = dict(zip(group["concept_id"].astype(int), group["probability"].astype(float)))
        terms.append((probabilities, exact_targets, no_match_correct))

    rows = []
    for match_threshold in match_thresholds:
        for prune_threshold in prune_thresholds:
            n_correct = n_incorrect = n_unresolved = n_gold_candidate = n_gold_pruned = 0
            llm_candidates = 0
            for probabilities, exact_targets, no_match_correct in terms:
                best_id = max(probabilities, key=probabilities.get)
                if probabilities[best_id] > match_threshold:
                    mapped_id = best_id
                else:
                    kept = {cid for cid, p in probabilities.items() if cid != -1 and p >= prune_threshold}
                    if kept:
                        n_unresolved += 1
                        llm_candidates += len(kept)
                        gold_candidates = exact_targets & (probabilities.keys() - {-1})
                        if gold_candidates:
                            n_gold_candidate += 1
                            if not gold_candidates & kept:
                                n_gold_pruned += 1
                        continue
                    mapped_id = -1
                is_correct = mapped_id in exact_targets or (mapped_id == -1 and no_match_correct)
                if is_correct:
                    n_correct += 1
                else:
                    n_incorrect += 1
            n_terms = len(terms)
            n_resolved = n_correct + n_incorrect
            rows.append(
                {
                    "match_threshold": match_threshold,
                    "prune_threshold": prune_threshold,
                    "n_terms": n_terms,
                    "fraction_resolved": n_resolved / n_terms if n_terms else float("nan"),
                    "resolved_accuracy": n_correct / n_resolved if n_resolved else float("nan"),
                    "n_resolved_correct": n_correct,
                    "n_resolved_incorrect": n_incorrect,
                    "mean_llm_candidates": llm_candidates / n_unresolved if n_unresolved else float("nan"),
                    "gold_pruned_rate": n_gold_pruned / n_gold_candidate if n_gold_candidate else float("nan"),
                }
            )
    return pd.DataFrame(rows)
