from typing import Optional, List

import pandas as pd

from ariadne.evaluation.concept_search_evaluator import _load_gold_standard
from ariadne.utils.utils import resolve_path

SOURCE_CONCEPT_ID = "source_concept_id"
SOURCE_TERM = "source_term"
TARGET_CONCEPT_ID = "target_concept_id"
TARGET_CONCEPT_NAME = "target_concept_name"
TARGET_CONCEPT_ID_B = "target_concept_id_b"
TARGET_CONCEPT_NAME_B = "target_concept_name_b"
PREDICATE = "predicate"
PREDICATE_B = "predicate_b"
EXACT_MATCH = "exactMatch"
BROAD_MATCH = "broadMatch"


def evaluate(
    selection_results: pd.DataFrame,
    gold_standard_file: str = "data/gold_standards/exact_matching_gs.csv",
    source_id_column: str = "source_concept_id",
    term_column: str = "cleaned_term",
    mapped_concept_id_column: str = "mapped_concept_id",
    mapped_concept_name_column: str = "mapped_concept_name",
    mapped_method_column: Optional[str] = "map_method",
    source_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Evaluate the concept selection results against the gold standard.

    Args:
        selection_results: Pandas DataFrame containing the results of selection results.
        gold_standard_file: Path to the CSV file containing the gold standard mappings.
        source_id_column: Name of the column with source concept IDs.
        term_column: Name of the column with source terms.
        mapped_concept_id_column: Name of the column with mapped concept IDs.
        mapped_concept_name_column: Name of the column with mapped concept names.
        mapped_method_column: Optional: Name of the column with mapping methods, e.g. "verbatim" or "llm".
        source_ids: Optional list of source concept IDs to evaluate. If None, evaluate all.

    Returns:
        A Pandas DataFrame with the evaluation results.
    """
    gold_standard = pd.read_csv(resolve_path(gold_standard_file))

    if mapped_method_column:
        output_mapped_method_column = mapped_method_column
    else:
        output_mapped_method_column = "map_method"

    selection_results.reset_index(drop=True, inplace=True)
    evaluation_results = []
    for index, row in selection_results.iterrows():
        source_id = int(row[source_id_column])
        if source_ids is not None and source_id not in source_ids:
            continue

        gold_entry = gold_standard[gold_standard[SOURCE_CONCEPT_ID] == source_id]
        if gold_entry.empty:
            continue
        gold_entry = gold_entry.iloc[0]
        gold_target_concept_id = gold_entry[TARGET_CONCEPT_ID]
        gold_target_concept_id_b = gold_entry[TARGET_CONCEPT_ID_B]
        gold_predicate = gold_entry[PREDICATE]
        gold_predicate_b = gold_entry[PREDICATE_B]

        mapped_concept_id = int(row[mapped_concept_id_column])
        if mapped_method_column and mapped_method_column in row:
            map_method = row[mapped_method_column]
        else:
            map_method = "unknown"
        is_correct = (
            (mapped_concept_id == gold_target_concept_id and gold_predicate == EXACT_MATCH)
            or (mapped_concept_id == gold_target_concept_id_b and gold_predicate_b == EXACT_MATCH)
            or (mapped_concept_id == -1 and gold_predicate == BROAD_MATCH)
            or (mapped_concept_id == -1 and gold_predicate_b == BROAD_MATCH)
        )
        result_row = {
            SOURCE_CONCEPT_ID: source_id,
            SOURCE_TERM: gold_entry.get(SOURCE_TERM),
            output_mapped_method_column: map_method,
            TARGET_CONCEPT_ID: gold_target_concept_id,
            TARGET_CONCEPT_NAME: gold_entry.get(TARGET_CONCEPT_NAME),
            PREDICATE: gold_predicate,
            TARGET_CONCEPT_ID_B: gold_target_concept_id_b,
            TARGET_CONCEPT_NAME_B: gold_entry.get(TARGET_CONCEPT_NAME_B),
            PREDICATE_B: gold_predicate_b,
        }
        if term_column != SOURCE_TERM:
            result_row[term_column] = row[term_column]
        result_row.update(
            {
                mapped_concept_id_column: mapped_concept_id,
                mapped_concept_name_column: row[mapped_concept_name_column],
                "is_correct": is_correct,
            }
        )
        evaluation_results.append(result_row)
    evaluation_df = pd.DataFrame(evaluation_results)

    # Add overall accuracy as a column:
    accuracy = evaluation_df["is_correct"].mean()
    evaluation_df["overall_accuracy"] = accuracy

    return evaluation_df
