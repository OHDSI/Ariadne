# Copyright 2025 Observational Health Data Sciences and Informatics
#
# This file is part of Ariadne
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from ariadne.utils.utils import resolve_path

# Gold standard column names:
SOURCE_CONCEPT_ID = "source_concept_id"
SOURCE_TERM = "source_term"
TARGET_CONCEPT_ID = "target_concept_id"
TARGET_CONCEPT_NAME = "target_concept_name"
TARGET_CONCEPT_ID_B = "target_concept_id_b"
TARGET_CONCEPT_NAME_B = "target_concept_name_b"
PREDICATE = "predicate"
PREDICATE_B = "predicate_b"
BROAD_MATCH = "broadMatch"


def _load_gold_standard(filename: str | Path) -> Dict[int, Dict[str, Any]]:
    df = pd.read_csv(filename)
    gold_standard = {}
    for index, row in df.iterrows():
        details: Dict[str, Any] = {SOURCE_TERM: str(row[SOURCE_TERM])}
        if row[PREDICATE] == BROAD_MATCH:
            details[TARGET_CONCEPT_ID] = None
            details[TARGET_CONCEPT_NAME] = None
        else:
            details[TARGET_CONCEPT_ID] = int(row[TARGET_CONCEPT_ID])
            details[TARGET_CONCEPT_NAME] = str(row[TARGET_CONCEPT_NAME])
        if row[PREDICATE_B] == BROAD_MATCH or pd.isna(row[TARGET_CONCEPT_ID_B]):
            details[TARGET_CONCEPT_ID_B] = None
            details[TARGET_CONCEPT_NAME_B] = None
        else:
            details[TARGET_CONCEPT_ID_B] = int(row[TARGET_CONCEPT_ID_B])
            details[TARGET_CONCEPT_NAME_B] = str(row[TARGET_CONCEPT_NAME_B])
        gold_standard[int(row[SOURCE_CONCEPT_ID])] = details

    return gold_standard


def evaluate_concept_search(
    search_results: pd.DataFrame,
    output_file: str | Path,
    gold_standard_file: str = "data/gold_standards/exact_matching_gs.csv",
    source_id_column: str = "source_concept_id",
    term_column: str = "cleaned_term",
    matched_concept_id_column: str = "matched_concept_id",
    matched_concept_name_column: str = "matched_concept_name",
    match_rank_column: str = "match_rank",
) -> None:
    """
    Evaluate the concept search results against the gold standard.

    Args:
        search_results: Pandas DataFrame containing the results of the concept search.
        output_file: Path to save the evaluation results.
        gold_standard_file: Path to the CSV file containing the gold standard mappings.
        source_id_column: Name of the column in the search results with source concept IDs.
        term_column: Name of the column in the search results with the search terms.
        matched_concept_id_column: Name of the column in the search results with matched concept IDs.
        matched_concept_name_column: Name of the column in the search results with matched concept names.
        match_rank_column: Name of the column in the search results with the rank of the matched concepts.

    Returns:
        None. Execution results are written to the specified output file.
    """
    detail_strings = []
    # gold_standard = _load_gold_standard(resolve_path(gold_standard_file))
    gold_standard = pd.read_csv(resolve_path(gold_standard_file))
    evaluated_gs_count = 0
    mean_average_precision = 0
    recall_1 = 0
    recall_3 = 0
    recall_10 = 0
    recall_25 = 0

    grouped = search_results.groupby(source_id_column)
    for source_id, group in grouped:
        gs_entry = gold_standard[gold_standard[SOURCE_CONCEPT_ID] == source_id]
        if gs_entry.empty:
            continue
        gs_entry = gs_entry.iloc[0]
        gs_source_term = gs_entry[SOURCE_TERM]
        gs_concept_id = gs_entry[TARGET_CONCEPT_ID]
        gs_concept_id_b = gs_entry[TARGET_CONCEPT_ID_B]
        gold_predicate = gs_entry[PREDICATE]
        gold_predicate_b = gs_entry[PREDICATE_B]
        if gold_predicate == BROAD_MATCH:
            gs_concept_id = None
        if gold_predicate_b == BROAD_MATCH:
            gs_concept_id_b = None
        if gs_concept_id is None and (gs_concept_id_b is None or math.isnan(gs_concept_id_b)):
            continue
        evaluated_gs_count = evaluated_gs_count + 1
        gs_rank = group.loc[group[matched_concept_id_column] == gs_concept_id, match_rank_column]
        if gs_concept_id_b is not None:
            gs_rank_b = group.loc[group[matched_concept_id_column] == gs_concept_id_b, match_rank_column]
            if not gs_rank_b.empty:
                if gs_rank.empty or gs_rank_b.iloc[0] < gs_rank.iloc[0]:
                    gs_rank = gs_rank_b
                    gs_concept_id = gs_concept_id_b
        detail_strings.append(f"Source term: {gs_source_term} ({source_id})")
        detail_strings.append(f"Searched term: {group[term_column].iloc[0]}")
        if gs_rank.empty:
            detail_strings.append("Gold standard concept not found")
            gs_concept_name = gs_entry[TARGET_CONCEPT_NAME]
            detail_strings.append(f"Correct target was: {gs_concept_name} ({gs_concept_id})")
        else:
            gs_rank = gs_rank.iloc[0]
            detail_strings.append(f"Gold standard concept rank: {gs_rank}")
            mean_average_precision = mean_average_precision + 1 / gs_rank
            if gs_rank <= 1:
                recall_1 = recall_1 + 1
            if gs_rank <= 3:
                recall_3 = recall_3 + 1
            if gs_rank <= 10:
                recall_10 = recall_10 + 1
            if gs_rank <= 25:
                recall_25 = recall_25 + 1

        detail_strings.append("")

        table = group[[match_rank_column, matched_concept_id_column, matched_concept_name_column]].copy()
        correct = np.where(table[matched_concept_id_column] == gs_concept_id, "Yes", "")
        if gs_concept_id_b:
            correct_b = np.where(table[matched_concept_id_column] == gs_concept_id_b, "Yes", "")
            correct = np.where(correct == "Yes", "Yes", correct_b)

        table.insert(1, "Correct", correct)
        detail_strings.append(table.to_string(index=False))
        detail_strings.append("")

    mean_average_precision = mean_average_precision / evaluated_gs_count
    recall_1 = recall_1 / evaluated_gs_count
    recall_3 = recall_3 / evaluated_gs_count
    recall_10 = recall_10 / evaluated_gs_count
    recall_25 = recall_25 / evaluated_gs_count

    summary_strings = [
        f"Evaluated gold standard concepts: {evaluated_gs_count}",
        f"Mean Average Precision: {mean_average_precision}",
        f"Recall@1: {recall_1}",
        f"Recall@3: {recall_3}",
        f"Recall@10: {recall_10}",
        f"Recall@25: {recall_25}",
    ]

    with open(output_file, "w", encoding="UTF-8") as f:
        f.write("\n".join(summary_strings))
        f.write("\n\n")
        f.write("\n".join(detail_strings))
        f.write("\n")

    print(f"Evaluation complete. Results written to {output_file}")
