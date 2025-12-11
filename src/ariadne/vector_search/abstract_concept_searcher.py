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

from abc import ABC as ABS, abstractmethod
from typing import Optional
import pandas as pd


class AbstractConceptSearcher(ABS):
    """Abstract base class for concept searchers."""

    @abstractmethod
    def search_term(self, term: str, limit: int = 25) -> Optional[pd.DataFrame]:
        """
        Searches for concepts matching the given term.

        Args:
            term: The clinical term to search for.
            limit: The maximum number of results to return.

        Returns:
            A DataFrame containing the matching concepts, or None if no matches are found.
        """
        pass

    @abstractmethod
    def search_terms(
        self,
        df: pd.DataFrame,
        term_column: str,
        matched_concept_id_column: str = "matched_concept_id",
        matched_concept_name_column: str = "matched_concept_name",
        match_score_column: str = "match_score",
        match_rank_column: str = "match_rank",
        limit: int = 25,
    ) -> pd.DataFrame:
        """
        Searches for concepts matching terms in a DataFrame column.

        Args:
            df: DataFrame containing the terms to search for.
            term_column: Name of the column with terms to search.
            matched_concept_id_column: Name of the column to store matched concept IDs.
            matched_concept_name_column: Name of the column to store matched concept names.
            match_score_column: Name of the column to store match scores.
            match_rank_column: Name of the column to store match ranks.
            limit: The maximum number of results to return for each term.

        Returns:
            A DataFrame containing the same columns as the input dataframe plus the matching concepts for each term. For
            each term in the input dataframe, multiple rows will be returned corresponding to each matching concept.

        """
        pass
