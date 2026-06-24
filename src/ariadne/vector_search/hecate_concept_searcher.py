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
import re
from typing import Optional

import requests

import pandas as pd
from ariadne.utils.settings import VectorSearchSettings
from ariadne.vector_search.abstract_concept_searcher import AbstractConceptSearcher

_HECATE_SEARCH_URL = "https://hecate.pantheon-hds.com/api/search"
_HECATE_SEARCH_STANDARD_URL = "https://hecate.pantheon-hds.com/api/search_standard"


class HecateConceptSearcher(AbstractConceptSearcher):

    """
    A concept searcher that uses the OHDSI Hecate API to find concepts based on query strings.
    """

    def __init__(
        self,
        settings: VectorSearchSettings,
    ):
        """
        Initializes the HecateConceptSearcher.

        Args:
            settings: Vector search settings controlling endpoint selection,
                candidate limits, and query filters.
        """
        self.settings = settings
        self._sorted_substrings_to_remove = sorted(
            self.settings.substrings_to_remove,
            key=len,
            reverse=True,
        )
        self.default_url = _HECATE_SEARCH_STANDARD_URL
        self.default_params = {}
        standard_concepts = self.settings.filter.standard_concept
        uses_standard_only_endpoint = len(standard_concepts) == 1 and standard_concepts[0] == "S"
        if not uses_standard_only_endpoint:
            self.default_url = _HECATE_SEARCH_URL
            self.default_params["standard_concept"] = ",".join(standard_concepts)

        if self.settings.filter.domain_ids:
            self.default_params["domain_id"] = ",".join(self.settings.filter.domain_ids)
        if self.settings.filter.concept_class_ids:
            self.default_params["concept_class_id"] = ",".join(self.settings.filter.concept_class_ids)
        if self.settings.filter.vocabulary_ids:
            self.default_params["vocabulary_id"] = ",".join(self.settings.filter.vocabulary_ids)
        if self.settings.filter.exclude_vocabulary_ids:
            self.default_params["exclude_vocabulary_id"] = ",".join(self.settings.filter.exclude_vocabulary_ids)

    def search_term(
        self,
        query_string: str,
    ) -> Optional[pd.DataFrame]:
        """
        Searches for concepts matching the given query string.

        Args:
            query_string: The term to search for.

        Returns:
            A DataFrame containing the matching concepts, with the same columns as the concept table in the OMOP CDM,
            plus a 'score' column indicating the relevance score from the search.

        """
        if query_string == 'Kent Pharma (UK) Ltd':
            print("check")


        # Remove substrings from query_string
        cleaned_query = query_string
        for substring in self._sorted_substrings_to_remove:
            cleaned_query = re.sub(re.escape(substring), "", cleaned_query, flags=re.IGNORECASE)
        cleaned_query = cleaned_query.strip()

        endpoint = self.default_url
        params = dict(self.default_params)
        params.update({"q": cleaned_query, "limit": self.settings.max_candidates})
        response = None

        try:
            response = requests.get(endpoint, params=params, timeout=15)
            response.raise_for_status()
            terms = response.json()
            concepts = []
            for term in terms:
                # add score:
                for concept in term.get("concepts", []):
                    concept["score"] = term.get("score", None)
                concepts.extend(term.get("concepts", []))
            results_df = pd.DataFrame(concepts)
            if results_df.empty:
                return results_df
            if self.settings.filter.vocabulary_ids and "vocabulary_id" in results_df.columns:
                results_df = results_df[results_df["vocabulary_id"].isin(self.settings.filter.vocabulary_ids)]
            return results_df

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            if response is not None:
                print(f"Response status code: {response.status_code}")
                print(f"Response content: {response.text}")
        except requests.exceptions.ConnectionError as conn_err:
            print(f"Connection error occurred: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            print(f"The request timed out: {timeout_err}")
        except requests.exceptions.RequestException as err:
            print(f"An unexpected error occurred: {err}")

        return None

    def search_terms(
        self,
        df: pd.DataFrame,
        term_column: str,
        matched_concept_id_column: str = "matched_concept_id",
        matched_concept_name_column: str = "matched_concept_name",
        match_score_column: str = "match_score",
        match_rank_column: str = "match_rank",
    ) -> pd.DataFrame:
        """
        Searches the Hecate API for concepts matching terms in a DataFrame column.

        Args:
            df: DataFrame containing the terms to search for.
            term_column: Name of the column with terms to search.
            matched_concept_id_column: Name of the column to store matched concept IDs.
            matched_concept_name_column: Name of the column to store matched concept names.
            match_score_column: Name of the column to store match scores.
            match_rank_column: Name of the column to store match ranks.
        Returns:
            A DataFrame containing the same columns as the input dataframe plus the matching concepts for each term. For
            each term in the input dataframe, multiple rows will be returned corresponding to each matching concept.

        """

        all_results = []
        for index, row in df.iterrows():
            term = row[term_column]
            print(f"Processing term '{term}'")
            results = self.search_term(
                term,
            )
            if results is not None:
                rows = []
                for rank, (_, concept) in enumerate(results.iterrows(), start=1):
                    rows.append(
                        {
                            matched_concept_id_column: concept["concept_id"],
                            matched_concept_name_column: concept["concept_name"],
                            match_score_column: concept["score"],
                            match_rank_column: rank,
                        }
                    )
                results = pd.DataFrame(rows)
                results[match_rank_column] = range(1, len(results) + 1)
                orig_cols = list(df.columns)
                new_columns = list(results.columns)
                results[term_column] = term
                for col in df.columns:
                    results[col] = row[col]
                results = results[orig_cols + new_columns]
                if not results.empty:
                    all_results.append(results)

        if not all_results:
            return pd.DataFrame()

        return pd.concat(all_results, ignore_index=True)


if __name__ == "__main__":
    concept_searcher = HecateConceptSearcher(settings=VectorSearchSettings())
    search_results = concept_searcher.search_term("Acute myocardial infarction")
    print(search_results)

    df = pd.DataFrame(
        {
            "concept_id_1": [1326717, 201820],
            "cleaned_term": [
                "Acute myocardial infarction",
                "Chronic kidney disease",
            ],
        }
    )
    results_df = concept_searcher.search_terms(df, term_column="cleaned_term")
    print(results_df)
    print(results_df.columns)

    # concept_searcher.process_file(
    #     source_file="./files/source_terms_not_unspecified.csv",
    #     results_file="./files/hecate_search_results_not_unspecified.csv"
    # )

    # results_df = pd.read_csv("./files/hecate_search_results_not_unspecified.csv")
    # results_df.sort_values(by=["concept_id_1", "score"], ascending=[True, False], inplace=True)
    # results_df["rank"] = results_df.groupby("concept_id_1").cumcount() + 1
    # results_df.to_csv("./files/hecate_search_results_not_unspecified_fix.csv", index=False)
