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


import requests
from typing import Optional, List

import pandas as pd
from ariadne.vector_search.abstract_concept_searcher import AbstractConceptSearcher

_HECATE_SEARCH_URL = "https://hecate.pantheon-hds.com/api/search"
_HECATE_SEARCH_STANDARD_URL = "https://hecate.pantheon-hds.com/api/search_standard"


class HecateConceptSearcher(AbstractConceptSearcher):

    """
    A concept searcher that uses the OHDSI Hecate API to find concepts based on query strings.
    """

    def __init__(
        self,
        for_evaluation: bool = False,
        standard_concept: Optional[str] = None,
        domain_ids: Optional[List[str]] = None,
        concept_class_ids: Optional[List[str]] = None,
        vocabulary_ids: Optional[List[str]] = None,
    ):
        """
        Initializes the HecateConceptSearcher.

        Args:
            for_evaluation: If True, configures the searcher for evaluation purposes.
        """
        self.for_evaluation = for_evaluation

        if for_evaluation:
            print("HecateConceptSearcher initialized in evaluation mode.")
            self.default_url = _HECATE_SEARCH_STANDARD_URL
            self.default_params = {
                "standard_concept": "S",
                "domain_id": "Condition,Observation,Measurement,Procedure",
                "concept_class_id": "3-dig billing code,3-dig nonbill code,4-dig billing code,Answer,Claims Attachment,Clinical Finding,Clinical Observation,Context-dependent,CPT4,CPT4 Modifier,Disorder,Event,Genetic Variation,HCPCS,Histopattern,ICD10PCS,ICD10PCS Hierarchy,ICDO Condition,ICDO Histology,Ingredient,Lab Test,MDC,Metastasis,MS-DRG,NAACCR Variable,Observable Entity,Procedure,Question,Social Context,Staging / Scales,Staging/Grading,Survey,Topic,Topography,Value,Variable",
                "exclude_vocabulary_id": "ICD9CM,ICD10CM,ICD10,ICD10CN,ICD10GM,CIM10,ICDO3,KCD7,Read",
            }
        else:
            print("HecateConceptSearcher initialized in standard mode.")
            self.default_url = _HECATE_SEARCH_STANDARD_URL
            self.default_params = {}
            normalized_standard = (standard_concept or "S").strip()
            if normalized_standard.lower() == "none":
                self.default_url = _HECATE_SEARCH_URL
                self.default_params["standard_concept"] = "None"
            elif normalized_standard != "S":
                self.default_url = _HECATE_SEARCH_URL
                self.default_params["standard_concept"] = normalized_standard
            if domain_ids:
                self.default_params["domain_id"] = ",".join(domain_ids)
            if concept_class_ids:
                self.default_params["concept_class_id"] = ",".join(concept_class_ids)
            if vocabulary_ids:
                self.default_params["vocabulary_id"] = ",".join(vocabulary_ids)

    def _resolve_endpoint_for_standard_concept(self, standard_concept: Optional[str]) -> tuple[str, dict]:
        params = dict(self.default_params)
        endpoint = self.default_url

        if standard_concept is None:
            return endpoint, params

        normalized_standard = standard_concept.strip()
        if normalized_standard.lower() == "none":
            endpoint = _HECATE_SEARCH_URL
            params["standard_concept"] = "None"
        elif normalized_standard == "S":
            endpoint = _HECATE_SEARCH_STANDARD_URL
            params.pop("standard_concept", None)
        else:
            endpoint = _HECATE_SEARCH_URL
            params["standard_concept"] = normalized_standard

        return endpoint, params

    def search_term(
        self,
        query_string: str,
        limit: int = 25,
        standard_concept: Optional[str] = None,
        domain_ids: Optional[List[str]] = None,
        concept_class_ids: Optional[List[str]] = None,
        vocabulary_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Searches for concepts matching the given query string.

        Args:
            query_string: The term to search for.
            limit: The maximum number of results to return.

        Returns:
            A DataFrame containing the matching concepts, with the same columns as the concept table in the OMOP CDM,
            plus a 'score' column indicating the relevance score from the search.

        """

        endpoint, params = self._resolve_endpoint_for_standard_concept(standard_concept)
        params.update({"q": query_string, "limit": limit})
        if domain_ids:
            params["domain_id"] = ",".join(domain_ids)
        if concept_class_ids:
            params["concept_class_id"] = ",".join(concept_class_ids)
        if vocabulary_ids:
            params["vocabulary_id"] = ",".join(vocabulary_ids)

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
            if vocabulary_ids and "vocabulary_id" in results_df.columns:
                results_df = results_df[results_df["vocabulary_id"].isin(vocabulary_ids)]
            return results_df

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
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
        limit: int = 25,
        standard_concept: Optional[str] = None,
        domain_ids: Optional[List[str]] = None,
        concept_class_ids: Optional[List[str]] = None,
        vocabulary_ids: Optional[List[str]] = None,
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
            limit: The maximum number of results to return for each term.

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
                limit=limit,
                standard_concept=standard_concept,
                domain_ids=domain_ids,
                concept_class_ids=concept_class_ids,
                vocabulary_ids=vocabulary_ids,
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
    concept_searcher = HecateConceptSearcher()
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
    results_df = concept_searcher.search_terms(df, term_column="cleaned_term", limit=10)
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
