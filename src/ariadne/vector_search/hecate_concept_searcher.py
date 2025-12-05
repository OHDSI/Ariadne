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

import pandas as pd
from ariadne.vector_search.abstract_concept_searcher import AbstractConceptSearcher
from pandas.core.interchange.dataframe_protocol import DataFrame

_HECATE_URL = "https://hecate.pantheon-hds.com/api/search_standard"


class HecateConceptSearcher(AbstractConceptSearcher):

    def __init__(self, for_evaluation: bool = False):
        """
        Initializes the HecateConceptSearcher.

        Args:
            for_evaluation: If True, configures the searcher for evaluation purposes.
        """
        self.for_evaluation = for_evaluation

        if for_evaluation:
            print("HecateConceptSearcher initialized in evaluation mode.")
            self.default_params = {
                "standard_concept": "S",
                "domain_id": "Condition,Observation,Measurement,Procedure",
                "concept_class_id": "3-dig billing code,3-dig nonbill code,4-dig billing code,Answer,Claims Attachment,Clinical Finding,Clinical Observation,Context-dependent,CPT4,CPT4 Modifier,Disorder,Event,Genetic Variation,HCPCS,Histopattern,ICD10PCS,ICD10PCS Hierarchy,ICDO Condition,ICDO Histology,Ingredient,Lab Test,MDC,Metastasis,MS-DRG,NAACCR Variable,Observable Entity,Procedure,Question,Social Context,Staging / Scales,Staging/Grading,Survey,Topic,Topography,Value,Variable",
                "exclude_vocabulary_id": "ICD9CM,ICD10CM,ICD10,ICD10CN,ICD10GM,CIM10,ICDO3,KCD7,Read",
            }
        else:
            print("HecateConceptSearcher initialized in standard mode.")
            self.default_params = {
                "standard_concept": "S",
            }

    def search(self, query_string: str, limit: int = 25) -> DataFrame:
        """
        Searches the Hecate API for concepts matching the given query string.

        Args:
            query_string: The term to search for.
            limit: The maximum number of results to return.

        Returns:
            A DataFrame containing the matching concepts, with the same columns as the concept table in the OMOP CDM,
            plus a 'score' column indicating the relevance score from the search.

        """

        params = {"q": query_string, "limit": limit}
        params.update(self.default_params)

        try:
            response = requests.get(_HECATE_URL, params=params, timeout=15)
            response.raise_for_status()
            terms = response.json()
            concepts = []
            for term in terms:
                # add score:
                for concept in term.get("concepts", []):
                    concept["score"] = term.get("score", None)
                concepts.extend(term.get("concepts", []))
            return pd.DataFrame(concepts)

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

    def search_in_df(
        self,
        df: pd.DataFrame,
        term_column: str,
        matched_concept_id_column: str = "matched_concept_id",
        matched_concept_name_column: str = "matched_concept_name",
        matched_domain_id_column: str = "matched_domain_id",
        matched_concept_class_id_column: str = "matched_concept_class_id",
        matched_vocabulary_id_column: str = "matched_vocabulary_id",
        match_score_column: str = "match_score",
        match_rank_column: str = "match_rank",
        limit: int = 25,
    ) -> pd.DataFrame:
        """
        Searches the Hecate API for concepts matching terms in a DataFrame column.

        Args:
            df: DataFrame containing the terms to search for.
            term_column: Name of the column with terms to search.
            matched_concept_id_column: Name of the column to store matched concept IDs.
            matched_concept_name_column: Name of the column to store matched concept names.
            matched_domain_id_column: Name of the column to store matched domain IDs.
            matched_concept_class_id_column: Name of the column to store matched concept class IDs.
            matched_vocabulary_id_column: Name of the column to store matched vocabulary IDs.
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
            results = self.search(term, limit=limit)
            if results is not None:
                for rank, (_, concept) in enumerate(results.iterrows(), start=1):
                    all_results.append(
                        {
                            term_column: term,
                            matched_concept_id_column: concept["concept_id"],
                            matched_concept_name_column: concept["concept_name"],
                            matched_domain_id_column: concept["domain_id"],
                            matched_concept_class_id_column: concept[
                                "concept_class_id"
                            ],
                            matched_vocabulary_id_column: concept["vocabulary_id"],
                            match_score_column: concept["score"],
                            match_rank_column: rank,
                        }
                    )
        results_df = pd.DataFrame(all_results)
        return results_df


if __name__ == "__main__":
    concept_searcher = HecateConceptSearcher()
    search_results = concept_searcher.search("Acute myocardial infarction")
    print(search_results)

    df = pd.DataFrame(
        {
            "concept_id_1": [1326717, 201820],
            "stripped_concept_name_1": [
                "Acute myocardial infarction",
                "Chronic kidney disease",
            ],
        }
    )
    results_df = concept_searcher.search_in_df(
        df, term_column="stripped_concept_name_1", limit=10
    )
    print(results_df)

    # concept_searcher.process_file(
    #     source_file="./files/source_terms_not_unspecified.csv",
    #     results_file="./files/hecate_search_results_not_unspecified.csv"
    # )

    # results_df = pd.read_csv("./files/hecate_search_results_not_unspecified.csv")
    # results_df.sort_values(by=["concept_id_1", "score"], ascending=[True, False], inplace=True)
    # results_df["rank"] = results_df.groupby("concept_id_1").cumcount() + 1
    # results_df.to_csv("./files/hecate_search_results_not_unspecified_fix.csv", index=False)
