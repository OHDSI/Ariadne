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

import os
import pickle
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv

from ariadne.utils.utils import get_environment_variable
from ariadne.utils.gen_ai_api import get_embedding_vectors
from ariadne.vector_search.abstract_concept_searcher import AbstractConceptSearcher


load_dotenv()


class PgvectorConceptSearcher(AbstractConceptSearcher):

    """
    A concept searcher that uses pgvector in a PostgreSQL database to find concepts based on embedding vectors.
    """

    def __init__(self, for_evaluation: bool = False, include_synonyms: bool = True, include_mapped_terms: bool = True):
        self.for_evaluation = for_evaluation
        self.include_synonyms = include_synonyms
        self.include_mapped_terms = include_mapped_terms
        self.cost = 0.0

        if for_evaluation:
            self.concept_classes_to_ignore = [
                "Disposition",
                "Morph Abnormality",
                "Organism",
                "Qualifier Value",
                "Substance",
                "ICDO Condition",
            ]
            self.vocabularies_to_ignore = [
                "ICD9CM",
                "ICD10CM",
                "ICD10",
                "ICD10CN",
                "ICD10GM",
                "CIM10",
                "ICDO3",
                "KCD7",
                "Read",
            ]
            print("ConceptSearcher initialized in evaluation mode.")
        else:
            self.concept_classes_to_ignore = None
            self.vocabularies_to_ignore = None

        connection = psycopg.connect(os.getenv("vocab_connection_string").replace("+psycopg", ""))
        register_vector(connection)
        with connection.cursor() as cur:
            cur.execute("SET hnsw.ef_search = 1000")
            cur.execute("SET hnsw.iterative_scan = relaxed_order")
        self.connection = connection

    def close(self):
        self.connection.close()

    def _search_pgvector(self, source_vector: np.ndarray, limit: int) -> List:
        if self.concept_classes_to_ignore is None:
            ignore_string_class = "'dummy'"
        else:
            ignore_string_class = ", ".join(f"'{y}'" for y in self.concept_classes_to_ignore)

        if self.include_synonyms:
            term_type_clause = ""
        else:
            term_type_clause = "AND vectors.term_type = 'Name'"

        vocabulary_schema = get_environment_variable("VOCAB_SCHEMA")
        vector_table = get_environment_variable("VOCAB_VECTOR_TABLE")

        if self.include_mapped_terms:
            if self.vocabularies_to_ignore is None:
                ignore_string_vocab = "'dummy'"
            else:
                ignore_string_vocab = ", ".join(f"'{x}'" for x in self.vocabularies_to_ignore)
            query = f"""
                WITH target_concept AS (
                    SELECT concept_id,
                        concept_name,
                        domain_id,
                        concept_class_id,
                        vocabulary_id,
                        MIN(relevance_score) AS relevance_score
                    FROM (
                        (
                            SELECT concept.concept_id,
                                concept.concept_name,
                                concept.domain_id,
                                concept.concept_class_id,
                                concept.vocabulary_id,
                                embedding_vector <=> %s AS relevance_score
                            FROM {vocabulary_schema}.{vector_table} vectors
                            INNER JOIN {vocabulary_schema}.concept source_concept
                                ON vectors.concept_id = source_concept.concept_id
                            INNER JOIN {vocabulary_schema}.concept_relationship
                                ON vectors.concept_id = concept_relationship.concept_id_1
                            INNER JOIN {vocabulary_schema}.concept
                                ON concept_relationship.concept_id_2 = concept.concept_id
                            WHERE relationship_id = 'Maps to'
                                AND source_concept.vocabulary_id NOT IN ({ignore_string_vocab})
                                AND concept.concept_class_id NOT IN ({ignore_string_class})
                                {term_type_clause}
                            ORDER BY embedding_vector <=> %s
                            LIMIT {limit * 4} -- May have duplicates due to synonyms
                        )

                        UNION ALL

                        (
                            SELECT concept.concept_id,
                                concept.concept_name,
                                concept.domain_id,
                                concept.concept_class_id,
                                concept.vocabulary_id,
                                embedding_vector <=> %s AS relevance_score
                            FROM {vocabulary_schema}.{vector_table} vectors
                            INNER JOIN {vocabulary_schema}.concept
                                ON vectors.concept_id = concept.concept_id
                            WHERE standard_concept = 'S'
                                AND concept.concept_class_id NOT IN ({ignore_string_class})
                                {term_type_clause}
                            ORDER BY embedding_vector <=> %s
                            LIMIT {limit * 4} -- May have duplicates due to synonyms
                        )
                    ) tmp
                    GROUP BY concept_id,
                        concept_name,
                        domain_id,
                        concept_class_id,
                        vocabulary_id
                )
                SELECT target_concept.concept_id,
                    target_concept.concept_name,
                    target_concept.domain_id,
                    target_concept.concept_class_id,
                    target_concept.vocabulary_id,
                    target_concept.relevance_score
                FROM target_concept
                ORDER BY relevance_score
                LIMIT {limit};
            """
            with self.connection.cursor() as cur:
                cur.execute(query, (source_vector, source_vector, source_vector, source_vector))
                results = cur.fetchall()
        else:
            query = f"""
                WITH target_concept AS (
                    SELECT concept_id,
                        concept_name,
                        domain_id,
                        concept_class_id,
                        vocabulary_id,
                        MIN(relevance_score) AS relevance_score
                    FROM (
                        SELECT concept.concept_id,
                            concept.concept_name,
                            concept.domain_id,
                            concept.concept_class_id,
                            concept.vocabulary_id,
                            embedding_vector <=> %s AS relevance_score
                        FROM {vocabulary_schema}.{vector_table} vectors
                        INNER JOIN {vocabulary_schema}.concept
                            ON vectors.concept_id = concept.concept_id
                        WHERE standard_concept = 'S'
                            AND concept.concept_class_id NOT IN ({ignore_string_class})
                            {term_type_clause}
                        ORDER BY embedding_vector <=> %s
                        LIMIT {limit * 4} -- May have duplicates due to synonyms
                    ) tmp
                    GROUP BY concept_id,
                        concept_name,
                        domain_id,
                        concept_class_id,
                        vocabulary_id
                )
                SELECT target_concept.concept_id,
                    target_concept.concept_name,
                    target_concept.domain_id,
                    target_concept.concept_class_id,
                    target_concept.vocabulary_id,
                    target_concept.relevance_score
                FROM target_concept
                ORDER BY relevance_score
                LIMIT {limit};
            """
            with self.connection.cursor() as cur:
                cur.execute(query, (source_vector, source_vector))
                results = cur.fetchall()

        return results

    def search(self, term: str, limit: int = 25) -> Optional[pd.DataFrame]:
        """
        Searches for concepts matching the given term.

        Args:
            term: The clinical term to search for.
            limit: The maximum number of results to return.

        Returns:
            A DataFrame containing the matching concepts, or None if no matches are found.
        """
        vectors_with_usage = get_embedding_vectors([term])
        self.cost = self.cost + vectors_with_usage["usage"]["total_cost_usd"]
        vector = vectors_with_usage["embeddings"][0]
        results = self._search_pgvector(vector, limit)
        if not results:
            return None
        df = pd.DataFrame(
            results,
            columns=[
                "concept_id",
                "concept_name",
                "domain_id",
                "concept_class_id",
                "synonyms",
                "score",
            ],
        )
        return df

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
        Searches for concepts matching terms in a DataFrame column.

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

        vectors_with_usage = get_embedding_vectors(df[term_column].tolist())
        self.cost = self.cost + vectors_with_usage["usage"]["total_cost_usd"]
        vectors = vectors_with_usage["embeddings"]

        all_results = []
        for index, row in df.iterrows():
            term = row[term_column]
            print(f"Processing term '{term}'")
            vector = vectors[index]
            results = self._search_pgvector(vector, limit=limit)
            results = pd.DataFrame(
                results,
                columns=[
                    matched_concept_id_column,
                    matched_concept_name_column,
                    matched_domain_id_column,
                    matched_concept_class_id_column,
                    matched_vocabulary_id_column,
                    match_score_column,
                ],
            )
            results[match_rank_column] = range(1, len(results) + 1)
            orig_cols = list(df.columns)
            new_columns = list(results.columns)
            results[term_column] = term
            for col in df.columns:
                results[col] = row[col]
            results = results[orig_cols + new_columns]
            all_results.append(results)

        all_results = pd.concat(all_results)
        return all_results

    def get_total_cost(self) -> float:
        """
        Returns the total cost incurred for embedding vector calls

        Returns:
            Total cost in USD.
        """

        return self.cost


if __name__ == "__main__":
    concept_searcher = PgvectorConceptSearcher()
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
    results_df = concept_searcher.search_in_df(df, term_column="stripped_concept_name_1", limit=10)
    print(results_df)
    print(results_df.columns)

    print(f"Total cost incurred: ${concept_searcher.get_total_cost():.6f} USD")

    concept_searcher.close()
