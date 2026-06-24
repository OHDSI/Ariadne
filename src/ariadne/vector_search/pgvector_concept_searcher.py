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
from typing import List, Optional

import numpy as np
import pandas as pd
import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv

from ariadne.utils.utils import get_environment_variable
from ariadne.utils.gen_ai_api import get_embedding_vectors
from ariadne.utils.settings import PgvectorSearchSettings
from ariadne.vector_search.abstract_concept_searcher import AbstractConceptSearcher

load_dotenv()


class PgvectorConceptSearcher(AbstractConceptSearcher):
    """
    A concept searcher that uses pgvector in a PostgreSQL database to find concepts based on embedding vectors.
    """

    def __init__(self, settings: PgvectorSearchSettings):
        self.settings = settings
        self._sorted_substrings_to_remove = sorted(
            self.settings.substrings_to_remove,
            key=len,
            reverse=True,
        )
        self.include_synonyms = settings.include_synonyms
        self.include_mapped_terms = settings.include_mapped_terms
        self.cost = 0.0

        connection = psycopg.connect(get_environment_variable("VOCAB_CONNECTION_STRING").replace("+psycopg", ""))
        register_vector(connection)
        with connection.cursor() as cur:
            cur.execute("SET hnsw.ef_search = 1000")
            cur.execute("SET hnsw.iterative_scan = relaxed_order")
        self.connection = connection

    def close(self):
        self.connection.close()

    def _search_pgvector(self, source_vector: np.ndarray) -> List:
        limit = self.settings.max_candidates

        concept_class_clause = ""
        if self.settings.filter.concept_class_ids:
            concept_classes = ", ".join(f"'{concept_class}'" for concept_class in self.settings.filter.concept_class_ids)
            concept_class_clause = f"AND concept.concept_class_id IN ({concept_classes})"

        concept_class_exclude_clause = ""
        if self.settings.filter.exclude_concept_class_ids:
            concept_classes_to_ignore = ", ".join(
                f"'{concept_class}'" for concept_class in self.settings.filter.exclude_concept_class_ids
            )
            concept_class_exclude_clause = f"AND concept.concept_class_id NOT IN ({concept_classes_to_ignore})"

        domain_clause = ""
        if self.settings.filter.domain_ids:
            domains = ", ".join(f"'{domain}'" for domain in self.settings.filter.domain_ids)
            domain_clause = f"AND concept.domain_id IN ({domains})"

        vocabulary_clause = ""
        if self.settings.filter.vocabulary_ids:
            vocabularies = ", ".join(f"'{vocab}'" for vocab in self.settings.filter.vocabulary_ids)
            vocabulary_clause = f"AND concept.vocabulary_id IN ({vocabularies})"

        exclude_vocabulary_clause = ""
        source_exclude_vocabulary_clause = ""
        if self.settings.filter.exclude_vocabulary_ids:
            excluded_vocabs = ", ".join(f"'{vocab}'" for vocab in self.settings.filter.exclude_vocabulary_ids)
            exclude_vocabulary_clause = f"AND concept.vocabulary_id NOT IN ({excluded_vocabs})"
            source_exclude_vocabulary_clause = f"AND source_concept.vocabulary_id NOT IN ({excluded_vocabs})"

        standard_clause = ""
        if self.settings.filter.standard_concept:
            include_null_standard = "None" in self.settings.filter.standard_concept
            explicit_standard = [
                value for value in self.settings.filter.standard_concept if value != "None"
            ]
            if include_null_standard and explicit_standard:
                explicit_standard_clause = ", ".join(f"'{value}'" for value in explicit_standard)
                standard_clause = (
                    f"AND (concept.standard_concept IN ({explicit_standard_clause}) "
                    "OR concept.standard_concept IS NULL)"
                )
            elif include_null_standard:
                standard_clause = "AND concept.standard_concept IS NULL"
            else:
                explicit_standard_clause = ", ".join(f"'{value}'" for value in explicit_standard)
                standard_clause = f"AND concept.standard_concept IN ({explicit_standard_clause})"

        if self.include_synonyms:
            term_type_clause = ""
        else:
            term_type_clause = "AND vectors.term_type = 'Name'"

        vocabulary_schema = get_environment_variable("VOCAB_SCHEMA")
        vector_table = get_environment_variable("VOCAB_VECTOR_TABLE")

        if self.include_mapped_terms:
            query = f"""
                WITH target_concept AS (
                    SELECT concept_id,
                        concept_name,
                        MIN(relevance_score) AS relevance_score
                    FROM (
                        (
                            SELECT concept.concept_id,
                                concept.concept_name,
                                embedding_vector <=> %s AS relevance_score
                            FROM {vocabulary_schema}.{vector_table} vectors
                            INNER JOIN {vocabulary_schema}.concept source_concept
                                ON vectors.concept_id = source_concept.concept_id
                            INNER JOIN {vocabulary_schema}.concept_relationship
                                ON vectors.concept_id = concept_relationship.concept_id_1
                            INNER JOIN {vocabulary_schema}.concept
                                ON concept_relationship.concept_id_2 = concept.concept_id
                            WHERE relationship_id = 'Maps to'
                                {source_exclude_vocabulary_clause}
                                {concept_class_clause}
                                {concept_class_exclude_clause}
                                {domain_clause}
                                {vocabulary_clause}
                                {exclude_vocabulary_clause}
                                {standard_clause}
                                {term_type_clause}
                            ORDER BY embedding_vector <=> %s
                            LIMIT {limit * 4} -- May have duplicates due to synonyms
                        )

                        UNION ALL

                        (
                            SELECT concept.concept_id,
                                concept.concept_name,
                                embedding_vector <=> %s AS relevance_score
                            FROM {vocabulary_schema}.{vector_table} vectors
                            INNER JOIN {vocabulary_schema}.concept
                                ON vectors.concept_id = concept.concept_id
                            WHERE 1=1
                                {standard_clause}
                                {concept_class_clause}
                                {concept_class_exclude_clause}
                                {domain_clause}
                                {vocabulary_clause}
                                {exclude_vocabulary_clause}
                                {term_type_clause}
                            ORDER BY embedding_vector <=> %s
                            LIMIT {limit * 4} -- May have duplicates due to synonyms
                        )
                    ) tmp
                    GROUP BY concept_id,
                        concept_name
                )
                SELECT target_concept.concept_id,
                    target_concept.concept_name,
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
                        MIN(relevance_score) AS relevance_score
                    FROM (
                        SELECT concept.concept_id,
                            concept.concept_name,
                            embedding_vector <=> %s AS relevance_score
                        FROM {vocabulary_schema}.{vector_table} vectors
                        INNER JOIN {vocabulary_schema}.concept
                            ON vectors.concept_id = concept.concept_id
                        WHERE 1=1
                            {standard_clause}
                            {concept_class_clause}
                            {concept_class_exclude_clause}
                            {domain_clause}
                            {vocabulary_clause}
                            {exclude_vocabulary_clause}
                            {term_type_clause}
                        ORDER BY embedding_vector <=> %s
                        LIMIT {limit * 4} -- May have duplicates due to synonyms
                    ) tmp
                    GROUP BY concept_id,
                        concept_name
                )
                SELECT target_concept.concept_id,
                    target_concept.concept_name,
                    target_concept.relevance_score
                FROM target_concept
                ORDER BY relevance_score
                LIMIT {limit};
            """
            with self.connection.cursor() as cur:
                cur.execute(query, (source_vector, source_vector))
                results = cur.fetchall()

        return results

    def search_term(self, term: str) -> Optional[pd.DataFrame]:
        """
        Searches for concepts matching the given term.

        Args:
            term: The clinical term to search for.
            The number of results and filters are controlled by settings.

        Returns:
            A DataFrame containing the matching concepts, or None if no matches are found.
        """
        # Remove substrings from term
        cleaned_term = term
        for substring in self._sorted_substrings_to_remove:
            cleaned_term = re.sub(re.escape(substring), "", cleaned_term, flags=re.IGNORECASE)
        cleaned_term = cleaned_term.strip()

        vectors_with_usage = get_embedding_vectors([cleaned_term])
        self.cost = self.cost + vectors_with_usage["usage"]["total_cost_usd"]
        vector = vectors_with_usage["embeddings"][0]
        results = self._search_pgvector(vector)
        if not results:
            return None
        df = pd.DataFrame(
            results,
            columns=[
                "concept_id",
                "concept_name",
                "score",
            ],
        )
        return df

    def search_terms(
            self,
            df: pd.DataFrame,
            term_column: str,
            matched_concept_id_column: str = "matched_concept_id",
            matched_concept_name_column: str = "matched_concept_name",
            match_score_column: str = "match_score",
            match_rank_column: str = "match_rank",
            return_embeddings: bool = False,
    ):
        """
        Searches for concepts matching terms in a DataFrame column.

        Args:
            df: DataFrame containing the terms to search for.
            term_column: Name of the column with terms to search.
            matched_concept_id_column: Name of the column to store matched concept IDs.
            matched_concept_name_column: Name of the column to store matched concept names.
            match_score_column: Name of the column to store match scores.
            match_rank_column: Name of the column to store match ranks.
            return_embeddings: When True, also return a ``dict[term -> np.ndarray]``
                mapping each unique source term to its embedding vector.  The
                caller can pass these vectors to
                ``find_attributes_two_stage(..., precomputed_embedding=...)`` to
                skip the duplicate reference-retrieval embedding call in Step 2.

        Returns:
            When *return_embeddings* is False (default): a DataFrame containing
            the same columns as the input dataframe plus the matching concepts
            for each term (multiple rows per input term).

            When *return_embeddings* is True: a ``(results_df, term_to_vector)``
            tuple where ``term_to_vector`` is ``dict[str, np.ndarray]``.
        """

        terms = df[term_column].tolist()
        # Remove substrings from all terms
        cleaned_terms = []
        for term in terms:
            cleaned_term = term
            for substring in self._sorted_substrings_to_remove:
                cleaned_term = cleaned_term.replace(substring, "")
            cleaned_terms.append(cleaned_term.strip())

        vectors_with_usage = get_embedding_vectors(cleaned_terms)
        self.cost = self.cost + vectors_with_usage["usage"]["total_cost_usd"]
        vectors = vectors_with_usage["embeddings"]

        df = df.reset_index(drop=True)
        all_results = []
        term_to_vector: dict[str, np.ndarray] = {}
        for index, row in df.iterrows():
            term = row[term_column]
            vector = vectors[index]
            if return_embeddings:
                term_to_vector[term] = vector
            results = self._search_pgvector(vector)
            results = pd.DataFrame(
                results,
                columns=[
                    matched_concept_id_column,
                    matched_concept_name_column,
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
        if return_embeddings:
            return all_results, term_to_vector
        return all_results

    def get_total_cost(self) -> float:
        """
        Returns the total cost incurred for embedding vector calls

        Returns:
            Total cost in USD.
        """

        return self.cost


if __name__ == "__main__":
    concept_searcher = PgvectorConceptSearcher(settings=PgvectorSearchSettings())
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

    print(f"Total cost incurred: ${concept_searcher.get_total_cost():.6f} USD")

    concept_searcher.close()
