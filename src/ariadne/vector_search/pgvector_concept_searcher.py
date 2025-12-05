# # Copyright 2025 Observational Health Data Sciences and Informatics
# #
# # This file is part of Ariadne
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
#
# import os
# import pickle
#
# import numpy as np
# import pandas as pd
# from ariadne.utils.utils import get_environment_variable
# from ariadne.vector_search.abstract_concept_searcher import AbstractConceptSearcher
#
# pd.set_option('display.max_columns', None)
#
# from typing import List, Dict
#
# from dataclasses import dataclass
#
# import psycopg
# from pgvector.psycopg import register_vector
# from dotenv import load_dotenv
#
# load_dotenv()
#
#
# class ConceptSearcher(AbstractConceptSearcher):
#
#     def __init__(self, for_evaluation: bool = False, include_synonyms: bool = True, include_mapped_terms: bool = False):
#         self.for_evaluation = for_evaluation
#         self.include_synonyms = include_synonyms
#         self.include_mapped_terms = include_mapped_terms
#
#         if for_evaluation:
#             self.concept_classes_to_ignore=["Disposition", "Morph Abnormality", "Organism", "Qualifier Value", "Substance", "ICDO Condition"]
#             self.vocabularies_to_ignore=["ICD9CM", "ICD10CM", "ICD10", "ICD10CN", "ICD10GM", "CIM10", "ICDO3", "KCD7", "Read"]
#             print("ConceptSearcher initialized in evaluation mode.")
#         else:
#             self.concept_classes_to_ignore=None
#             self.vocabularies_to_ignore=None
#
#         connection = psycopg.connect(os.getenv("vocab_connection_string").replace("+psycopg", ""))
#         register_vector(connection)
#         with connection.cursor() as cur:
#             cur.execute("SET hnsw.ef_search = 1000")
#             cur.execute("SET hnsw.iterative_scan = relaxed_order")
#         self.connection = connection
#
#
#
#
#     def search(self, term: str, limit: int = 25) -> Optional[pd.DataFrame]:
#         """
#         Searches for concepts matching the given term.
#
#         Args:
#             term: The clinical term to search for.
#             limit: The maximum number of results to return.
#
#         Returns:
#             A DataFrame containing the matching concepts, or None if no matches are found.
#         """
#
#
#
#     def _search_pgvector(self, source_vector: np.ndarray, limit: int) -> List:
#
#         if self.target_classes_to_ignore is None:
#             ignore_string_class = "'dummy'"
#         else:
#             ignore_string_class = ", ".join(f"'{y}'" for y in self.target_classes_to_ignore)
#
#         if self.include_synonyms:
#             term_type_clause = ""
#         else:
#              term_type_clause = "AND vectors.term_type = 'Name'"
#
#         vocabulary_schema = get_environment_variable("VOCABULARY_SCHEMA")
#
#         if self.include_mapped_terms:
#             if self.source_vocabs_to_ignore is None:
#                 ignore_string_vocab = "'dummy'"
#             else:
#                 ignore_string_vocab = ", ".join(f"'{x}'" for x in self.source_vocabs_to_ignore)
#             query = f"""
#                 WITH target_concept AS (
#                     SELECT concept_id,
#                         concept_name,
#                         domain_id,
#                         concept_class_id,
#                         vocabulary_id,
#                         MIN(relevance_score) AS relevance_score
#                     FROM (
#                         (
#                             SELECT concept.concept_id,
#                                 concept.concept_name,
#                                 concept.domain_id,
#                                 concept.concept_class_id,
#                                 concept.vocabulary_id,
#                                 embedding_vector <=> %s AS relevance_score
#                             FROM {vocabulary_schema}.{self.settings.vector_table} vectors
#                             INNER JOIN {vocabulary_schema}.concept source_concept
#                                 ON vectors.concept_id = source_concept.concept_id
#                             INNER JOIN {vocabulary_schema}.concept_relationship
#                                 ON vectors.concept_id = concept_relationship.concept_id_1
#                             INNER JOIN {vocabulary_schema}.concept
#                                 ON concept_relationship.concept_id_2 = concept.concept_id
#                             WHERE relationship_id = 'Maps to'
#                                 AND source_concept.vocabulary_id NOT IN ({ignore_string_vocab})
#                                 AND concept.concept_class_id NOT IN ({ignore_string_class})
#                                 {term_type_clause}
#                             ORDER BY embedding_vector <=> %s
#                             LIMIT {limit * 4} -- May have duplicates due to synonyms
#                         )
#
#                         UNION ALL
#
#                         (
#                             SELECT concept.concept_id,
#                                 concept.concept_name,
#                                 concept.domain_id,
#                                 concept.concept_class_id,
#                                 concept.vocabulary_id,
#                                 embedding_vector <=> %s AS relevance_score
#                             FROM {vocabulary_schema}.{self.settings.vector_table} vectors
#                             INNER JOIN {vocabulary_schema}.concept
#                                 ON vectors.concept_id = concept.concept_id
#                             WHERE standard_concept = 'S'
#                                 AND concept.concept_class_id NOT IN ({ignore_string_class})
#                                 {term_type_clause}
#                             ORDER BY embedding_vector <=> %s
#                             LIMIT {limit * 4} -- May have duplicates due to synonyms
#                         )
#                     ) tmp
#                     GROUP BY concept_id,
#                         concept_name,
#                         domain_id,
#                         concept_class_id,
#                         vocabulary_id
#                 )
#                 SELECT target_concept.concept_id,
#                     target_concept.concept_name,
#                     target_concept.domain_id,
#                     target_concept.concept_class_id,
#                     target_concept.vocabulary_id,
#                     COALESCE(parent_names.parent_names, '') AS parent_names,
#                     COALESCE(child_names.child_names, '') AS child_names,
#                     COALESCE(synonyms.synonyms, '') AS synonyms,
#                     target_concept.relevance_score
#                 FROM target_concept
#                 LEFT JOIN (
#                     SELECT target_concept.concept_id,
#                         string_agg(parent_concept.concept_name, '; ' ORDER BY parent_concept.concept_name) AS parent_names
#                     FROM target_concept
#                     INNER JOIN {vocabulary_schema}.concept_ancestor
#                         ON target_concept.concept_id = concept_ancestor.descendant_concept_id
#                         AND concept_ancestor.min_levels_of_separation = 1
#                     INNER JOIN {vocabulary_schema}.concept AS parent_concept
#                         ON concept_ancestor.ancestor_concept_id = parent_concept.concept_id
#                     GROUP BY target_concept.concept_id
#                 ) AS parent_names
#                     ON target_concept.concept_id = parent_names.concept_id
#                 LEFT JOIN (
#                     SELECT concept_id,
#                         string_agg(concept_name, '; ' ORDER BY concept_name) AS child_names
#                     FROM (
#                         SELECT target_concept.concept_id,
#                             child_concept.concept_name
#                         FROM target_concept
#                         INNER JOIN {vocabulary_schema}.concept_ancestor
#                             ON target_concept.concept_id = concept_ancestor.ancestor_concept_id
#                             AND concept_ancestor.min_levels_of_separation = 1
#                         INNER JOIN {vocabulary_schema}.concept AS child_concept
#                             ON concept_ancestor.descendant_concept_id = child_concept.concept_id
#                         ORDER BY RANDOM()
#                         LIMIT 10 -- Prevent excessive number of children
#                     ) AS limited_children
#                     GROUP BY concept_id
#                 ) AS child_names
#                     ON target_concept.concept_id = child_names.concept_id
#                 LEFT JOIN (
#                     SELECT target_concept.concept_id,
#                         string_agg(concept_synonym_name, '; ' ORDER BY concept_synonym_name) AS synonyms
#                     FROM target_concept
#                     INNER JOIN {vocabulary_schema}.concept_synonym
#                         ON target_concept.concept_id = concept_synonym.concept_id
#                     WHERE concept_synonym.language_concept_id = 4180186 -- English
#                     GROUP BY target_concept.concept_id
#                 ) AS synonyms
#                     ON target_concept.concept_id = synonyms.concept_id
#                 ORDER BY relevance_score
#                 LIMIT {limit};
#             """
#             with self.connection.cursor() as cur:
#                 cur.execute(query, (source_vector, source_vector, source_vector, source_vector))
#                 results = cur.fetchall()
#         else:
#             query = f"""
#                 WITH target_concept AS (
#                     SELECT concept_id,
#                         concept_name,
#                         domain_id,
#                         concept_class_id,
#                         vocabulary_id,
#                         MIN(relevance_score) AS relevance_score
#                     FROM (
#                         SELECT concept.concept_id,
#                             concept.concept_name,
#                             concept.domain_id,
#                             concept.concept_class_id,
#                             concept.vocabulary_id,
#                             embedding_vector <=> %s AS relevance_score
#                         FROM {vocabulary_schema}.{self.settings.vector_table} vectors
#                         INNER JOIN {vocabulary_schema}.concept
#                             ON vectors.concept_id = concept.concept_id
#                         WHERE standard_concept = 'S'
#                             AND concept.concept_class_id NOT IN ({ignore_string_class})
#                             {term_type_clause}
#                         ORDER BY embedding_vector <=> %s
#                         LIMIT {limit * 4} -- May have duplicates due to synonyms
#                     ) tmp
#                     GROUP BY concept_id,
#                         concept_name,
#                         domain_id,
#                         concept_class_id,
#                         vocabulary_id
#                 )
#                 SELECT target_concept.concept_id,
#                     target_concept.concept_name,
#                     target_concept.domain_id,
#                     target_concept.concept_class_id,
#                     target_concept.vocabulary_id,
#                     COALESCE(parent_names.parent_names, '') AS parent_names,
#                     COALESCE(child_names.child_names, '') AS child_names,
#                     COALESCE(synonyms.synonyms, '') AS synonyms,
#                     target_concept.relevance_score
#                 FROM target_concept
#                 LEFT JOIN (
#                     SELECT target_concept.concept_id,
#                         string_agg(parent_concept.concept_name, '; ' ORDER BY parent_concept.concept_name) AS parent_names
#                     FROM target_concept
#                     INNER JOIN {vocabulary_schema}.concept_ancestor
#                         ON target_concept.concept_id = concept_ancestor.descendant_concept_id
#                         AND concept_ancestor.min_levels_of_separation = 1
#                     INNER JOIN {vocabulary_schema}.concept AS parent_concept
#                         ON concept_ancestor.ancestor_concept_id = parent_concept.concept_id
#                     GROUP BY target_concept.concept_id
#                 ) AS parent_names
#                     ON target_concept.concept_id = parent_names.concept_id
#                 LEFT JOIN (
#                     SELECT concept_id,
#                         string_agg(concept_name, '; ' ORDER BY concept_name) AS child_names
#                     FROM (
#                         SELECT target_concept.concept_id,
#                             child_concept.concept_name
#                         FROM target_concept
#                         INNER JOIN {vocabulary_schema}.concept_ancestor
#                             ON target_concept.concept_id = concept_ancestor.ancestor_concept_id
#                             AND concept_ancestor.min_levels_of_separation = 1
#                         INNER JOIN {vocabulary_schema}.concept AS child_concept
#                             ON concept_ancestor.descendant_concept_id = child_concept.concept_id
#                         ORDER BY RANDOM()
#                         LIMIT 10 -- Prevent excessive number of children
#                     ) AS limited_children
#                     GROUP BY concept_id
#                 ) AS child_names
#                     ON target_concept.concept_id = child_names.concept_id
#                 LEFT JOIN (
#                     SELECT target_concept.concept_id,
#                         string_agg(concept_synonym_name, '; ' ORDER BY concept_synonym_name) AS synonyms
#                     FROM target_concept
#                     INNER JOIN {vocabulary_schema}.concept_synonym
#                         ON target_concept.concept_id = concept_synonym.concept_id
#                     WHERE concept_synonym.language_concept_id = 4180186 -- English
#                     GROUP BY target_concept.concept_id
#                 ) AS synonyms
#                     ON target_concept.concept_id = synonyms.concept_id
#                 ORDER BY relevance_score
#                 LIMIT {limit};
#             """
#             with self.connection.cursor() as cur:
#                 cur.execute(query, (source_vector, source_vector))
#                 results = cur.fetchall()
#
#         return results
#
#     def find_concepts(self, vectors: Dict) -> pd.DataFrame:
#         rows = []
#         length = len(vectors["texts"])
#
#         with psycopg.connect(os.getenv("vocab_connection_string").replace("+psycopg", "")) as self.connection:
#             register_vector(conn)
#             with conn.cursor() as cur:
#                 cur.execute("SET hnsw.ef_search = 1000")
#                 cur.execute("SET hnsw.iterative_scan = relaxed_order")
#
#             for i in range(length):
#                 source_name = vectors["texts"][i]
#                 source_id = vectors["ids"][i]
#                 source_vector = vectors["vectors"][i]
#                 if self.settings.source_concept_ids is not None and source_id not in self.settings.source_concept_ids:
#                     continue
#                 result = self._find_concepts(source_vector, conn)
#                 result = pd.DataFrame(result, columns=["concept_id_2",
#                                                        "concept_name_2",
#                                                        "domain_id_2",
#                                                        "concept_class_id_2",
#                                                        "vocabulary_id_2",
#                                                        "parent_names_2",
#                                                        "child_names_2",
#                                                        "synonyms_2",
#                                                        "score"])
#                 result.insert(0, "concept_id_1", source_id)
#                 result.insert(1, "concept_name_1", source_name)
#                 rows.append(result)
#
#         df = pd.concat(rows, ignore_index=True)
#
#         # Remove maps of terms to themselves
#         mask = df["concept_id_1"] == df["concept_id_2"]
#         df = df[~mask]
#
#         # add filter for verbatim string
#         # df = (
#         #     df.groupby('concept_id_1', group_keys=False)
#         #     .apply(lambda g: g[g['concept_name_1'] == g['concept_name_2']] if (
#         #                 g['concept_name_1'] == g['concept_name_2']).any() else g)
#         #     .reset_index(drop=True)
#         # )
#         # Make sure they are integers
#         df["concept_id_1"] = df["concept_id_1"].astype(int)
#         df["concept_id_2"] = df["concept_id_2"].astype(int)
#
#         # Add rank
#         df["rank"] = df.groupby("concept_id_1")["score"].rank(method="min", ascending=True).astype(int)
#         return df
#
#
#     def process_file(self, source_vector_file: str, results_file: str) -> None:
#         with open(source_vector_file, "rb") as f:
#             vectors = pickle.load(f)
#
#         result_df = self.find_concepts(vectors)
#         result_df.to_csv(results_file, index=False)
#         print(f"Results saved to {results_file}")
#
#
# if __name__ == "__main__":
#     settings = ConceptSearcherSettings(
#         k=25,
#         vocab_schema="vocabulary_feb2025",
#         vector_table="concept_vector_eval_sep_syns",
#         include_source_terms = False,
#         include_synonyms=True,
#         target_classes_to_ignore = ["Disposition", "Morph Abnormality", "Organism", "Qualifier Value", "Substance", "ICDO Condition"],
#         source_vocabs_to_ignore = ["ICD9CM", "ICD10CM", "ICD10", "ICD10CN", "ICD10GM",
#                                    "CIM10", "ICDO3", "KCD7", "Read"]
#     )
#     concept_searcher = ConceptSearcher(settings)
#     concept_searcher.process_file(
#         source_vector_file="./files/source_terms_not_unspecified.pkl",
#         results_file="./files/vector_search_results_synonyms_not_unspecified.csv"
#     )
