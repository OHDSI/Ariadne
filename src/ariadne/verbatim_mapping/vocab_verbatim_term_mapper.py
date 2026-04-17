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

import pandas as pd
from typing import List

from ariadne.utils.settings import VerbatimMappingSettings
from ariadne.verbatim_mapping.term_normalizer import TermNormalizer


class VocabVerbatimTermMapper:
    """
    Maps source terms to concept IDs using a pre-built index of normalized terms.
    The index is created from vocabulary term files stored in Parquet format, downloaded using the download_terms
    module.

    1. If an index file exists at the verbatim_mapping_index_file path specified in the settings, it is loaded.
    2. If not, the index is created by processing all Parquet files in the terms folder specified in the settings.
    """

    def __init__(self, settings: VerbatimMappingSettings):
        self.term_normalizer = TermNormalizer(settings.substrings_to_remove)
        self.preferred_vocabulary_ids = settings.preferred_vocabulary_ids
        self._vocabulary_rank = {
            vocabulary_id: idx for idx, vocabulary_id in enumerate(self.preferred_vocabulary_ids)
        }
        if os.path.exists(settings.verbatim_mapping_index_file):
            with open(settings.verbatim_mapping_index_file, "rb") as handle:
                self.index = pickle.load(handle)
            print(f"Index loaded from {settings.verbatim_mapping_index_file}")
        else:
            self._create_index(settings)

    def _create_index(self, settings: VerbatimMappingSettings):
        print("Creating index")
        if not os.path.exists(settings.terms_folder):
            raise FileNotFoundError(
                f"Terms folder {settings.terms_folder} does not exist. Make sure to run the download_terms module first."
            )
        all_files = [
            os.path.join(settings.terms_folder, f)
            for f in os.listdir(settings.terms_folder)
            if f.endswith(".parquet")
        ]
        index_data = {}
        for file in all_files:
            print(f"Processing file: {file}")
            df = pd.read_parquet(file)
            normalized_terms = self.term_normalizer.normalize_terms(df["term"].tolist())
            for norm_term, concept_id, concept_name, vocabulary_id in zip(
                normalized_terms,
                df["concept_id"].tolist(),
                df["concept_name"].tolist(),
                df["vocabulary_id"].tolist(),
            ):
                concept = {
                    "concept_id": int(concept_id),
                    "concept_name": concept_name,
                    "vocabulary_id": vocabulary_id,
                }
                if norm_term in index_data:
                    existing = index_data[norm_term]
                    if isinstance(existing, list):
                        if concept["concept_id"] not in [c["concept_id"] for c in existing]:
                            existing.append(concept)
                    else:
                        if concept["concept_id"] != existing["concept_id"]:
                            index_data[norm_term] = [existing, concept]
                else:
                    index_data[norm_term] = concept

        self.index = index_data

        try:
            with open(settings.verbatim_mapping_index_file, "wb") as f:
                pickle.dump(index_data, f)
            print(f"Index saved to {settings.verbatim_mapping_index_file}")
        except OSError as e:
            print(f"Error saving index: {e}")

    def map_term(self, source_term: str) -> List[tuple[int, str]]:
        """
        Maps a source term to concept IDs using the pre-built index.

        Args:
            source_term: the source clinical term to map

        Returns:
            A list of concept ID - concept name tuples, possibly empty if no match is found.
        """
        normalized_source = self.term_normalizer.normalize_term(source_term)
        if normalized_source in self.index:
            concepts = self.index[normalized_source]
            candidates = concepts if isinstance(concepts, list) else [concepts]

            if self.preferred_vocabulary_ids and len(candidates) > 1:
                max_rank = len(self.preferred_vocabulary_ids)
                preferred = min(
                    candidates,
                    key=lambda c: self._vocabulary_rank.get(c["vocabulary_id"], max_rank),
                )
                return [(preferred["concept_id"], preferred["concept_name"])]

            return [(c["concept_id"], c["concept_name"]) for c in candidates]
        return []

    def map_terms(
        self,
        source_terms: pd.DataFrame,
        term_column: str = "cleaned_term",
        mapped_concept_id_column: str = "mapped_concept_id",
        mapped_concept_name_column: str = "mapped_concept_name",
    ) -> pd.DataFrame:
        """
        Maps source terms in a DataFrame column to concept IDs using the pre-built index.

        Args:
            source_terms: DataFrame containing the source clinical terms to map
            term_column: Name of the column with terms to map
            mapped_concept_id_column: Name of the column to store matched concept IDs.
            mapped_concept_name_column: Name of the column to store matched concept names.

        Returns:
            A DataFrame with the original columns and their mapped concept IDs and names.
        """
        def _pick_first_match(term: str) -> pd.Series:
            concepts = self.map_term(term)
            return pd.Series(concepts[0] if concepts else (-1, ""))

        source_terms[[mapped_concept_id_column, mapped_concept_name_column]] = source_terms[term_column].apply(
            _pick_first_match
        )
        return source_terms


if __name__ == "__main__":
    from ariadne.utils.config import Config

    config = Config()
    mapper = VocabVerbatimTermMapper(settings=config.verbatim_mapping)

    concepts = mapper.map_term("Acute myocardial infarction")
    for concept in concepts:
        print(f"Mapped to concept: {concept[1]} ({concept[0]})")

    source_terms_df = pd.DataFrame({"source_term": ["Acute myocardial infarction", "Liver disorder", "Unknown term"]})
    mapped_df = mapper.map_terms(source_terms_df, term_column="source_term")
    print(mapped_df)

    # new_index = {}
    # for term, concepts in mapper.index.items():
    #     if isinstance(concepts, list):
    #         # # Remove duplicates:
    #         # new_concepts = []
    #         # seen_ids = set()
    #         # for concept in concepts:
    #         #     if concept[0] not in seen_ids:
    #         #         new_concepts.append(concept)
    #         #         seen_ids.add(concept[0])
    #         # concepts = new_concepts
    #         # if len(concepts) == 1:
    #         #     new_index[term] = concepts[0]
    #         # else:
    #         #     new_index[term] = concepts
    #         new_concepts = []
    #         for concept in concepts:
    #             concept[0] = int(concept[0])
    #             new_concepts.add(concept)
    #         new_index[term] = new_concepts
    #     else:
    #         concepts[0] = int(concepts[0])
    #         new_index[term] = concepts
    # with open("E:/temp/mapping_quality/vocab_verbatim_index.pkl", "wb") as f:
    #     pickle.dump(new_index, f)
