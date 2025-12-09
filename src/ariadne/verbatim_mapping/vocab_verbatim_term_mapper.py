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


import multiprocessing
import os
import pickle

import pandas as pd
from typing import Set, List, Optional

from ariadne.utils.config import Config
from ariadne.verbatim_mapping.term_normalizer import TermNormalizer


class VocabVerbatimTermMapper:
    """
    Maps source terms to concept IDs using a pre-built index of normalized terms.
    The index is created from vocabulary term files stored in Parquet format, downloaded using the download_terms
    module.

    1. If an index file exists at the verbatim_mapping_index_file path specified in the config, it is loaded.
    2. If not, the index is created by processing all Parquet files in the terms folder specified in the config.
    """

    def __init__(self, config: Config = Config()):
        self.term_normalizer = TermNormalizer(config)
        if os.path.exists(config.system.verbatim_mapping_index_file):
            with open(config.system.verbatim_mapping_index_file, "rb") as handle:
                self.index = pickle.load(handle)
            print(f"Index loaded from {config.system.verbatim_mapping_index_file}")
        else:
            self._create_index(config)

    def _create_index(self, config: Config):
        print("Creating index")
        if not os.path.exists(config.system.terms_folder):
            raise FileNotFoundError(
                f"Terms folder {config.system.terms_folder} does not exist. Make sure to run the download_terms module first."
            )
        all_files = [
            os.path.join(config.system.terms_folder, f)
            for f in os.listdir(config.system.terms_folder)
            if f.endswith(".parquet")
        ]
        pool = multiprocessing.get_context("spawn").Pool(processes=config.system.max_cores)
        index_data = {}
        for file in all_files:
            print(f"Processing file: {file}")
            df = pd.read_parquet(file)
            normalized_terms = pool.map(self.term_normalizer.normalize_term, df["term"].tolist())
            for norm_term, concept_id, concept_name in zip(
                normalized_terms, df["concept_id"].tolist(), df["concept_name"].tolist()
            ):
                concept = (int(concept_id), concept_name)
                if norm_term in index_data:
                    existing = index_data[norm_term]
                    if isinstance(existing, list):
                        if concept_id not in [c[0] for c in existing]:
                            existing.append(concept)
                    else:
                        if concept_id != existing[0]:
                            index_data[norm_term] = [existing, concept]
                else:
                    index_data[norm_term] = concept

        pool.close()
        self.index = index_data

        try:
            with open(config.system.verbatim_mapping_index_file, "wb") as f:
                pickle.dump(index_data, f)
            print(f"Index saved to {config.system.verbatim_mapping_index_file}")
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
            if isinstance(concepts, list):
                return concepts
            else:
                return [concepts]
        return []

    def map_terms(
        self,
        source_terms: pd.DataFrame,
        term_column: str,
        matched_concept_id_column: str = "matched_concept_id",
        matched_concept_name_column: str = "matched_concept_name",
    ) -> pd.DataFrame:
        """
        Maps source terms in a DataFrame column to concept IDs using the pre-built index.

        Args:
            source_terms: DataFrame containing the source clinical terms to map
            term_column: Name of the column with terms to map
            matched_concept_id_column: Name of the column to store matched concept IDs.
            matched_concept_name_column: Name of the column to store matched concept names.

        Returns:
            A DataFrame with the original columns and their mapped concept IDs and names.
        """
        source_terms[[matched_concept_id_column, matched_concept_name_column]] = source_terms[term_column].apply(
            lambda term: pd.Series(self.map_term(term)[0] if self.map_term(term) else (-1, ""))
        )
        return source_terms

        # mapped_data = []
        # for term in source_terms[term_column]:
        #     concepts = self.map_term(term)
        #     if concepts:
        #         for concept in concepts:
        #             mapped_data.append({
        #                 term_column: term,
        #                 matched_concept_id_column: concept[0],
        #                 matched_concept_name_column: concept[1]
        #             })
        #     else:
        #         mapped_data.append({
        #             term_column: term,
        #             matched_concept_id_column: -1,
        #             matched_concept_name_column: ""
        #         })
        # return pd.DataFrame(mapped_data)


if __name__ == "__main__":
    mapper = VocabVerbatimTermMapper()

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
