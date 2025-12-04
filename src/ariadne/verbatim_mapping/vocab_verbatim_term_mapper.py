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
from dataclasses import dataclass

import pandas as pd
from typing import Set, List, Optional

from ariadne.utils.config import Config
from ariadne.verbatim_mapping.term_normalizer import TermNormalizer


@dataclass(slots=True)
class Concept:
    concept_id: int
    concept_name: str


class VocabVerbatimTermMapper:
    """
    Maps source terms to concept IDs using a pre-built index of normalized terms.
    The index is created from vocabulary term files stored in Parquet format, downloaded using the download_terms
    module.
    1. If an index file exists at the verbatim_mapping_index_file path specified in the config, it is loaded.
    2. If not, the index is created by processing all Parquet files in the terms folder specified in the config.
    """

    def __init__(self, config: Config = Config()):
        self.term_normalizer = TermNormalizer()
        if os.path.exists(config.verbatim_mapping_index_file):
            with open(config.verbatim_mapping_index_file, "rb") as handle:
                self.index = pickle.load(handle)
            print(f"Index loaded from {config.verbatim_mapping_index_file}")
        else:
            self._create_index(config)

    def _create_index(self, config: Config):
        print("Creating index")
        if not os.path.exists(config.terms_folder):
            raise FileNotFoundError(
                f"Terms folder {config.terms_folder} does not exist. Make sure to run the download_terms module first."
            )
        all_files = [
            os.path.join(config.terms_folder, f)
            for f in os.listdir(config.terms_folder)
            if f.endswith(".parquet")
        ]
        pool = multiprocessing.get_context("spawn").Pool(processes=config.max_cores)
        index_data = {}
        for file in all_files:
            print(f"Processing file: {file}")
            df = pd.read_parquet(file)
            normalized_terms = pool.map(
                self.term_normalizer.normalize_term, df["term"].tolist()
            )
            for norm_term, concept_id, concept_name in zip(
                normalized_terms, df["concept_id"].tolist(), df["concept_name"].tolist()
            ):
                concept = Concept(concept_id, concept_name)
                if norm_term in index_data:
                    existing = index_data[norm_term]
                    if isinstance(existing, list):
                        if concept_id not in existing:
                            existing.append(concept)
                    else:
                        if concept_id != existing:
                            index_data[norm_term] = [existing, concept]
                else:
                    index_data[norm_term] = concept

        pool.close()
        self.index = index_data

        try:
            with open(config.verbatim_mapping_index_file, "wb") as f:
                pickle.dump(index_data, f)
            print(f"Index saved to {config.verbatim_mapping_index_file}")
        except OSError as e:
            print(f"Error saving index: {e}")

    def map_term(self, source_term: str) -> List[Concept]:
        """
        Maps a source term to concept IDs using the pre-built index.
        :param source_term: the source clinical term to map
        :return: a list of matching concepts, possibly empty if no match is found.
        """
        normalized_source = self.term_normalizer.normalize_term(source_term)
        if normalized_source in self.index:
            concept_ids = self.index[normalized_source]
            if isinstance(concept_ids, list):
                return concept_ids
            else:
                return [concept_ids]
        return []


if __name__ == "__main__":
    mapper = VocabVerbatimTermMapper()

    concept = mapper.map_term("Hepatic disorder")
    for c in concept:
        print(f"Mapped to concept: {c.concept_name} ({c.concept_id})")

    # terms = pd.read_csv("E:/temp/mapping_quality/ICD10CMterms.csv")
    # mapped_count = 0
    # unmapped_count = 0
    # for term in terms["concept_name"].tolist():
    #     mapped_ids = mapper.map_term_using_index(term)
    #     if mapped_ids:
    #         mapped_count += 1
    #     else:
    #         unmapped_count += 1
    # print(f"Mapped terms: {mapped_count}, Unmapped terms: {unmapped_count}")
