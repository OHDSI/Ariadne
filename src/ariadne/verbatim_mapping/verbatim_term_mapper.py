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


import pandas as pd

from typing import List, Union

from ariadne.utils.config import Config
from ariadne.verbatim_mapping.term_normalizer import TermNormalizer


class VerbatimTermMapper:
    """
    Maps a source term to a provided subset of target concepts based on exact matches of normalized terms.
    """

    def __init__(self, config: Config = Config()):
        self.term_normalizer = TermNormalizer(config)

    def map_term(
        self,
        source_term: str,
        target_concept_ids: List[int],
        target_terms: List[str],
        target_synonyms: List[str],
    ) -> (Union[int, None], Union[str, None]):
        """
        Maps a source term to the best matching target concept ID based on normalized terms.

        Args:
            source_term: the source clinical term to map
            target_concept_ids: a list of target concept IDs
            target_terms: a list of target clinical terms
            target_synonyms: a list of target synonyms. Each string is semicolon separated synonyms for the
            corresponding target term.

        Returns:
            A tuple of (mapped_concept_id, mapped_term) if a match is found, otherwise (None, None)
        """
        normalized_source = self.term_normalizer.normalize_term(source_term)
        for concept_id, term, synonyms in zip(
            target_concept_ids, target_terms, target_synonyms
        ):
            normalized_term = self.term_normalizer.normalize_term(term)
            if normalized_source == normalized_term:
                return concept_id, term
            if not pd.isna(synonyms):
                for synonym in synonyms.split(";"):
                    normalized_synonym = self.term_normalizer.normalize_term(synonym)
                    if normalized_source == normalized_synonym:
                        return concept_id, term
        return None, None


if __name__ == "__main__":
    mapper = VerbatimTermMapper()

    # Example mapping
    source = "Acute myocardial infarction"
    target_ids = [1001, 1002, 1003]
    target_terms = ["Liver disorder", "Kidney disorder", "Heart disease"]
    target_synonyms = [
        "Hepatic disorder; Liver diseases",
        "Renal disorder; Kidney diseases",
        "Cardiac disease; Heart conditions (disorder)",
    ]
    mapped_id, mapped_term = mapper.map_term(
        source, target_ids, target_terms, target_synonyms
    )
    print(f"Source term '{source}' mapped to concept: {mapped_term} ({mapped_id})")
