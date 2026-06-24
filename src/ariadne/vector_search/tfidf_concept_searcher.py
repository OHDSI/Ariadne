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
import re
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ariadne.utils.settings import TfidfSearchSettings
from ariadne.vector_search.abstract_concept_searcher import AbstractConceptSearcher
from ariadne.verbatim_mapping.term_downloader import download_terms_for_tfidf


class TfidfConceptSearcher(AbstractConceptSearcher):
    """Concept searcher based on a word-level TF-IDF index built from downloaded terms."""

    def __init__(self, settings: TfidfSearchSettings):
        self.settings = settings
        self._sorted_substrings_to_remove = sorted(settings.substrings_to_remove, key=len, reverse=True)
        self.vectorizer: TfidfVectorizer
        self.term_matrix = None
        self.term_metadata = pd.DataFrame()

        if os.path.exists(self.settings.tfidf_index_file):
            self._load_index()
        else:
            self._create_index()

    def _load_index(self) -> None:
        with open(self.settings.tfidf_index_file, "rb") as handle:
            index_data = pickle.load(handle)
        self.vectorizer = index_data["vectorizer"]
        self.term_matrix = index_data["term_matrix"]
        self.term_metadata = index_data["term_metadata"]

    def _create_index(self) -> None:
        if not os.path.exists(self.settings.terms_folder) or not os.listdir(self.settings.terms_folder):
            download_terms_for_tfidf(settings=self.settings)

        all_files = [
            os.path.join(self.settings.terms_folder, f)
            for f in os.listdir(self.settings.terms_folder)
            if f.endswith(".parquet")
        ]
        if not all_files:
            raise FileNotFoundError(
                f"No parquet files found in {self.settings.terms_folder} after attempting download."
            )

        frames = [pd.read_parquet(file) for file in all_files]
        terms_df = pd.concat(frames, ignore_index=True)
        terms_df["normalized_term"] = terms_df["term"].fillna("").map(self._normalize_text)
        terms_df = terms_df[terms_df["normalized_term"] != ""].copy()
        terms_df["concept_id"] = terms_df["concept_id"].astype(int)

        if terms_df.empty:
            raise ValueError("Cannot build TF-IDF index because all source terms are empty after normalization.")

        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 1),
            sublinear_tf=True,
            min_df=1,
        )
        self.term_matrix = self.vectorizer.fit_transform(terms_df["normalized_term"].tolist())

        self.term_metadata = terms_df[["concept_id", "concept_name", "vocabulary_id", "normalized_term"]].reset_index(drop=True)
        os.makedirs(os.path.dirname(self.settings.tfidf_index_file), exist_ok=True)
        with open(self.settings.tfidf_index_file, "wb") as handle:
            pickle.dump(
                {
                    "vectorizer": self.vectorizer,
                    "term_matrix": self.term_matrix,
                    "term_metadata": self.term_metadata,
                },
                handle,
            )

    def _normalize_text(self, text: str) -> str:
        normalized = text.lower()
        for substring in self._sorted_substrings_to_remove:
            normalized = re.sub(re.escape(substring), " ", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def search_term(self, term: str) -> Optional[pd.DataFrame]:
        query = self._normalize_text(term)
        if not query:
            return pd.DataFrame(columns=["concept_id", "concept_name", "vocabulary_id", "score"])

        query_vector = self.vectorizer.transform([query])
        scores = (self.term_matrix @ query_vector.T).toarray().ravel()
        if scores.size == 0:
            return pd.DataFrame(columns=["concept_id", "concept_name", "vocabulary_id", "score"])

        scored = self.term_metadata.copy()
        scored["score"] = scores
        scored = scored[scored["score"] > 0].copy()

        if self.settings.filter.vocabulary_ids:
            scored = scored[scored["vocabulary_id"].isin(self.settings.filter.vocabulary_ids)]

        if scored.empty:
            return pd.DataFrame(columns=["concept_id", "concept_name", "vocabulary_id", "score"])

        scored = scored.sort_values(by=["score", "concept_id"], ascending=[False, True])
        deduped = scored.drop_duplicates(subset=["concept_id"], keep="first")
        deduped = deduped.sort_values(by=["score", "concept_id"], ascending=[False, True])
        top_matches = deduped.head(self.settings.max_candidates).reset_index(drop=True)

        return top_matches[["concept_id", "concept_name", "vocabulary_id", "score"]]

    def search_terms(
        self,
        df: pd.DataFrame,
        term_column: str,
        matched_concept_id_column: str = "matched_concept_id",
        matched_concept_name_column: str = "matched_concept_name",
        match_score_column: str = "match_score",
        match_rank_column: str = "match_rank",
    ) -> pd.DataFrame:
        all_results = []
        for _, row in df.iterrows():
            term = row[term_column]
            matches = self.search_term(term)
            if matches is None or matches.empty:
                continue

            rows = []
            for rank, (_, concept) in enumerate(matches.iterrows(), start=1):
                rows.append(
                    {
                        matched_concept_id_column: int(concept["concept_id"]),
                        matched_concept_name_column: concept["concept_name"],
                        match_score_column: float(concept["score"]),
                        match_rank_column: rank,
                    }
                )

            result_df = pd.DataFrame(rows)
            original_columns = list(df.columns)
            result_columns = list(result_df.columns)
            for col in original_columns:
                result_df[col] = row[col]
            result_df = result_df[original_columns + result_columns]
            all_results.append(result_df)

        if not all_results:
            return pd.DataFrame()

        return pd.concat(all_results, ignore_index=True)


if __name__ == "__main__":
    concept_searcher = TfidfConceptSearcher(settings=TfidfSearchSettings())
    search_results = concept_searcher.search_term("Kent Pharma UK Ltd")
    print(search_results)


