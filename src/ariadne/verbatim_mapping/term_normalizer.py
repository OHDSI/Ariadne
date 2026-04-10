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
from typing import List

import spacy


class TermNormalizer:
    """
    Normalizes clinical term strings for high-precision matching.
    """

    def __init__(self, substrings_to_remove: List[str] | None = None):
        self.substrings_to_remove = substrings_to_remove or []
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("spaCy model 'en_core_web_sm' loaded successfully.")
        except IOError:
            print("spaCy model 'en_core_web_sm' not found.")
            print("Please run: python -m spacy download en_core_web_sm")
            raise

    def normalize_term(self, term: str) -> str:
        """
        Normalizes a clinical term string for high-precision matching.

        The pipeline is:

        1. Convert to lowercase.
        2. Remove possessive "'s" at the end of words.
        3. Remove specific non-informative substrings (e.g., '(disorder)'). The strings are taken from the
           substrings_to_remove list in the config yaml file
        4. Remove all punctuation.
        5. Tokenize and lemmatize (e.g., "disorders" -> "disorder").
        6. Join tokens into a single string, preserving order.

        This makes "liver disorders" and "Liver-Disorders (disorder)"
        both normalize to "liver disorder".

        Args:
            term: The clinical term string to normalize.
        Returns:
            The normalized term string.
        """
        # 1. Convert to lowercase
        term = term.lower()

        # 2. Remove possessive 's at the end of a word
        # This handles "Alzheimer's disease" -> "Alzheimer disease"
        # It finds a word character (\w) followed by 's and a word boundary (\b),
        # and replaces the whole thing with just the captured word character (group 1).
        term = re.sub(r"(\w)'s\b", r"\1", term)

        # 3. Remove specific non-informative substrings
        for sub in self.substrings_to_remove:
            term = term.replace(sub, ' ')

        # 4. Remove all punctuation (replace with a space)
        # This handles "liver-disorder" and "liver, disorder"
        term = re.sub(r'[^\w\s]', ' ', term)

        # 5. Tokenize and lemmatize using spaCy
        doc = self.nlp(term)

        processed_tokens = []
        for token in doc:
            # Get the lemma (base form)
            lemma = token.lemma_

            # 6. Remove empty tokens (from extra spaces)
            if lemma.strip():
                processed_tokens.append(lemma)

        # 7. Join tokens into a single string
        return " ".join(processed_tokens)

    def normalize_terms(self, terms: List[str], batch_size: int = 1000) -> List[str]:
        """
        Normalizes a list of clinical term strings in batch using spaCy's nlp.pipe for efficiency.

        Args:
            terms: List of clinical term strings to normalize.
            batch_size: Number of terms to process in each spaCy batch.
        Returns:
            List of normalized term strings in the same order as the input.
        """
        # Pre-process steps 1-4 before sending to spaCy
        preprocessed = []
        for term in terms:
            t = term.lower()
            t = re.sub(r"(\w)'s\b", r"\1", t)
            for sub in self.substrings_to_remove:
                t = t.replace(sub, ' ')
            t = re.sub(r'[^\w\s]', ' ', t)
            preprocessed.append(t)

        # Use spaCy's built-in batch processing (no multiprocessing pickle needed)
        results = []
        for doc in self.nlp.pipe(preprocessed, batch_size=batch_size):
            processed_tokens = [token.lemma_ for token in doc if token.lemma_.strip()]
            results.append(" ".join(processed_tokens))
        return results


if __name__ == "__main__":
    mapper = TermNormalizer()

    terms_to_test = [
        "Liver-Disorders",
        "Enthesopathy of bilateral feet (disorder)",
        "Prinzmetal's angina",
        "Fractures of the left leg",
        "Depression in remission",
        "Severe depression (disorder)",
        "Skin, disorder",
        "Skin disorder"
    ]
    for term in terms_to_test:
        normalized = mapper.normalize_term(term)
        print(f"Original: '{term}' -> Normalized: '{normalized}'")
