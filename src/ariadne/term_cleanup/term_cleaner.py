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

import hashlib
import os
import json
import pandas as pd
import re
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

from ariadne.utils.gen_ai_api import get_llm_response
from ariadne.utils.settings import TermCleanerSettings

_BATCH_SIZE = 25

_TERM_CLEANING_BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "row_number": {"type": "integer"},
                    "cleaned_term": {"type": "string"},
                },
                "required": ["row_number", "cleaned_term"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["results"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# ICD "and" → "and/or" rewrite logic
# Per ICD-10-CM Section I.A: "the word 'and' should be interpreted to mean
# either 'and' or 'or' when it appears in a title."  The vocabularies and
# exceptions this applies to are supplied via AndOrRewriteSettings (config).
# ---------------------------------------------------------------------------


class TermCleaner:
    """
    A class to clean clinical terms by removing non-essential modifiers and information using a Large Language Model (LLM).
    """

    def __init__(self, settings: TermCleanerSettings):
        self.system_prompt = settings.system_prompt
        self.cost = 0.0
        self._cost_lock = threading.Lock()

        and_or = settings.and_or_rewrite
        self._and_or_vocabularies = frozenset(and_or.vocabulary_ids)
        self._and_or_excluded_codes = frozenset(and_or.excluded_codes)
        self._and_or_excluded_prefixes = tuple(and_or.excluded_prefixes)
        self._and_or_both_combined_exception_prefixes = tuple(and_or.both_combined_exception_prefixes)
        self._and_or_excluded_name_re = (
            re.compile("|".join(and_or.excluded_name_patterns), flags=re.IGNORECASE)
            if and_or.excluded_name_patterns
            else None
        )

    def _should_replace_and(self, term: str, vocabulary_id: str, concept_code: str) -> bool:
        """
        Return True if every ' and ' in *term* should be rewritten to ' and/or '.

        Rules (data supplied via AndOrRewriteSettings):
          - vocabulary_id must be in the configured vocabularies
          - concept_code must NOT be an excluded exact code or have an excluded prefix
          - term must NOT contain 'both'/'combined' (unless code has a both/combined exception prefix)
          - term must NOT match any TRUE_AND name pattern
        """
        if vocabulary_id not in self._and_or_vocabularies:
            return False
        if not re.search(r" and ", term, flags=re.IGNORECASE):
            return False
        if concept_code in self._and_or_excluded_codes:
            return False
        if any(concept_code.startswith(p) for p in self._and_or_excluded_prefixes):
            return False
        if re.search(r"\b(both|combined)\b", term, flags=re.IGNORECASE):
            if not any(concept_code.startswith(p) for p in self._and_or_both_combined_exception_prefixes):
                return False
        if self._and_or_excluded_name_re is not None and self._and_or_excluded_name_re.search(term):
            return False
        return True

    def rewrite_and(
        self, term: str, vocabulary_id: str = "", concept_code: str = ""
    ) -> str:
        """
        Rewrite ' and ' → ' and/or ' for ICD-family terms where appropriate.
        Args:
            term: The clinical term to be cleaned.
            vocabulary_id: Source vocabulary (e.g. 'ICD10CM'). Used for
                           the and/or rewrite; safe to omit for non-ICD terms.
            concept_code: Source concept code. Used for and/or rewrite
                          exclusions; safe to omit.

        Returns:
            The cleaned clinical term.
        """
        if self._should_replace_and(term, vocabulary_id, concept_code):
            term = re.sub(r" and ", " and/or ", term, flags=re.IGNORECASE)
        return term

    def _clean_terms_batch(self, terms: list[str]) -> list[str]:
        """Cleans up to 25 terms in one LLM request and returns results in input order."""
        if not terms:
            return []

        rows = [{"row_number": i, "source_term": term} for i, term in enumerate(terms)]
        prompt = (
            "Input JSON:\n"
            f"{json.dumps({'terms': rows}, ensure_ascii=False)}"
        )
        response = get_llm_response(
            prompt=prompt,
            system_prompt=self.system_prompt,
            json_schema=_TERM_CLEANING_BATCH_SCHEMA,
            json_schema_name="term_cleaning_batch",
        )
        self.cost += response["usage"]["total_cost_usd"]
        cleaned_terms = list(terms)
        parsed = response["parsed_json"]
        if not isinstance(parsed, dict):
            raise ValueError("Term cleaning response must be a JSON object with a 'results' array.")
        results = parsed["results"]
        if not isinstance(results, list):
            raise ValueError("Term cleaning response 'results' must be a list.")
        for item in results:
            if not isinstance(item, dict):
                raise ValueError("Each term cleaning result must be an object.")
            if "row_number" not in item:
                raise ValueError("Each term cleaning result must include integer 'row_number'.")
            if "cleaned_term" not in item:
                raise ValueError("Each term cleaning result must include string 'cleaned_term'.")
            row_number = item["row_number"]
            cleaned_term = item["cleaned_term"]
            if not isinstance(row_number, int):
                raise ValueError("Each term cleaning result must include integer 'row_number'.")
            if row_number < 0 or row_number >= len(cleaned_terms):
                raise ValueError("Row number out of range in term cleaning result.")
            if not isinstance(cleaned_term, str):
                raise ValueError("Each term cleaning result must include string 'cleaned_term'.")
            cleaned_term = cleaned_term.strip()
            cleaned_terms[row_number] = cleaned_term
        return cleaned_terms

    def clean_terms(
        self,
        df: pd.DataFrame,
        term_column: str = "source_term",
        output_column: str = "cleaned_term",
        vocabulary_column: str = "source_vocabulary_id",
        code_column: str = "source_concept_code",
    ) -> pd.DataFrame:
        """
        Cleans clinical terms in a DataFrame column.

        LLM cleanup runs first, then the and/or rewrite is applied to the cleaned
        terms.  Running the rewrite last keeps it deterministic, since the LLM
        cannot collapse ' and/or ' back to ' and '.

        *term_column*, *vocabulary_column*, and *code_column* are all required and
        must be present in *df*; a ValueError is raised otherwise.

        Args:
            df: DataFrame containing the terms to be cleaned.
            term_column: Column with source terms.
            output_column: Column to write cleaned terms to.
            vocabulary_column: Column with the source vocabulary id.
            code_column: Column with the source concept code.

        Returns:
            DataFrame with cleaned terms in *output_column*.
        """
        missing_columns = [c for c in (term_column, vocabulary_column, code_column) if c not in df.columns]
        if missing_columns:
            raise ValueError(f"clean_terms requires the following missing columns: {missing_columns}")

        terms = df[term_column].astype(str).tolist()

        for start in range(0, len(terms), _BATCH_SIZE):
            batch_indices = df.index[start : start + _BATCH_SIZE]
            batch_terms = terms[start : start + _BATCH_SIZE]
            cleaned_batch = self._clean_terms_batch(batch_terms)
            df.loc[batch_indices, output_column] = cleaned_batch

        vocab_values = df[vocabulary_column].fillna("").astype(str).tolist()
        code_values = df[code_column].fillna("").astype(str).tolist()
        cleaned_values = df[output_column].astype(str).tolist()
        df[output_column] = [
            self.rewrite_and(cleaned_term, vocabulary_id, concept_code)
            for cleaned_term, vocabulary_id, concept_code in zip(cleaned_values, vocab_values, code_values)
        ]

        # Remove rows where LLM returned an empty cleaned term (sign of bad input like 'number of goats', 'declined' or 'primary')
        df = df[df[output_column].notna() & (df[output_column].str.strip() != "")].reset_index(drop=True)

        return df

    def get_total_cost(self) -> float:
        """
        Returns the total cost incurred for LLM calls during term cleaning.

        Returns:
            Total cost in USD.
        """

        return self.cost


if __name__ == "__main__":
    from ariadne.utils.config import Config

    config = Config()
    term_cleaner = TermCleaner(settings=config.term_cleaning)
    data = {
        "source_term": [
            "Acute myocardial infarction, unspecified",
            "Chronic kidney disease without hypertension",
            "Diabetes mellitus type 2, nos",
        ],
        "source_vocabulary_id": ["ICD10CM", "ICD10CM", "ICD10CM"],
        "source_concept_code": ["I21.9", "N18.9", "E11.9"],
    }
    df = pd.DataFrame(data)
    cleaned_df = term_cleaner.clean_terms(df)
    print(cleaned_df)
    print(f"Total LLM cost: ${term_cleaner.get_total_cost():.6f}")
