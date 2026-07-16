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
# either 'and' or 'or' when it appears in a title."  Exceptions are enumerated
# below and applied before the rewrite.
# ---------------------------------------------------------------------------

# Vocabularies subject to the ICD 'and = and/or' convention
_AND_OR_VOCABULARIES: frozenset[str] = frozenset({
    "ICD10CM", "ICD10", "ICD9CM", "CIM10", "KCD7", "EDI", "ICD10GM", "ICD10CN",
})

# Exact concept codes that are TRUE_AND (both components must co-occur)
_AND_OR_EXCLUDED_CODES: frozenset[str] = frozenset({
    "N70.13",   # Chronic salpingitis and oophoritis
    "N70.93",   # Salpingitis and oophoritis, unspecified
    "J35.03",   # Chronic tonsillitis and adenoiditis
    "474.02",   # ICD9CM: Chronic tonsillitis and adenoiditis
})

# Concept code prefixes that are TRUE_AND (SIMILAR TO patterns from SQL)
_AND_OR_EXCLUDED_PREFIXES: tuple[str, ...] = (
    "N83.33",   # Acquired atrophy of ovary and fallopian tube
    "N83.51",   # Torsion of ovary and ovarian ligament
    "N83.53",   # Torsion of ovary, ovarian pedicle and fallopian tube
    "L76.",     # Intraoperative/postprocedural hemorrhage and hematoma
)

# Name substring patterns that signal TRUE_AND (case-insensitive)
_AND_OR_EXCLUDED_NAME_RE: re.Pattern = re.compile(
    r"heart and .*(kidney|renal)"
    r"|calculus .* gallbladder and bile"
    r"|heatstroke and .*sunstroke",
    flags=re.IGNORECASE,
)


def _should_replace_and(term: str, vocabulary_id: str, concept_code: str) -> bool:
    """
    Return True if every ' and ' in *term* should be rewritten to ' and/or '.

    Replicates the SQL logic:
      - vocabulary_id must be in _AND_OR_VOCABULARIES
      - concept_code must NOT be in excluded exact codes or have an excluded prefix
      - term must NOT contain 'both'/'combined' (unless code starts with H26.06)
      - term must NOT match any TRUE_AND name pattern
    """
    if vocabulary_id not in _AND_OR_VOCABULARIES:
        return False
    if not re.search(r" and ", term, flags=re.IGNORECASE):
        return False
    if concept_code in _AND_OR_EXCLUDED_CODES:
        return False
    if any(concept_code.startswith(p) for p in _AND_OR_EXCLUDED_PREFIXES):
        return False
    if re.search(r"\b(both|combined)\b", term, flags=re.IGNORECASE):
        if not concept_code.startswith("H26.06"):
            return False
    if _AND_OR_EXCLUDED_NAME_RE.search(term):
        return False
    return True


class TermCleaner:
    """
    A class to clean clinical terms by removing non-essential modifiers and information using a Large Language Model (LLM).
    """

    def __init__(self, settings: TermCleanerSettings):
        self.system_prompt = settings.system_prompt
        self.cost = 0.0
        self._cost_lock = threading.Lock()

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
        if _should_replace_and(term, vocabulary_id, concept_code):
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
        vocabulary_column: str = "vocabulary_id",
        code_column: str = "concept_code",
    ) -> pd.DataFrame:
        """
        Cleans clinical terms in a DataFrame column.

        When *vocabulary_column* and *code_column* are present in *df*, they are
        used for the and/or rewrite (pass 1).  If absent the rewrite is skipped
        for all rows and only LLM cleanup (pass 2) runs — preserving full
        backward compatibility with callers that don't supply those columns.

        Args:
            df: DataFrame containing the terms to be cleaned.
            term_column: Column with source terms.
            output_column: Column to write cleaned terms to.
            vocabulary_column: Column with vocabulary_id (optional).
            code_column: Column with concept_code (optional).

        Returns:
            DataFrame with cleaned terms in *output_column*.
        """
        has_vocab = vocabulary_column in df.columns
        has_code = code_column in df.columns

        terms = df[term_column].astype(str).tolist()
        if has_vocab and has_code:
            vocab_values = df[vocabulary_column].fillna("").astype(str).tolist()
            code_values = df[code_column].fillna("").astype(str).tolist()
            terms = [
                self.rewrite_and(term, vocabulary_id, concept_code)
                for term, vocabulary_id, concept_code in zip(terms, vocab_values, code_values)
            ]

        for start in range(0, len(terms), _BATCH_SIZE):
            batch_indices = df.index[start : start + _BATCH_SIZE]
            batch_terms = terms[start : start + _BATCH_SIZE]
            cleaned_batch = self._clean_terms_batch(batch_terms)
            df.loc[batch_indices, output_column] = cleaned_batch

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
        "term": [
            "Acute myocardial infarction, unspecified",
            "Chronic kidney disease without hypertension",
            "Diabetes mellitus type 2, nos",
        ]
    }
    df = pd.DataFrame(data)
    cleaned_df = term_cleaner.clean_terms(df, "term", "cleaned_term")
    print(cleaned_df)
    print(f"Total LLM cost: ${term_cleaner.get_total_cost():.6f}")
