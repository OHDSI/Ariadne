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
import pandas as pd
import re
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

from ariadne.utils.gen_ai_api import get_llm_response
from ariadne.utils.config import Config


_TRIGGER_PATTERN = (
    r"not\b|unspecified|unidentified|without\b|other\b"
    r"| nos\b|,nos\b| nec\b|,nec\b|encounter"
    r"|uncomplicated|classified elsewhere|with or without|\bunknown\b"
)

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

    def __init__(self, config: Config = Config(), max_workers: int = 8):
        self.system_prompt = config.term_cleaning.system_prompt
        self.responses_folder = config.system.term_cleaner_responses_folder
        os.makedirs(self.responses_folder, exist_ok=True)
        self.cost = 0.0
        self._cost_lock = threading.Lock()
        self.max_workers = max_workers

    def clean_term(
        self, term: str, vocabulary_id: str = "", concept_code: str = ""
    ) -> str:
        """
        Cleans a clinical term in two passes:
          1. Offline: rewrite ' and ' → ' and/or ' for ICD-family terms where
             appropriate (no LLM, no network).
          2. LLM: remove non-essential modifiers (NOS, unspecified, etc.).

        Args:
            term: The clinical term to be cleaned.
            vocabulary_id: Source vocabulary (e.g. 'ICD10CM'). Used for
                           the and/or rewrite; safe to omit for non-ICD terms.
            concept_code: Source concept code. Used for and/or rewrite
                          exclusions; safe to omit.

        Returns:
            The cleaned clinical term.
        """
        # Pass 1 — and → and/or rewrite (offline, no LLM)
        if _should_replace_and(term, vocabulary_id, concept_code):
            term = re.sub(r" and ", " and/or ", term, flags=re.IGNORECASE)

        # Pass 2 — NOS/unspecified/etc. LLM cleanup
        if re.search(_TRIGGER_PATTERN, term, flags=re.IGNORECASE) is None:
            return term

        # Return cached result if available
        cache_key = hashlib.md5(term.encode()).hexdigest()
        cache_file = os.path.join(self.responses_folder, f"term_clean_{cache_key}.txt")
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as fh:
                return fh.read().strip()

        prompt = f"#Term: {term}"
        response = get_llm_response(prompt=prompt, system_prompt=self.system_prompt)
        with self._cost_lock:
            self.cost += response["usage"]["total_cost_usd"]

        # Try '#Term:' prefix (case-insensitive), fall back to stripped raw response
        content = response["content"].strip()
        match = re.search(r"#[Tt]erm:\s*(.+)$", content, flags=re.MULTILINE)
        if match:
            cleaned = match.group(1).strip()
        else:
            warnings.warn(f"Could not parse '#Term:' from response for '{term}'; using raw response")
            cleaned = content

        # Persist to cache
        with open(cache_file, "w", encoding="utf-8") as fh:
            fh.write(cleaned)

        return cleaned

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
        has_code  = code_column in df.columns

        # Deduplicate on (term, vocab, code) when available, else just term.
        # Two rows with the same term but different concept_code may produce
        # different results (one excluded from the rewrite, one not).
        if has_vocab and has_code:
            keys = (
                df[[term_column, vocabulary_column, code_column]]
                .dropna(subset=[term_column])
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )
            keys = list(keys)
            result_map: dict = {}
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.clean_term, t, str(v), str(c)): (t, v, c)
                    for t, v, c in keys
                }
                for future in as_completed(futures):
                    t, v, c = futures[future]
                    result_map[(t, v, c)] = future.result()
            df[output_column] = df.apply(
                lambda r: result_map.get(
                    (r[term_column], r[vocabulary_column], r[code_column]), r[term_column]
                ),
                axis=1,
            )
        else:
            unique_terms = list(df[term_column].dropna().unique())
            term_map: dict = {}
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self.clean_term, t): t for t in unique_terms}
                for future in as_completed(futures):
                    term_map[futures[future]] = future.result()
            df[output_column] = df[term_column].map(term_map)

        return df

    def get_total_cost(self) -> float:
        """
        Returns the total cost incurred for LLM calls during term cleaning.

        Returns:
            Total cost in USD.
        """

        return self.cost


if __name__ == "__main__":
    term_cleaner = TermCleaner()
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
