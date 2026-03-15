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

import json
import pandas as pd
import re
from ariadne.utils.gen_ai_api import get_llm_response
from ariadne.utils.config import Config


_TRIGGER_PATTERN = r"not|unspecified|unidentified|without|other| nos|,nos| nec|,nec|encounter"
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


class TermCleaner:
    """
    A class to clean clinical terms by removing non-essential modifiers and information using a Large Language Model (LLM).
    """

    def __init__(self, config: Config = Config()):
        self.system_prompt = config.term_cleaning.system_prompt
        self.cost = 0.0

    def clean_term(self, term: str) -> str:
        """
        Cleans a clinical term using an LLM to remove non-essential modifiers and information.

        Args:
            term: The clinical term to be cleaned.

        Returns:
            The cleaned clinical term.
        """

        if re.search(_TRIGGER_PATTERN, term, flags=re.IGNORECASE) is None:
            return term

        return self._clean_terms_batch([term])[0]

    def _clean_terms_batch(self, terms: list[str]) -> list[str]:
        """Cleans up to 25 terms in one LLM request and returns results in input order."""
        if not terms:
            return []

        rows = [{"row_number": i, "source_term": term} for i, term in enumerate(terms)]
        prompt = (
            "Clean each source term in the provided JSON and return one cleaned term per row_number.\n"
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
            if not isinstance(cleaned_term, str):
                raise ValueError("Each term cleaning result must include string 'cleaned_term'.")
            cleaned_term = cleaned_term.strip()
            if cleaned_term and 0 <= row_number < len(cleaned_terms):
                cleaned_terms[row_number] = cleaned_term
        return cleaned_terms

    def clean_terms(
        self, df: pd.DataFrame, term_column: str = "source_term", output_column: str = "cleaned_term"
    ) -> pd.DataFrame:
        """
        Cleans clinical terms in a DataFrame column using the LLM.

        Args:
            df: DataFrame containing the terms to be cleaned.
            term_column: Name of the column with terms to be cleaned.
            output_column: Name of the column to store cleaned terms.

        Returns:
            DataFrame with an additional column for cleaned terms.
        """

        df[output_column] = df[term_column]

        trigger_indices = [
            i
            for i, term in enumerate(df[term_column].tolist())
            if re.search(_TRIGGER_PATTERN, term, flags=re.IGNORECASE) is not None
        ]

        for start in range(0, len(trigger_indices), _BATCH_SIZE):
            batch_indices = trigger_indices[start : start + _BATCH_SIZE]
            batch_terms = [df.iloc[idx][term_column] for idx in batch_indices]
            cleaned_batch = self._clean_terms_batch(batch_terms)
            for idx, cleaned in zip(batch_indices, cleaned_batch):
                df.iat[idx, df.columns.get_loc(output_column)] = cleaned

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
