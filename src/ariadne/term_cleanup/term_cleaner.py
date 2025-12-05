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

import pickle
import pandas as pd
import re
from ariadne.utils.gen_ai_api import get_embedding_vectors, get_llm_response
from ariadne.utils.utils import get_environment_variable
from ariadne.utils.config import Config
import warnings


_TRIGGER_PATTERN = (
    r"not|unspecified|unidentified|without|other| nos|,nos| nec|,nec|encounter"
)


class TermCleaner:
    """
    A class to clean clinical terms by removing non-essential modifiers and information using a Large Language Model (LLM).
    """

    def __init__(self, config: Config = Config()):
        self.system_prompt = config.term_clean_system_prompt
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
        prompt = f"#Term: {term}"
        response = get_llm_response(prompt=prompt, system_prompt=self.system_prompt)
        self.cost += response["usage"]["total_cost_usd"]
        pattern = r"#Term: (.+)$"
        match = re.match(pattern, response["content"].strip())
        if match:
            return match.group(1)  # Returns the captured answer
        else:
            warnings.warn(f"Term {term} not found in response {response}")
            return term

    def clean_terms_in_df(
        self, df: pd.DataFrame, term_column: str, output_column: str
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

        df[output_column] = df[term_column].apply(self.clean_term)
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
    cleaned_df = term_cleaner.clean_terms_in_df(df, "term", "cleaned_term")
    print(cleaned_df)
    print(f"Total LLM cost: ${term_cleaner.get_total_cost():.6f}")
