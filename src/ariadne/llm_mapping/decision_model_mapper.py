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

"""
Decision-model pre-step for exact matching.

Asks a System One decision model (e.g. TypeSafe's Jev) a single ``choice`` question per source term,
with the candidate concepts plus a ``no_match`` option. Confident decisions are final; otherwise
unlikely candidates are pruned before the (slower, more expensive) :class:`LlmMapper` runs.
"""

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

from ariadne.utils.settings import ConceptContextSettings, DecisionModelMapperSettings

load_dotenv()

logger = logging.getLogger(__name__)

NO_MATCH_OPTION = "no_match"
QUESTION_ID = "mapping"
API_KEY_ENV_VAR = "TYPESAFE_API_KEY"
_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


class DecisionModelError(Exception):
    """Raised when the decision model call fails or returns an unusable answer."""


class DecisionModelMapper:
    """
    Maps source terms to candidate concepts using a decision model, as a cheap pre-step for :class:`LlmMapper`.

    For each source term:

    - If one option (a candidate or ``no_match``) has probability > ``match_threshold``, it is the final mapping
      (``no_match`` maps to -1).
    - Otherwise candidates with probability < ``prune_threshold`` are removed. If no candidates remain the term
      maps to -1; else the remaining candidates are returned for the LLM mapper.
    - If the decision model fails, the term is passed to the LLM mapper with all its candidates.

    Responses are cached on disk per source ID; thresholds are applied after loading, so they can be tuned
    without new API calls.
    """

    def __init__(self, settings: DecisionModelMapperSettings, context_settings: ConceptContextSettings):
        """
        Args:
            settings: Decision model settings (``decision_model_mapping`` in the config).
            context_settings: Which target context to include (normally ``llm_mapping.context``).
        """
        self.settings = settings
        self.context_settings = context_settings
        self.responses_folder = settings.responses_folder
        os.makedirs(self.responses_folder, exist_ok=True)
        self._input_tokens = 0
        self._output_tokens = 0

    def _context_columns(
        self,
        concept_class_id_column: Optional[str],
        parents_column: Optional[str],
        domain_id_column: Optional[str],
        vocabulary_id_column: Optional[str],
        children_column: Optional[str],
        synonyms_column: Optional[str],
        clinical_drug_form_child_count_column: Optional[str],
    ) -> List[Tuple[str, str]]:
        """Returns (input column, criteria field) pairs, mirroring the context given to the LLM mapper."""
        cs = self.context_settings
        candidates = [
            (cs.include_target_class, concept_class_id_column, "class"),
            (cs.include_target_parents, parents_column, "parents"),
            (cs.include_target_domain, domain_id_column, "domain"),
            (cs.include_target_vocabulary, vocabulary_id_column, "vocabulary"),
            (cs.include_target_children, children_column, "children"),
            (cs.include_target_synonyms, synonyms_column, "synonyms"),
            (
                cs.include_target_clinical_drug_form_child_count,
                clinical_drug_form_child_count_column,
                "clinical_drug_form_child_count",
            ),
        ]
        return [(column, name) for include, column, name in candidates if include and column]

    @staticmethod
    def _to_plain(value: Any) -> Any:
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return value
        return value

    def build_request(
        self,
        source_term: str,
        target_concepts: pd.DataFrame,
        source_context: Optional[Mapping[str, Any]] = None,
        concept_id_column: str = "matched_concept_id",
        concept_name_column: str = "matched_concept_name",
        context_columns: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Builds the System One request body for one source term."""
        state = {"source_term": source_term, **dict(source_context or {})}
        criteria: Dict[str, Any] = {}
        for _, row in target_concepts.drop_duplicates(subset=[concept_id_column]).iterrows():
            option: Dict[str, Any] = {"name": str(row[concept_name_column])}
            for column, name in context_columns or []:
                if column not in row.index:
                    continue
                value = self._to_plain(row[column])
                if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
                    continue
                option[name] = value
            criteria[str(int(row[concept_id_column]))] = option
        criteria[NO_MATCH_OPTION] = self.settings.no_match_description
        return {
            "model": self.settings.model,
            "state": state,
            "questions": {
                QUESTION_ID: {
                    "type": "choice",
                    "instructions": self.settings.instructions,
                    "criteria": criteria,
                }
            },
        }

    def _call_api(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        api_key = os.getenv(API_KEY_ENV_VAR)
        if not api_key:
            raise EnvironmentError(f"Environment variable {API_KEY_ENV_VAR} is not set.")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if self.settings.long_context:
            headers["TypeSafe-Danger-Zone-Long-Context"] = "true"

        last_error: Optional[Exception] = None
        for attempt in range(self.settings.max_retries + 1):
            if attempt > 0:
                time.sleep(2 ** (attempt - 1))
            try:
                response = requests.post(
                    self.settings.api_url,
                    headers=headers,
                    json=request_body,
                    timeout=self.settings.timeout_seconds,
                )
            except (requests.ConnectionError, requests.Timeout) as e:
                last_error = e
                continue
            if response.status_code in _RETRY_STATUS_CODES:
                last_error = DecisionModelError(f"HTTP {response.status_code}: {response.text[:500]}")
                continue
            if response.status_code != 200:
                raise DecisionModelError(f"HTTP {response.status_code}: {response.text[:500]}")
            try:
                return response.json()
            except ValueError as e:
                raise DecisionModelError(f"Response is not valid JSON: {e}") from e
        raise DecisionModelError(f"Decision model call failed after {self.settings.max_retries + 1} attempts: {last_error}")

    @staticmethod
    def parse_answer(response: Mapping[str, Any], option_keys: List[str]) -> Tuple[Dict[str, float], Optional[float]]:
        """
        Extracts per-option probabilities and confidence from a System One response.

        Raises:
            DecisionModelError: If the answer is missing or does not cover all options.
        """
        try:
            answer = response["answers"][QUESTION_ID]
            probabilities = {str(k): float(v) for k, v in answer["probabilities"].items()}
        except (KeyError, TypeError, ValueError) as e:
            raise DecisionModelError(f"Unexpected decision model response: {e}") from e
        missing = [key for key in option_keys if key not in probabilities]
        if missing:
            raise DecisionModelError(f"Decision model response is missing options: {missing}")
        confidence = answer.get("confidence")
        return {key: probabilities[key] for key in option_keys}, (float(confidence) if confidence is not None else None)

    def score_term(
        self,
        source_term: str,
        source_id: str,
        target_concepts: pd.DataFrame,
        source_context: Optional[Mapping[str, Any]] = None,
        concept_id_column: str = "matched_concept_id",
        concept_name_column: str = "matched_concept_name",
        context_columns: Optional[List[Tuple[str, str]]] = None,
    ) -> Tuple[Dict[str, float], Optional[float]]:
        """
        Gets option probabilities for one source term, from cache or the API.

        Returns:
            (probabilities keyed by concept ID string or ``no_match``, confidence)

        Raises:
            DecisionModelError: If the call fails or the answer is unusable. Failures are not cached.
        """
        request_body = self.build_request(
            source_term=source_term,
            target_concepts=target_concepts,
            source_context=source_context,
            concept_id_column=concept_id_column,
            concept_name_column=concept_name_column,
            context_columns=context_columns,
        )
        option_keys = list(request_body["questions"][QUESTION_ID]["criteria"].keys())
        response_file = os.path.join(self.responses_folder, f"response_{source_id}.json")
        if os.path.exists(response_file):
            with open(response_file, "r", encoding="utf-8") as f:
                response = json.load(f)
            return self.parse_answer(response, option_keys)

        try:
            response = self._call_api(request_body)
        except requests.RequestException as e:
            raise DecisionModelError(str(e)) from e
        result = self.parse_answer(response, option_keys)
        usage = response.get("usage") or {}
        self._input_tokens += int(usage.get("input_tokens") or 0)
        self._output_tokens += int(usage.get("output_tokens") or 0)

        with open(response_file, "w", encoding="utf-8") as f:
            json.dump(response, f, indent=2)
        request_file = os.path.join(self.responses_folder, f"request_{source_id}.json")
        with open(request_file, "w", encoding="utf-8") as f:
            json.dump(request_body, f, indent=2, ensure_ascii=False)
        return result

    def _iterate_groups(
        self,
        source_target_concepts: pd.DataFrame,
        term_column: str,
        source_id_column: Optional[str],
        source_context_columns: List[str],
        source_ids: Optional[List[str]],
    ):
        grouping_column = term_column
        if source_id_column:
            if source_id_column not in source_target_concepts.columns:
                raise ValueError(f"source_id_column '{source_id_column}' is not present in input data.")
            grouping_column = source_id_column
        for _, group in source_target_concepts.groupby(grouping_column, sort=False):
            term = group.iloc[0][term_column]
            if source_id_column:
                source_id = str(group.iloc[0][source_id_column])
                if source_ids is not None and source_id not in source_ids:
                    continue
            else:
                source_id = hashlib.md5(str(term).encode("utf-8")).hexdigest()[:12]
            source_context: Dict[str, Any] = {}
            source_context_output: Dict[str, Any] = {}
            for column in source_context_columns:
                if column not in group.columns:
                    raise ValueError(f"source_context_columns column '{column}' is not present in input data.")
                value = self._to_plain(group.iloc[0][column])
                source_context_output[column] = value
                if not pd.isna(value):
                    source_context[column] = value
            yield term, source_id, group, source_context, source_context_output

    def score_terms(
        self,
        source_target_concepts: pd.DataFrame,
        term_column: str = "cleaned_term",
        source_id_column: Optional[str] = "source_code",
        source_context_columns: Optional[List[str]] = None,
        concept_id_column: str = "matched_concept_id",
        concept_name_column: str = "matched_concept_name",
        domain_id_column: Optional[str] = "matched_domain_id",
        concept_class_id_column: Optional[str] = "matched_concept_class_id",
        vocabulary_id_column: Optional[str] = "matched_vocabulary_id",
        parents_column: Optional[str] = "matched_parents",
        children_column: Optional[str] = "matched_children",
        synonyms_column: Optional[str] = "matched_synonyms",
        clinical_drug_form_child_count_column: Optional[str] = "matched_clinical_drug_form_child_count",
        source_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Returns the decision model probabilities for every source term and option, without applying thresholds.
        Useful for threshold tuning (see :func:`sweep_decision_thresholds`). Terms whose call fails are skipped.

        Returns:
            A long DataFrame with columns ``source_id`` (str), ``concept_id`` (int, -1 for no match),
            ``probability`` and ``confidence``.
        """
        context_columns = self._context_columns(
            concept_class_id_column, parents_column, domain_id_column, vocabulary_id_column,
            children_column, synonyms_column, clinical_drug_form_child_count_column,
        )
        rows = []
        for term, source_id, group, source_context, _ in self._iterate_groups(
            source_target_concepts, term_column, source_id_column, source_context_columns or [], source_ids
        ):
            try:
                probabilities, confidence = self.score_term(
                    term, source_id, group, source_context, concept_id_column, concept_name_column, context_columns
                )
            except DecisionModelError as e:
                logger.warning("Decision model failed for source '%s': %s", source_id, e)
                continue
            for key, probability in probabilities.items():
                rows.append(
                    {
                        "source_id": source_id,
                        "concept_id": -1 if key == NO_MATCH_OPTION else int(key),
                        "probability": probability,
                        "confidence": confidence,
                    }
                )
        return pd.DataFrame(rows, columns=["source_id", "concept_id", "probability", "confidence"])

    def map_terms(
        self,
        source_target_concepts: pd.DataFrame,
        term_column: str = "cleaned_term",
        source_id_column: Optional[str] = "source_code",
        source_term_column: Optional[str] = "source_term",
        source_context_columns: Optional[List[str]] = None,
        concept_id_column: str = "matched_concept_id",
        concept_name_column: str = "matched_concept_name",
        domain_id_column: Optional[str] = "matched_domain_id",
        concept_class_id_column: Optional[str] = "matched_concept_class_id",
        vocabulary_id_column: Optional[str] = "matched_vocabulary_id",
        parents_column: Optional[str] = "matched_parents",
        children_column: Optional[str] = "matched_children",
        synonyms_column: Optional[str] = "matched_synonyms",
        clinical_drug_form_child_count_column: Optional[str] = "matched_clinical_drug_form_child_count",
        mapped_concept_id_column: str = "mapped_concept_id",
        mapped_concept_name_column: str = "mapped_concept_name",
        mapped_rationale_column: str = "mapped_rationale",
        source_ids: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Maps source terms using the decision model. Takes the same input as :meth:`LlmMapper.map_terms`
        (one row per source term and candidate concept). Single-target mapping only.

        Returns:
            A tuple ``(mapped, remaining)``:

            - ``mapped``: final mappings, in the same format as :meth:`LlmMapper.map_terms` output
              (``mapped_concept_id`` is -1 for no match).
            - ``remaining``: the input rows for terms that still need the LLM mapper, restricted to candidates
              with probability >= ``prune_threshold`` (all candidates if the decision model failed).
        """
        context_columns = self._context_columns(
            concept_class_id_column, parents_column, domain_id_column, vocabulary_id_column,
            children_column, synonyms_column, clinical_drug_form_child_count_column,
        )
        match_threshold = self.settings.match_threshold
        prune_threshold = self.settings.prune_threshold
        mapped_data = []
        remaining_groups = []

        for term, source_id, group, source_context, source_context_output in self._iterate_groups(
            source_target_concepts, term_column, source_id_column, source_context_columns or [], source_ids
        ):
            try:
                probabilities, confidence = self.score_term(
                    term, source_id, group, source_context, concept_id_column, concept_name_column, context_columns
                )
            except DecisionModelError as e:
                logger.warning("Decision model failed for source '%s'; passing all candidates to LLM: %s", source_id, e)
                remaining_groups.append(group)
                continue

            confidence_text = f"{confidence:.2f}" if confidence is not None else "n/a"
            best_key = max(probabilities, key=probabilities.get)
            best_p = probabilities[best_key]
            if best_p > match_threshold:
                if best_key == NO_MATCH_OPTION:
                    mapped_id, mapped_name = -1, "no_match"
                else:
                    mapped_id = int(best_key)
                    matched = group[group[concept_id_column].astype(int) == mapped_id]
                    mapped_name = str(matched.iloc[0][concept_name_column])
                rationale = f"Decision model p={best_p:.2f} (confidence {confidence_text})"
            else:
                keep_ids = {
                    int(key) for key, p in probabilities.items() if key != NO_MATCH_OPTION and p >= prune_threshold
                }
                if keep_ids:
                    remaining_groups.append(group[group[concept_id_column].astype(int).isin(keep_ids)])
                    continue
                mapped_id, mapped_name = -1, "no_match"
                rationale = (
                    f"Decision model: all candidates p<{prune_threshold} "
                    f"(no_match p={probabilities[NO_MATCH_OPTION]:.2f}, confidence {confidence_text})"
                )

            row = {term_column: term}
            if source_id_column:
                row[source_id_column] = group.iloc[0][source_id_column]
            if source_term_column and source_term_column in group.columns:
                row[source_term_column] = group.iloc[0][source_term_column]
            row.update(source_context_output)
            row.update(
                {
                    mapped_concept_id_column: mapped_id,
                    mapped_concept_name_column: mapped_name,
                    mapped_rationale_column: rationale,
                }
            )
            mapped_data.append(row)

        remaining = (
            pd.concat(remaining_groups) if remaining_groups else source_target_concepts.iloc[0:0].copy()
        )
        return pd.DataFrame(mapped_data), remaining

    def get_total_tokens(self) -> Tuple[int, int]:
        """Returns (input_tokens, output_tokens) used by non-cached calls."""
        return self._input_tokens, self._output_tokens

    def get_total_cost(self) -> float:
        """Returns the total cost in USD of non-cached calls, based on the configured token prices."""
        return (
            self._input_tokens * self.settings.input_cost_per_million
            + self._output_tokens * self.settings.output_cost_per_million
        ) / 1_000_000


if __name__ == "__main__":
    from ariadne.utils.config import Config

    config = Config()
    mapper = DecisionModelMapper(config.decision_model_mapping, config.llm_mapping.context)
    candidates = pd.DataFrame(
        {
            "source_code": ["test1"] * 3,
            "cleaned_term": ["Acute myocardial infarction"] * 3,
            "source_term": ["Acute myocardial infarction"] * 3,
            "matched_concept_id": [1, 2, 3],
            "matched_concept_name": ["Heart attack", "Myocardial infarction", "Liver disease"],
        }
    )
    mapped, remaining = mapper.map_terms(candidates)
    print(mapped)
    print(remaining)
    print(f"Total decision model cost: ${mapper.get_total_cost():.6f}")
