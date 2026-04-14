import json
import os
import re
from typing import Any, List, Mapping, Optional, Tuple

import pandas as pd

from ariadne.utils.settings import LlmMapperSettings
from ariadne.utils.gen_ai_api import get_llm_response


_FINAL_MAPPING_SCHEMA = {
    "type": "object",
    "properties": {
        "source_term": {"type": "string"},
        "match_found": {"type": "boolean"},
        "concept_id": {"type": ["integer", "null"]},
        "justification": {"type": "string"},
    },
    "required": ["source_term", "match_found", "concept_id", "justification"],
    "additionalProperties": False,
}

_FINAL_MAPPING_MULTI_SCHEMA = {
    "type": "object",
    "properties": {
        "source_term": {"type": "string"},
        "match_found": {"type": "boolean"},
        "concept_ids": {
            "type": "array",
            "items": {"type": "integer"},
        },
        "justification": {"type": "string"},
    },
    "required": ["source_term", "match_found", "concept_ids", "justification"],
    "additionalProperties": False,
}


class LlmMapper:
    def __init__(self, settings: LlmMapperSettings):
        self.system_prompts = settings.system_prompts
        self.context_settings = settings.context
        self.responses_folder = settings.llm_mapper_responses_folder
        os.makedirs(self.responses_folder, exist_ok=True)
        self._cost = 0.0
        """
        Initializes the LlmMapper with configuration settings, specific system prompts, and context settings for 
        LLM-based term mapping. Also sets up a folder to store LLM responses.
        """

    @staticmethod
    def _extract_json_dict(response_text: str) -> dict | None:
        """Extracts and parses a JSON object from a response string."""
        response_json_match = re.search(r"{.*}", response_text, flags=re.DOTALL)
        if not response_json_match:
            return None
        try:
            return json.loads(response_json_match.group(0))
        except json.JSONDecodeError:
            return None

    def map_term(
        self,
        source_term: str,
        source_id: Optional[str],
        target_concepts: pd.DataFrame,
        source_context: Optional[Mapping[str, Any]] = None,
        concept_id_column: str = "matched_concept_id",
        concept_name_column: str = "matched_concept_name",
        domain_id_column: Optional[str] = "matched_domain_id",
        concept_class_id_column: Optional[str] = "matched_concept_class_id",
        vocabulary_id_column: Optional[str] = "matched_vocabulary_id",
        parents_column: Optional[str] = "matched_parents",
        children_column: Optional[str] = "matched_children",
        synonyms_column: Optional[str] = "matched_synonyms",
        allow_multiple_targets: bool = False,
    ) -> Tuple[int | List[int] | None, str | List[str] | None, str | None]:
        """
        Maps a source term to the matching target concept using LLM prompts. The LLM can be prompted in multiple
        steps. The first step provides the source term and candidate target concepts as prompt, with information
        specified in config.llm_mapping.context. Subsequent steps use the response from the previous step as prompt,
        unless config.llm_mapping.context.re_insert_target_details is set to True, in which case the target concept
        details are re-inserted into the response JSON for the next step.

        Finally, the response is processed to extract the matched concept ID and name, looking for a line starting with
        "Match: <concept_id>" or "Match: no_match".

        Args:
            source_term: The source clinical term to map.
            source_id: An optional unique identifier for the source term, used for caching responses.
            target_concepts: A DataFrame containing candidate target concepts with columns:
            source_context: Optional additional source details (column_name -> value) to add to prompts.
            concept_id_column: The name of the column containing target concept IDs.
            concept_name_column: The name of the column containing target concept names.
            domain_id_column: The name of the column containing target domain IDs.
            concept_class_id_column: The name of the column containing target concept class IDs.
            vocabulary_id_column: The name of the column containing target vocabulary IDs.
            parents_column: The name of the column containing target concept parents.
            children_column: The name of the column containing target concept children.
            synonyms_column: The name of the column containing target concept synonyms.

        Returns:
            A tuple of (matched_concept_id, matched_concept_name, match_rationale).
            - When allow_multiple_targets=False, matched_concept_id/name are scalar values.
            - When allow_multiple_targets=True and matches are found, matched_concept_id/name are lists.
            If no match is found, returns (-1, "no_match", ""). If the content filter is hit, returns
            (None, None, None).
        """

        num_prompts = len(self.system_prompts)
        if source_id is None:
            source_id = abs(hash(source_term)) % (10**8)

        source_context_payload = dict(source_context or {})
        source_details = {"source_term": source_term, **source_context_payload}

        input_columns = [concept_id_column, concept_name_column]
        context_columns = ["concept_id", "concept_name"]
        if self.context_settings.include_target_class:
            input_columns.append(concept_class_id_column)
            context_columns.append("concept_class_id")
        if self.context_settings.include_target_parents:
            input_columns.append(parents_column)
            context_columns.append("concept_parents")
        if self.context_settings.include_target_domain:
            input_columns.append(domain_id_column)
            context_columns.append("concept_domain")
        if self.context_settings.include_target_vocabulary:
            input_columns.append(vocabulary_id_column)
            context_columns.append("concept_vocabulary")
        if self.context_settings.include_target_children:
            input_columns.append(children_column)
            context_columns.append("concept_children")
        if self.context_settings.include_target_synonyms:
            input_columns.append(synonyms_column)
            context_columns.append("concept_synonyms")
        context = target_concepts[input_columns]
        context.columns = context_columns

        prompt = ""
        for step in range(num_prompts):
            response_file = os.path.join(self.responses_folder, f"response_{source_id}_s{step + 1}.txt")

            # Load response from file if it exists:
            if os.path.exists(response_file):
                with open(response_file, "r", encoding="utf-8") as f:
                    response = f.read()
                if response == "*Content filter triggered*":
                    return None, None, None
            else:
                # Else generate a new response from the LLM:
                system_prompt = self.system_prompts[step]
                if step == 0:
                    context_json = context.to_json(orient="records", lines=True)
                    if source_context_payload:
                        source_details_json = json.dumps(source_details, ensure_ascii=False, indent=2)
                        prompt = f"Source details:\n{source_details_json}\n\nCandidate target concepts:\n{context_json}"
                    else:
                        prompt = f"Source term: {source_term}\n\nCandidate target concepts:\n{context_json}"

                use_final_structured_output = step == num_prompts - 1
                response_with_usage = get_llm_response(
                    prompt,
                    system_prompt,
                    json_schema=(
                        _FINAL_MAPPING_MULTI_SCHEMA
                        if use_final_structured_output and allow_multiple_targets
                        else _FINAL_MAPPING_SCHEMA if use_final_structured_output else None
                    ),
                )
                response = response_with_usage["content"]
                if not response:
                    # We hit the content filter:
                    with open(response_file, "w", encoding="utf-8") as f:
                        f.write("*Content filter triggered*")
                    return None, None, None
                self._cost = self._cost + response_with_usage["usage"]["total_cost_usd"]

                if step == 0 and num_prompts > 1 and self.context_settings.re_insert_source_target_details:
                    try:
                        data = self._extract_json_dict(response)
                        if data:
                            new_source_data = data.get("source_term", source_term)
                            new_source_data.update(source_context_payload)

                            target_definitions = pd.DataFrame(data["target_concepts"])
                            target_definitions["id"] = pd.to_numeric(target_definitions["id"], errors="coerce")
                            merged = pd.merge(
                                target_definitions, context, left_on="id", right_on="concept_id", how="left"
                            )
                            merged = merged.drop(columns=["concept_id"])
                            new_target_data = merged.to_dict(orient="records")

                            new_data: dict[str, Any] = {
                                "source_term": new_source_data,
                                "target_concepts": new_target_data
                            }
                            response = json.dumps(new_data, indent=2)
                    except Exception as e:
                        print(f"Warning: Could not re-insert source/target details: {e}")

                with open(response_file, "w", encoding="utf-8") as f:
                    f.write(response)
            if step < num_prompts - 1:
                # Use the response as the prompt for the next step:
                prompt = response

        # Process the final response to extract the match:
        response = response.replace("**", "")
        parsed = self._extract_json_dict(response)
        if parsed and "match_found" in parsed and "justification" in parsed:
            justification = str(parsed["justification"])
            if not parsed["match_found"]:
                return -1, "no_match", justification

            raw_match_values: List[int] = []
            if "concept_ids" in parsed and isinstance(parsed["concept_ids"], list):
                for value in parsed["concept_ids"]:
                    try:
                        raw_match_values.append(int(value))
                    except (TypeError, ValueError):
                        raise ValueError(f"Match value '{value}' is not a valid integer.")
            elif "concept_id" in parsed:
                try:
                    raw_match_values = [int(parsed["concept_id"])]
                except (TypeError, ValueError):
                    raise ValueError(f"Match value '{parsed['concept_id']}' is not a valid integer.")
            else:
                raise ValueError("Could not find concept_id or concept_ids in LLM response.")

            # In multi-target mode, ignore no-match sentinel values if valid targets are present.
            if allow_multiple_targets and any(value != -1 for value in raw_match_values):
                raw_match_values = [value for value in raw_match_values if value != -1]

            deduplicated_values: List[int] = []
            for value in raw_match_values:
                if value not in deduplicated_values:
                    deduplicated_values.append(value)

            if not deduplicated_values or deduplicated_values == [-1]:
                return -1, "no_match", justification

            matched_names: List[str] = []
            for match_value_int in deduplicated_values:
                matched_row = target_concepts[target_concepts[concept_id_column] == match_value_int]
                if matched_row.empty:
                    raise ValueError(f"Match '{match_value_int}' not found in search results.")
                matched_names.append(str(matched_row.iloc[0][concept_name_column]))

            if allow_multiple_targets:
                return deduplicated_values, matched_names, justification
            return deduplicated_values[0], matched_names[0], justification

        match = re.findall(r"^#* ?Match ?:.*", response, flags=re.MULTILINE | re.IGNORECASE)
        if match:
            # Parse legacy format:
            if re.search("no[ _]match|-1", match[-1], re.IGNORECASE):
                match_value_int = -1
                concept_name = "no_match"
            else:
                number_match = re.findall(r"\d+", match[-1])
                if not number_match:
                    raise ValueError(f"No numeric match found in response: {response}")
                match_values = [int(value) for value in number_match]
                if allow_multiple_targets and any(value != -1 for value in match_values):
                    match_values = [value for value in match_values if value != -1]
                if not match_values:
                    match_values = [-1]

                if match_values == [-1]:
                    match_value_int = -1
                    concept_name = "no_match"
                elif allow_multiple_targets:
                    deduplicated_values: List[int] = []
                    for value in match_values:
                        if value not in deduplicated_values:
                            deduplicated_values.append(value)
                    matched_names: List[str] = []
                    for value in deduplicated_values:
                        matched_row = target_concepts[target_concepts[concept_id_column] == value]
                        if matched_row.empty:
                            raise ValueError(f"Match '{value}' not found in search results.")
                        matched_names.append(str(matched_row.iloc[0][concept_name_column]))
                    match_value_int = deduplicated_values
                    concept_name = matched_names
                else:
                    number_match_value = match_values[0]
                    matched_row = target_concepts[target_concepts[concept_id_column] == number_match_value]
                    if matched_row.empty:
                        raise ValueError(f"Match '{number_match_value}' not found in search results.")
                    match_value_int = number_match_value
                    concept_name = str(matched_row.iloc[0][concept_name_column])
            # Extract the rationale if provided.
            rationale_match = re.search(r"Justification[:\-]?(.*)", response, flags=re.DOTALL | re.IGNORECASE)
            rationale = ""
            if rationale_match:
                rationale = rationale_match.group(1).strip()
                rationale = rationale.replace("\n", " ").replace("\\n", "\n")

            return match_value_int, concept_name, rationale
        raise ValueError("Could not parse match from LLM response.")

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
        mapped_concept_id_column: str = "mapped_concept_id",
        mapped_concept_name_column: str = "mapped_concept_name",
        mapped_rationale_column: str = "mapped_rationale",
        source_ids: List[str] | None = None,
        allow_multiple_targets: bool = False,
    ) -> pd.DataFrame:
        """
        Maps source terms in a DataFrame column to target concepts using LLM prompts. The system prompts are taken
        from the configuration file. Multiple steps are supported as per the map_term method.

        The input DataFrame should contain multiple rows per source term, one for each candidate target concept.

        Be aware that LLM responses are cached based on source term and source ID, so if the same term appears
        multiple times with the same source ID, the cached response will be used. The cache is stored in the
        llm_mapper_responses_folder specified in the config.

        Args:
            source_target_concepts: DataFrame containing the source clinical terms and candidate target concepts.
            term_column: The name of the column containing source terms fed to the LLM.
            source_id_column: The name of the column containing the unique source term IDs.
            source_term_column: The name of the column containing the original source terms.
            source_context_columns: Optional list of source-side columns to include in prompts and output rows.
            concept_id_column: The name of the column containing the target concept IDs.
            concept_name_column: The name of the column containing the target concept names.
            domain_id_column: The name of the column containing the target domain IDs.
            concept_class_id_column: The name of the column containing the target concept class IDs.
            vocabulary_id_column: The name of the column containing the target vocabulary IDs.
            parents_column: The name of the column containing the target concept parents.
            children_column: The name of the column containing the target concept children.
            synonyms_column: The name of the column containing the target concept synonyms.
            mapped_concept_id_column: The name of the output column for mapped concept IDs.
            mapped_concept_name_column: The name of the output column for mapped concept names.
            mapped_rationale_column: The name of the output column for mapping rationale.
            source_ids: (Optional): A list of source IDs to restrict to.
        Returns:
            A DataFrame with the original terms and their mapped concept IDs and names.
        """

        mapped_data = []
        source_context_columns = source_context_columns or []
        grouped = source_target_concepts.groupby(term_column)
        for term, group in grouped:
            source_id = None
            if source_id_column and source_id_column in group.columns:
                source_id = str(group.iloc[0][source_id_column])
                if source_ids is not None and source_id not in source_ids:
                    continue

            source_context: dict[str, Any] = {}
            source_context_output: dict[str, Any] = {}
            for column in source_context_columns:
                if column not in group.columns:
                    raise ValueError(f"source_context_columns column '{column}' is not present in input data.")
                value = group.iloc[0][column]
                if hasattr(value, "item"):
                    try:
                        value = value.item()
                    except Exception:
                        pass
                source_context_output[column] = value
                if pd.isna(value):
                    continue
                source_context[column] = value

            matched_concept_id, matched_concept_name, match_rationale = self.map_term(
                source_term=term,
                source_id=source_id,
                target_concepts=group,
                source_context=source_context,
                concept_id_column=concept_id_column,
                concept_name_column=concept_name_column,
                domain_id_column=domain_id_column,
                concept_class_id_column=concept_class_id_column,
                vocabulary_id_column=vocabulary_id_column,
                parents_column=parents_column,
                children_column=children_column,
                synonyms_column=synonyms_column,
                allow_multiple_targets=allow_multiple_targets,
            )
            if matched_concept_id is None:
                # Content filter was hit:
                continue

            concept_ids = matched_concept_id if isinstance(matched_concept_id, list) else [matched_concept_id]
            concept_names = matched_concept_name if isinstance(matched_concept_name, list) else [matched_concept_name]

            for idx, mapped_id in enumerate(concept_ids):
                mapped_name = concept_names[idx] if idx < len(concept_names) else None
                mapped_data.append(
                    {
                        term_column: term,
                        source_id_column: source_id,
                        source_term_column: group.iloc[0][source_term_column],
                        **source_context_output,
                        mapped_concept_id_column: mapped_id,
                        mapped_concept_name_column: mapped_name,
                        mapped_rationale_column: match_rationale,
                    }
                )
        return pd.DataFrame(mapped_data)

    def get_total_cost(self) -> float:
        """
        Returns the total cost incurred for LLM calls

        Returns:
            Total cost in USD.
        """

        return self._cost


if __name__ == "__main__":
    from ariadne.utils.config import Config

    config = Config()
    mapper = LlmMapper(settings=config.llm_mapping)

    source_term = "Acute myocardial infarction"
    target_concepts = pd.DataFrame(
        {
            "matched_concept_id": [1, 2, 3],
            "matched_concept_name": ["Heart attack", "Myocardial infarction", "Liver disease"],
            "matched_domain_id": ["Condition", "Condition", "Condition"],
            "matched_concept_class_id": ["Clinical Finding", "Clinical Finding", "Clinical Finding"],
            "matched_vocabulary_id": ["SNOMED", "SNOMED", "SNOMED"],
            "matched_parents": ["Cardiovascular disease", "Cardiovascular disease", "Digestive disease"],
            "matched_children": ["Acute coronary syndrome", "Chronic ischemic heart disease", "Hepatitis"],
            "matched_synonyms": ["AMI; Heart attack", "MI; Myocardial infarction", "Liver disorder"],
        }
    )
    mapped_id, mapped_name, rationale = mapper.map_term(
        source_term,
        source_id="test1",
        target_concepts=target_concepts,
    )
    print(f"Source term '{source_term}' mapped to concept: {mapped_name} ({mapped_id}) with rationale: {rationale}")
    print(f"Total LLM cost: ${mapper.get_total_cost():.4f}")
