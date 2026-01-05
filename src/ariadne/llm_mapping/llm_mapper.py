import os
import re
from typing import Optional, Tuple, List
import json

import pandas as pd

from ariadne.utils.config import Config
from ariadne.utils.gen_ai_api import get_llm_response


class LlmMapper:
    def __init__(self, config: Config = Config()):
        self.system_prompts = config.llm_mapping.system_prompts
        self.context_settings = config.llm_mapping.context
        self.responses_folder = config.system.llm_mapper_responses_folder
        os.makedirs(self.responses_folder, exist_ok=True)
        self._cost = 0.0
        """
        Initializes the LlmMapper with configuration settings, specific system prompts, and context settings for 
        LLM-based term mapping. Also sets up a folder to store LLM responses.
        """

    def map_term(
        self,
        source_term: str,
        source_id: Optional[str],
        target_concepts: pd.DataFrame,
        concept_id_column: str = "matched_concept_id",
        concept_name_column: str = "matched_concept_name",
        domain_id_column: Optional[str] = "matched_domain_id",
        concept_class_id_column: Optional[str] = "matched_concept_class_id",
        vocabulary_id_column: Optional[str] = "matched_vocabulary_id",
        parents_column: Optional[str] = "matched_parents",
        children_column: Optional[str] = "matched_children",
        synonyms_column: Optional[str] = "matched_synonyms",
    ) -> Tuple[int | None, str | None, str | None]:
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
            concept_id_column: The name of the column containing target concept IDs.
            concept_name_column: The name of the column containing target concept names.
            domain_id_column: The name of the column containing target domain IDs.
            concept_class_id_column: The name of the column containing target concept class IDs.
            vocabulary_id_column: The name of the column containing target vocabulary IDs.
            parents_column: The name of the column containing target concept parents.
            children_column: The name of the column containing target concept children.
            synonyms_column: The name of the column containing target concept synonyms.

        Returns:
            A tuple of (matched_concept_id, matched_concept_name, match_rationale). If no match is found, returns
            (-1, "no_match", ""). If the content filter is hit, returns (None, None, None).
        """

        num_prompts = len(self.system_prompts)
        if source_id is None:
            source_id = abs(hash(source_term)) % (10**8)

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
                    prompt = f"Source term: {source_term}\n\nCandidate target concepts:\n{context_json}"

                response_with_usage = get_llm_response(prompt, system_prompt)
                response = response_with_usage["content"]
                if not response:
                    # We hit the content filter:
                    with open(response_file, "w", encoding="utf-8") as f:
                        f.write("*Content filter triggered*")
                    return None, None, None
                self._cost = self._cost + response_with_usage["usage"]["total_cost_usd"]

                if step == 0 and self.context_settings.re_insert_target_details:
                    # Re-insert target details into the response JSON for the next step:
                    response_json_match = re.search(r"{.*}", response, flags=re.DOTALL)
                    if response_json_match:
                        response_json_str = response_json_match.group(0)
                        try:
                            data = json.loads(response_json_str)
                            target_definitions = data["target_concepts"]
                            target_definitions = pd.DataFrame(target_definitions)
                            target_definitions["id"] = pd.to_numeric(target_definitions["id"], errors="coerce")
                            merged = pd.merge(
                                target_definitions, context, left_on="id", right_on="concept_id", how="left"
                            )
                            merged = merged.drop(columns=["concept_id"])
                            new_data = {
                                "source_term": data["source_term"],
                                "target_concepts": merged.to_dict(orient="records"),
                            }
                            response = json.dumps(new_data, indent=2)
                        except Exception as e:
                            print(f"Warning: Could not re-insert target details: {e}")

                with open(response_file, "w", encoding="utf-8") as f:
                    f.write(response)
            if step < num_prompts - 1:
                # Use the response as the prompt for the next step:
                prompt = response

        # Process the final response to extract the match:
        response = response.replace("**", "")
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
                number_match_value = number_match[0]
                try:
                    match_value_int = int(number_match_value)
                except ValueError:
                    raise ValueError(f"Match value '{number_match_value}' is not a valid integer.")
                matched_row = target_concepts[target_concepts[concept_id_column] == match_value_int]
                if matched_row.empty:
                    raise ValueError(f"Match '{number_match_value}' not found in search results.")
                concept_name = str(matched_row.iloc[0][concept_name_column])
            # Extract the rationale if provided.
            rationale_match = re.search(r"Justification[:\-]?(.*)", response, flags=re.DOTALL | re.IGNORECASE)
            rationale = ""
            if rationale_match:
                rationale = rationale_match.group(1).strip()
                rationale = rationale.replace("\n", " ").replace("\\n", "\n")

            return match_value_int, concept_name, rationale
        else:
            # Parse JSON format:
            response_json_match = re.search(r"{.*}", response, flags=re.DOTALL)
            if response_json_match:
                response_json_str = response_json_match.group(0)
                data = json.loads(response_json_str)
                justification = data["justification"]
                if not data["match_found"]:
                    return -1, "no_match", justification
                else:
                    try:
                        match_value_int = int(data["concept_id"])
                    except ValueError:
                        raise ValueError(f"Match value '{data["concept_id"]}' is not a valid integer.")
                    matched_row = target_concepts[target_concepts[concept_id_column] == match_value_int]
                    if matched_row.empty:
                        raise ValueError(f"Match '{match_value_int}' not found in search results.")
                    concept_name = str(matched_row.iloc[0][concept_name_column])
                    return match_value_int, concept_name, justification

    def map_terms(
        self,
        source_target_concepts: pd.DataFrame,
        term_column: str = "cleaned_term",
        source_id_column: Optional[str] = "source_concept_id",
        source_term_column: Optional[str] = "source_term",
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
        grouped = source_target_concepts.groupby(term_column)
        for term, group in grouped:
            source_id = None
            if source_id_column and source_id_column in group.columns:
                source_id = str(group.iloc[0][source_id_column])
                if source_ids is not None and source_id not in source_ids:
                    continue
            matched_concept_id, matched_concept_name, match_rationale = self.map_term(
                term,
                source_id,
                group,
                concept_id_column,
                concept_name_column,
                domain_id_column,
                concept_class_id_column,
                vocabulary_id_column,
                parents_column,
                children_column,
                synonyms_column,
            )
            if matched_concept_id is None:
                # Content filter was hit:
                continue
            mapped_data.append(
                {
                    term_column: term,
                    source_id_column: source_id,
                    source_term_column: group.iloc[0][source_term_column],
                    mapped_concept_id_column: matched_concept_id,
                    mapped_concept_name_column: matched_concept_name,
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
    mapper = LlmMapper()

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
