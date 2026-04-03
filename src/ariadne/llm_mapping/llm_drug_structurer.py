import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ariadne.utils.config import Config
from ariadne.utils.gen_ai_api import get_llm_response
from ariadne.utils.utils import get_project_root


_BATCH_SIZE = 25

_CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "row_number": {"type": "integer"},
                    "category": {"type": "string", "enum": ["drug", "device", "other"]},
                },
                "required": ["row_number", "category"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}

_INGREDIENT_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "row_number": {"type": "integer"},
                    "ingredient_name": {"type": ["string", "null"]},
                    "ingredient_code": {"type": ["string", "null"]},
                    "amount_value": {"type": ["number", "null"]},
                    "amount_unit": {"type": ["string", "null"]},
                    "numerator_value": {"type": ["number", "null"]},
                    "numerator_unit": {"type": ["string", "null"]},
                    "denominator_value": {"type": ["number", "null"]},
                    "denominator_unit": {"type": ["string", "null"]},
                },
                "required": [
                    "row_number",
                    "ingredient_name",
                    "ingredient_code",
                    "amount_value",
                    "amount_unit",
                    "numerator_value",
                    "numerator_unit",
                    "denominator_value",
                    "denominator_unit",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}

_DRUG_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "row_number": {"type": "integer"},
                    "full_product_name": {"type": ["string", "null"]},
                    "brand_name": {"type": ["string", "null"]},
                    "brand_code": {"type": ["string", "null"]},
                    "supplier_name": {"type": ["string", "null"]},
                    "supplier_code": {"type": ["string", "null"]},
                    "dose_form": {"type": ["string", "null"]},
                    "box_size": {"type": ["integer", "null"]},
                },
                "required": [
                    "row_number",
                    "full_product_name",
                    "brand_name",
                    "brand_code",
                    "supplier_name",
                    "supplier_code",
                    "dose_form",
                    "box_size",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


class LlmDrugStructurer:
    def __init__(self, config: Config = Config(), config_filename: str = "config.yaml"):
        self.responses_folder = config.system.llm_mapper_responses_folder
        os.makedirs(self.responses_folder, exist_ok=True)
        self._cost = 0.0
        self.prompts = self._load_drug_prompts(config_filename)

    @staticmethod
    def _load_drug_prompts(config_filename: str) -> dict[str, str]:
        path = Path.cwd() / config_filename
        if not path.exists():
            path = get_project_root() / config_filename
        if not path.exists():
            raise FileNotFoundError(f"Could not find {config_filename} in {Path.cwd()} or project root.")

        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        drug_mapping = raw.get("drug_mapping", {})
        required_keys = ["drug_device_system_prompt", "ingredient_system_prompt", "drug_system_prompt"]
        missing_keys = [key for key in required_keys if key not in drug_mapping]
        if missing_keys:
            raise ValueError(f"Missing drug_mapping prompt keys in {config_filename}: {missing_keys}")
        return {key: drug_mapping[key] for key in required_keys}

    @staticmethod
    def _extract_json_dict(response_text: str) -> dict[str, Any] | None:
        response_json_match = re.search(r"{.*}", response_text, flags=re.DOTALL)
        if not response_json_match:
            return None
        try:
            return json.loads(response_json_match.group(0))
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _first_non_code_column(df: pd.DataFrame, drug_code_column: str) -> str | None:
        for column in df.columns:
            if column != drug_code_column:
                return column
        return None

    @staticmethod
    def _normalize_category(value: Any) -> str:
        text = str(value or "").strip().lower()
        if text in {"drug", "device", "other"}:
            return text
        if "drug" in text:
            return "drug"
        if "device" in text:
            return "device"
        return "other"

    @staticmethod
    def _normalize_optional_code(value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        if not normalized:
            return None
        return normalized

    @staticmethod
    def _format_strength(value: Any, unit: Any) -> str | None:
        normalized_unit = ""
        if isinstance(unit, str):
            normalized_unit = unit.strip()

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value_text = f"{value:g}"
            return f"{value_text} {normalized_unit}".strip()

        if isinstance(value, str):
            value_text = value.strip()
            if value_text:
                return f"{value_text} {normalized_unit}".strip()

        return None

    def _cache_key(self, stage: str, records: list[dict[str, Any]]) -> str:
        payload = json.dumps({"stage": stage, "rows": records}, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def _call_llm_batch(
        self,
        stage: str,
        records: list[dict[str, Any]],
        system_prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any] | None:
        cache_key = self._cache_key(stage, records)
        response_file = os.path.join(self.responses_folder, f"drug_structurer_{stage}_{cache_key}.txt")

        if os.path.exists(response_file):
            with open(response_file, "r", encoding="utf-8") as f:
                response = f.read()
            if response == "*Content filter triggered*":
                return None
            parsed_cached = self._extract_json_dict(response)
            if parsed_cached is None:
                raise ValueError(f"Could not parse cached response from {response_file}.")
            return parsed_cached

        prompt = (
            "Process the provided JSON rows and return structured output exactly as requested.\n"
            "Input JSON:\n"
            f"{json.dumps({'rows': records}, ensure_ascii=False)}"
        )

        response_with_usage = get_llm_response(
            prompt=prompt,
            system_prompt=system_prompt,
            json_schema=schema,
            json_schema_name=f"drug_structurer_{stage}",
        )
        response = response_with_usage["content"]
        if not response:
            with open(response_file, "w", encoding="utf-8") as f:
                f.write("*Content filter triggered*")
            return None

        self._cost = self._cost + response_with_usage["usage"]["total_cost_usd"]

        with open(response_file, "w", encoding="utf-8") as f:
            f.write(response)

        parsed = response_with_usage.get("parsed_json")
        if isinstance(parsed, dict):
            return parsed

        parsed = self._extract_json_dict(response)
        if parsed is None:
            raise ValueError(f"Could not parse {stage} response JSON.")
        return parsed

    @staticmethod
    def _batch_records(batch_df: pd.DataFrame, drug_code_column: str) -> list[dict[str, Any]]:
        payload_df = batch_df.drop(columns=[drug_code_column]).reset_index(drop=True)
        records: list[dict[str, Any]] = []
        for row_number, (_, row) in enumerate(payload_df.iterrows()):
            record = {"row_number": row_number}
            record.update(row.to_dict())
            records.append(record)
        return records

    def structure_drugs(self, df: pd.DataFrame, drug_code_column: str) -> pd.DataFrame:
        if drug_code_column not in df.columns:
            raise ValueError(f"drug_code_column '{drug_code_column}' not found in dataframe.")

        sort_column = self._first_non_code_column(df, drug_code_column)
        sorted_df = df.copy()
        if sort_column is not None:
            sorted_df = sorted_df.sort_values(by=sort_column, kind="mergesort")
        sorted_df = sorted_df.reset_index(drop=True)

        output_rows: list[dict[str, Any]] = []

        for start in range(0, len(sorted_df), _BATCH_SIZE):
            batch_df = sorted_df.iloc[start : start + _BATCH_SIZE].copy().reset_index(drop=True)
            if batch_df.empty:
                continue

            batch_records = self._batch_records(batch_df, drug_code_column)

            classified = self._call_llm_batch(
                stage="classify",
                records=batch_records,
                system_prompt=self.prompts["drug_device_system_prompt"],
                schema=_CLASSIFICATION_SCHEMA,
            )
            if classified is None:
                continue

            category_by_row: dict[int, str] = {}
            for item in classified.get("results", []):
                if not isinstance(item, dict):
                    continue
                row_number = item.get("row_number")
                if isinstance(row_number, int):
                    category_by_row[row_number] = self._normalize_category(item.get("category"))

            drug_row_numbers = [idx for idx in range(len(batch_df)) if category_by_row.get(idx) == "drug"]
            if not drug_row_numbers:
                continue

            drug_records = [batch_records[idx] for idx in drug_row_numbers]

            ingredients = self._call_llm_batch(
                stage="ingredient",
                records=drug_records,
                system_prompt=self.prompts["ingredient_system_prompt"],
                schema=_INGREDIENT_SCHEMA,
            )
            if ingredients is not None:
                for item in ingredients.get("results", []):
                    if not isinstance(item, dict):
                        continue
                    row_number = item.get("row_number")
                    ingredient_name = item.get("ingredient_name")
                    ingredient_code = item.get("ingredient_code")
                    amount_value = item.get("amount_value")
                    amount_unit = item.get("amount_unit")
                    numerator_value = item.get("numerator_value")
                    numerator_unit = item.get("numerator_unit")
                    denominator_value = item.get("denominator_value")
                    denominator_unit = item.get("denominator_unit")
                    if not isinstance(row_number, int) or not (0 <= row_number < len(batch_df)):
                        continue
                    if not isinstance(ingredient_name, str) or not ingredient_name.strip():
                        # Requested behavior: skip rows with missing ingredient output.
                        continue

                    drug_concept_code = batch_df.iloc[row_number][drug_code_column]

                    output_rows.append(
                        {
                            "drug_concept_code": drug_concept_code,
                            "concept_name": ingredient_name.strip(),
                            "concept_code": self._normalize_optional_code(ingredient_code),
                            "concept_class_id": "Ingredient",
                            "domain": "Drug",
                        }
                    )

                    amount_text = self._format_strength(amount_value, amount_unit)
                    if amount_text is not None:
                        output_rows.append(
                            {
                                "drug_concept_code": drug_concept_code,
                                "concept_name": amount_text,
                                "concept_code": None,
                                "concept_class_id": "Amount",
                                "domain": "Drug",
                            }
                        )

                    numerator_text = self._format_strength(numerator_value, numerator_unit)
                    if numerator_text is not None:
                        output_rows.append(
                            {
                                "drug_concept_code": drug_concept_code,
                                "concept_name": numerator_text,
                                "concept_code": None,
                                "concept_class_id": "Numerator",
                                "domain": "Drug",
                            }
                        )

                    denominator_text = self._format_strength(denominator_value, denominator_unit)
                    if denominator_text is not None:
                        output_rows.append(
                            {
                                "drug_concept_code": drug_concept_code,
                                "concept_name": denominator_text,
                                "concept_code": None,
                                "concept_class_id": "Denominator",
                                "domain": "Drug",
                            }
                        )

            drugs = self._call_llm_batch(
                stage="drug",
                records=drug_records,
                system_prompt=self.prompts["drug_system_prompt"],
                schema=_DRUG_SCHEMA,
            )
            if drugs is not None:
                for item in drugs.get("results", []):
                    if not isinstance(item, dict):
                        continue
                    row_number = item.get("row_number")
                    full_product_name = item.get("full_product_name")
                    brand_name = item.get("brand_name")
                    brand_code = item.get("brand_code")
                    supplier_name = item.get("supplier_name")
                    supplier_code = item.get("supplier_code")
                    dose_form = item.get("dose_form")
                    box_size = item.get("box_size")
                    if not isinstance(row_number, int) or not (0 <= row_number < len(batch_df)):
                        continue

                    drug_concept_code = batch_df.iloc[row_number][drug_code_column]

                    if isinstance(full_product_name, str) and full_product_name.strip():
                        output_rows.append(
                            {
                                "drug_concept_code": drug_concept_code,
                                "concept_name": full_product_name.strip(),
                                "concept_code": None,
                                "concept_class_id": "Full Product Name",
                                "domain": "Drug",
                            }
                        )

                    if isinstance(brand_name, str) and brand_name.strip():
                        output_rows.append(
                            {
                                "drug_concept_code": drug_concept_code,
                                "concept_name": brand_name.strip(),
                                "concept_code": self._normalize_optional_code(brand_code),
                                "concept_class_id": "Brand name",
                                "domain": "Drug",
                            }
                        )

                    if isinstance(supplier_name, str) and supplier_name.strip():
                        output_rows.append(
                            {
                                "drug_concept_code": drug_concept_code,
                                "concept_name": supplier_name.strip(),
                                "concept_code": self._normalize_optional_code(supplier_code),
                                "concept_class_id": "Supplier",
                                "domain": "Drug",
                            }
                        )

                    if isinstance(dose_form, str) and dose_form.strip():
                        output_rows.append(
                            {
                                "drug_concept_code": drug_concept_code,
                                "concept_name": dose_form.strip(),
                                "concept_code": None,
                                "concept_class_id": "Dose Form",
                                "domain": "Drug",
                            }
                        )

                    if isinstance(box_size, int):
                        output_rows.append(
                            {
                                "drug_concept_code": drug_concept_code,
                                "concept_name": str(box_size),
                                "concept_code": None,
                                "concept_class_id": "Box Size",
                                "domain": "Drug",
                            }
                        )

        result = pd.DataFrame(
            output_rows,
            columns=["drug_concept_code", "concept_name", "concept_code", "concept_class_id", "domain"],
        )
        if not result.empty:
            result = result.drop_duplicates().reset_index(drop=True)
        return result

    def get_total_cost(self) -> float:
        return self._cost


