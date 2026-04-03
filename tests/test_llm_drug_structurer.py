import types

import pandas as pd

from ariadne.llm_mapping.llm_drug_structurer import LlmDrugStructurer


def _make_config(responses_folder):
    return types.SimpleNamespace(
        system=types.SimpleNamespace(llm_mapper_responses_folder=responses_folder)
    )


def _write_minimal_drug_config(path):
    path.write_text(
        """
drug_mapping:
  drug_device_system_prompt: classify prompt
  ingredient_system_prompt: ingredient prompt
  drug_system_prompt: drug prompt
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_load_drug_prompts_requires_drug_system_prompt(tmp_path):
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        """
drug_mapping:
  drug_device_system_prompt: classify prompt
  ingredient_system_prompt: ingredient prompt
""".strip()
        + "\n",
        encoding="utf-8",
    )

    try:
        LlmDrugStructurer(config=_make_config(str(tmp_path)), config_filename=str(config_file))
        raise AssertionError("Expected ValueError for missing prompt key")
    except ValueError as err:
        assert "drug_system_prompt" in str(err)


def test_structure_drugs_extracts_full_product_name_dose_form_and_box_size(tmp_path, monkeypatch):
    config_file = tmp_path / "test_config.yaml"
    _write_minimal_drug_config(config_file)

    structurer = LlmDrugStructurer(
        config=_make_config(str(tmp_path)),
        config_filename=str(config_file),
    )

    call_args = []

    def fake_call_llm_batch(stage, records, system_prompt, schema):
        call_args.append(
            {
                "stage": stage,
                "system_prompt": system_prompt,
                "schema": schema,
                "records": records,
            }
        )
        if stage == "classify":
            return {
                "results": [
                    {"row_number": 0, "category": "drug"},
                    {"row_number": 1, "category": "device"},
                ]
            }
        if stage == "ingredient":
            return {
                "results": [
                    {
                        "row_number": 0,
                        "ingredient_name": "Ibuprofen",
                        "ingredient_code": "ING-123",
                        "amount_value": 200,
                        "amount_unit": "mg",
                        "numerator_value": None,
                        "numerator_unit": None,
                        "denominator_value": None,
                        "denominator_unit": None,
                    },
                ]
            }
        if stage == "drug":
            return {
                "results": [
                    {
                        "row_number": 0,
                        "full_product_name": "Advil 200 mg oral tablet",
                        "brand_name": "Advil",
                        "brand_code": "BR-55",
                        "supplier_name": "Pfizer",
                        "supplier_code": "SUP-9",
                        "dose_form": "oral tablet",
                        "box_size": 30,
                    }
                ]
            }
        raise AssertionError(f"Unexpected stage {stage}")

    monkeypatch.setattr(structurer, "_call_llm_batch", fake_call_llm_batch)

    source_df = pd.DataFrame(
        {
            "drug_code": ["D1", "D2"],
            "name": ["Advil 200 mg tablet", "Blood glucose strip"],
        }
    )

    result = structurer.structure_drugs(source_df, "drug_code")

    assert result.to_dict("records") == [
        {
            "drug_concept_code": "D1",
            "concept_name": "Ibuprofen",
            "concept_code": "ING-123",
            "concept_class_id": "Ingredient",
            "domain": "Drug",
        },
        {
            "drug_concept_code": "D1",
            "concept_name": "200 mg",
            "concept_code": None,
            "concept_class_id": "Amount",
            "domain": "Drug",
        },
        {
            "drug_concept_code": "D1",
            "concept_name": "Advil 200 mg oral tablet",
            "concept_code": None,
            "concept_class_id": "Full Product Name",
            "domain": "Drug",
        },
        {
            "drug_concept_code": "D1",
            "concept_name": "Advil",
            "concept_code": "BR-55",
            "concept_class_id": "Brand name",
            "domain": "Drug",
        },
        {
            "drug_concept_code": "D1",
            "concept_name": "Pfizer",
            "concept_code": "SUP-9",
            "concept_class_id": "Supplier",
            "domain": "Drug",
        },
        {
            "drug_concept_code": "D1",
            "concept_name": "oral tablet",
            "concept_code": None,
            "concept_class_id": "Dose Form",
            "domain": "Drug",
        },
        {
            "drug_concept_code": "D1",
            "concept_name": "30",
            "concept_code": None,
            "concept_class_id": "Box Size",
            "domain": "Drug",
        },
    ]

    ingredient_call = next(call for call in call_args if call["stage"] == "ingredient")
    ingredient_required_keys = ingredient_call["schema"]["properties"]["results"]["items"]["required"]
    assert "amount_value" in ingredient_required_keys
    assert "amount_unit" in ingredient_required_keys
    assert "numerator_value" in ingredient_required_keys
    assert "numerator_unit" in ingredient_required_keys
    assert "denominator_value" in ingredient_required_keys
    assert "denominator_unit" in ingredient_required_keys

    drug_call = next(call for call in call_args if call["stage"] == "drug")
    assert drug_call["system_prompt"] == structurer.prompts["drug_system_prompt"]
    drug_required_keys = drug_call["schema"]["properties"]["results"]["items"]["required"]
    assert "full_product_name" in drug_required_keys
    assert "dose_form" in drug_required_keys
    assert "box_size" in drug_required_keys
