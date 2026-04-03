import types

import pandas as pd

from ariadne.llm_mapping.llm_drug_structurer import (
    DrugStructureResult,
    LlmDrugStructurer,
    normalize_structured_drugs,
)


def _make_config(responses_folder):
    return types.SimpleNamespace(
        system=types.SimpleNamespace(llm_mapper_responses_folder=responses_folder)
    )


def _write_minimal_drug_config(path):
    path.write_text(
        (
            "drug_mapping:\n"
            "  drug_device_system_prompt: classify prompt\n"
            "  ingredient_system_prompt: ingredient prompt\n"
            "  drug_system_prompt: drug prompt\n"
            "  device_system_prompt: device prompt\n"
        ),
        encoding="utf-8",
    )


def test_load_drug_prompts_requires_device_system_prompt(tmp_path):
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        (
            "drug_mapping:\n"
            "  drug_device_system_prompt: classify prompt\n"
            "  ingredient_system_prompt: ingredient prompt\n"
            "  drug_system_prompt: drug prompt\n"
        ),
        encoding="utf-8",
    )

    try:
        LlmDrugStructurer(config=_make_config(str(tmp_path)), config_filename=str(config_file))
        raise AssertionError("Expected ValueError for missing prompt key")
    except ValueError as err:
        assert "device_system_prompt" in str(err)


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
        if stage == "device":
            return {
                "results": [
                    {
                        "row_number": 1,
                        "full_device_name": "Blood glucose strip",
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

    assert result.classification_df.to_dict("records") == [
        {
            "drug_code": "D1",
            "category": "drug",
        },
        {
            "drug_code": "D2",
            "category": "device",
        },
    ]

    assert result.ingredient_df.to_dict("records") == [
        {
            "drug_code": "D1",
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

    assert result.drug_df.to_dict("records") == [
        {
            "drug_code": "D1",
            "full_product_name": "Advil 200 mg oral tablet",
            "brand_name": "Advil",
            "brand_code": "BR-55",
            "supplier_name": "Pfizer",
            "supplier_code": "SUP-9",
            "dose_form": "oral tablet",
            "box_size": 30,
        }
    ]

    assert result.device_df.to_dict("records") == [
        {
            "drug_code": "D2",
            "full_device_name": "Blood glucose strip",
        }
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

    device_call = next(call for call in call_args if call["stage"] == "device")
    assert device_call["system_prompt"] == structurer.prompts["device_system_prompt"]
    device_required_keys = device_call["schema"]["properties"]["results"]["items"]["required"]
    assert "full_device_name" in device_required_keys


def test_normalize_structured_drugs_builds_all_stages_with_fallback_codes():
    structured = DrugStructureResult(
        classification_df=pd.DataFrame(columns=["drug_code", "category"]),
        ingredient_df=pd.DataFrame(
            [
                {
                    "drug_code": "D1",
                    "ingredient_name": "Ibuprofen",
                    "ingredient_code": "ING-123",
                    "amount_value": 200,
                    "amount_unit": "mg",
                    "numerator_value": None,
                    "numerator_unit": None,
                    "denominator_value": None,
                    "denominator_unit": None,
                },
                {
                    "drug_code": "D2",
                    "ingredient_name": "Paracetamol",
                    "ingredient_code": None,
                    "amount_value": 500,
                    "amount_unit": "mg",
                    "numerator_value": None,
                    "numerator_unit": None,
                    "denominator_value": None,
                    "denominator_unit": None,
                },
            ]
        ),
        drug_df=pd.DataFrame(
            [
                {
                    "drug_code": "D1",
                    "full_product_name": "Advil 200 mg oral tablet",
                    "brand_name": "Advil",
                    "brand_code": "BR-55",
                    "supplier_name": "Pfizer",
                    "supplier_code": "SUP-9",
                    "dose_form": "oral tablet",
                    "box_size": 30,
                },
                {
                    "drug_code": "D2",
                    "full_product_name": "Paracetamol 500 mg capsule",
                    "brand_name": None,
                    "brand_code": None,
                    "supplier_name": "Acme Pharma",
                    "supplier_code": None,
                    "dose_form": "capsule",
                    "box_size": 20,
                },
            ]
        ),
        device_df=pd.DataFrame(
            [
                {
                    "drug_code": "DEV-1",
                    "full_device_name": "Blood glucose strip",
                }
            ]
        ),
    )

    normalized = normalize_structured_drugs(structured)

    concept_records = normalized.drug_concept_stage.to_dict("records")
    assert {
        "concept_name": "Advil 200 mg oral tablet",
        "domain_id": "Drug",
        "concept_class_id": "Branded Drug",
        "concept_code": "D1",
    } in concept_records
    assert {
        "concept_name": "Paracetamol 500 mg capsule",
        "domain_id": "Drug",
        "concept_class_id": "Clinical Drug",
        "concept_code": "D2",
    } in concept_records
    assert {
        "concept_name": "oral tablet",
        "domain_id": "Drug",
        "concept_class_id": "Dose Form",
        "concept_code": "oral tablet",
    } in concept_records
    assert {
        "concept_name": "Advil",
        "domain_id": "Drug",
        "concept_class_id": "Brand Name",
        "concept_code": "BR-55",
    } in concept_records
    assert {
        "concept_name": "Acme Pharma",
        "domain_id": "Drug",
        "concept_class_id": "Supplier",
        "concept_code": "Acme Pharma",
    } in concept_records
    assert {
        "concept_name": "Paracetamol",
        "domain_id": "Drug",
        "concept_class_id": "Ingredient",
        "concept_code": "Paracetamol",
    } in concept_records
    assert {
        "concept_name": "Blood glucose strip",
        "domain_id": "Device",
        "concept_class_id": "Device",
        "concept_code": "DEV-1",
    } in concept_records

    relationship_records = normalized.internal_relationship_stage.to_dict("records")
    assert {"concept_code_1": "D1", "concept_code_2": "oral tablet"} in relationship_records
    assert {"concept_code_1": "D1", "concept_code_2": "BR-55"} in relationship_records
    assert {"concept_code_1": "D1", "concept_code_2": "SUP-9"} in relationship_records
    assert {"concept_code_1": "D1", "concept_code_2": "ING-123"} in relationship_records
    assert {"concept_code_1": "D2", "concept_code_2": "capsule"} in relationship_records
    assert {"concept_code_1": "D2", "concept_code_2": "Acme Pharma"} in relationship_records
    assert {"concept_code_1": "D2", "concept_code_2": "Paracetamol"} in relationship_records

    ds_records = normalized.ds_stage.to_dict("records")
    assert {
        "drug_concept_code": "D1",
        "ingredient_concept_code": "ING-123",
        "amount_value": 200,
        "amount_unit": "mg",
        "numerator_value": None,
        "numerator_unit": None,
        "denominator_value": None,
        "denominator_unit": None,
        "box_size": 30,
    } in ds_records
    assert {
        "drug_concept_code": "D2",
        "ingredient_concept_code": "Paracetamol",
        "amount_value": 500,
        "amount_unit": "mg",
        "numerator_value": None,
        "numerator_unit": None,
        "denominator_value": None,
        "denominator_unit": None,
        "box_size": 20,
    } in ds_records
