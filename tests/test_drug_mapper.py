from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ariadne.llm_mapping.drug_mapper import DrugMapper
from ariadne.utils.settings import (
    ConceptFilterSettings,
    ConceptContextSettings,
    HecateSearchSettings,
    LlmMapperSettings,
    MappingPerConceptClassSettings,
    VerbatimMappingSettings,
)


def _build_test_config(tmp_path):
    shared_context = ConceptContextSettings(
        include_target_parents=False,
        include_target_children=False,
        include_target_synonyms=False,
        include_target_domain=False,
        include_target_class=False,
        include_target_vocabulary=False,
        re_insert_source_target_details=False,
    )

    mapping_per_concept_class = {}
    class_key_to_label = {
        "ingredient": "Ingredient",
        "brand_name": "Brand Name",
        "dose_form": "Dose Form",
        "supplier": "Supplier",
        "unit": "Unit",
        "device": "Device",
    }
    for class_key, label in class_key_to_label.items():
        mapping_per_concept_class[class_key] = MappingPerConceptClassSettings(
            verbatim_mapping=VerbatimMappingSettings(
                terms_folder=str(Path(tmp_path) / f"terms_{label.replace(' ', '_').lower()}"),
                verbatim_mapping_index_file=str(Path(tmp_path) / f"index_{label.replace(' ', '_').lower()}.pkl"),
                download_batch_size=1000,
                log_folder=str(Path(tmp_path) / "logs"),
                substrings_to_remove=[],
                filter=ConceptFilterSettings(
                    domain_ids=["Drug"],
                    standard_concept=["S"],
                    concept_class_ids=[label],
                ),
            ),
            hecate_search=HecateSearchSettings(
                max_candidates=17,
                filter=ConceptFilterSettings(standard_concept=["S"]),
            ),
            llm_mapping=LlmMapperSettings(
                llm_mapper_responses_folder=str(Path(tmp_path) / "responses"),
                context=shared_context,
                system_prompts=[f"Prompt for {label}", "global-step-2"],
            ),
        )

    return SimpleNamespace(mapping_per_concept_class=mapping_per_concept_class)


def test_drug_mapper_runs_class_specific_pipeline(monkeypatch, tmp_path):
    config = _build_test_config(tmp_path)
    mapper = DrugMapper(config=config)

    download_calls = []
    hecate_max_candidates = []

    def fake_download_terms(settings):
        download_calls.append(settings.terms_folder)

    class FakeVerbatimMapper:
        def __init__(self, settings):
            self.settings = settings

        def map_terms(self, source_terms, term_column, mapped_concept_id_column, mapped_concept_name_column):
            mapped = source_terms.copy()
            mapped[mapped_concept_id_column] = -1
            mapped[mapped_concept_name_column] = ""
            mapped.loc[mapped[term_column] == "Aspirin", mapped_concept_id_column] = 100
            mapped.loc[mapped[term_column] == "mg", mapped_concept_id_column] = 200
            mapped.loc[mapped[term_column] == "Aspirin", mapped_concept_name_column] = "Aspirin"
            mapped.loc[mapped[term_column] == "mg", mapped_concept_name_column] = "Milligram"
            return mapped

    class FakeHecateConceptSearcher:
        def __init__(self, settings):
            hecate_max_candidates.append(settings.max_candidates)

        def search_terms(self, df, term_column):
            rows = []
            for _, row in df.iterrows():
                rows.append(
                    {
                        "cleaned_term": row[term_column],
                        "concept_code": row["concept_code"],
                        "concept_name": row["concept_name"],
                        "matched_concept_id": 9000 + len(str(row[term_column])),
                        "matched_concept_name": f"Candidate for {row[term_column]}",
                        "match_score": 0.95,
                        "match_rank": 1,
                    }
                )
            return pd.DataFrame(rows)

    def fake_add_concept_context(concept_table, **kwargs):
        return concept_table

    class FakeLlmMapper:
        def __init__(self, settings):
            self.system_prompts = settings.system_prompts

        def map_terms(self, source_target_concepts, term_column, source_id_column, source_term_column, **kwargs):
            outputs = []
            for term, group in source_target_concepts.groupby(term_column):
                mapped_id = {
                    "Tylenol": 300,
                    "Tablet": -1,
                    "Acme Pharma": 400,
                    "Syringe": 500,
                }.get(term, -1)
                outputs.append(
                    {
                        term_column: term,
                        source_id_column: group.iloc[0][source_id_column],
                        source_term_column: group.iloc[0][source_term_column],
                        "mapped_concept_id": mapped_id,
                        "mapped_concept_name": "mapped" if mapped_id != -1 else "no_match",
                        "mapped_rationale": "test",
                    }
                )
            return pd.DataFrame(outputs)

    monkeypatch.setattr("ariadne.llm_mapping.drug_mapper.download_terms", fake_download_terms)
    monkeypatch.setattr("ariadne.llm_mapping.drug_mapper.VocabVerbatimTermMapper", FakeVerbatimMapper)
    monkeypatch.setattr("ariadne.llm_mapping.drug_mapper.HecateConceptSearcher", FakeHecateConceptSearcher)
    monkeypatch.setattr("ariadne.llm_mapping.drug_mapper.add_concept_context", fake_add_concept_context)
    monkeypatch.setattr("ariadne.llm_mapping.drug_mapper.LlmMapper", FakeLlmMapper)

    drug_concept_stage = pd.DataFrame(
        [
            {"concept_name": "Aspirin", "concept_class_id": "Ingredient", "concept_code": "ING_1"},
            {"concept_name": "Tylenol", "concept_class_id": "Brand Name", "concept_code": "BR_1"},
            {"concept_name": "Tablet", "concept_class_id": "Dose Form", "concept_code": "DF_1"},
            {"concept_name": "Acme Pharma", "concept_class_id": "Supplier", "concept_code": "SUP_1"},
            {"concept_name": "mg", "concept_class_id": "Unit", "concept_code": "UNIT_1"},
            {"concept_name": "Syringe", "concept_class_id": "Device", "concept_code": "DEV_1"},
            {"concept_name": "Ignored", "concept_class_id": "Drug Product", "concept_code": "DRUG_1"},
        ]
    )

    relationship_to_concept = mapper.map_drug_concepts(drug_concept_stage)

    assert len(download_calls) == 7
    assert hecate_max_candidates and all(value == 17 for value in hecate_max_candidates)
    assert set(relationship_to_concept.columns) == {
        "concept_code",
        "source_name",
        "mapped_concept_id",
        "mapped_concept_name",
        "drug_class_id",
    }

    result = {
        row["concept_code"]: row["mapped_concept_id"]
        for row in relationship_to_concept.to_dict("records")
    }
    assert result["ING_1"] == 100
    assert result["BR_1"] == 300
    assert result["DF_1"] is None
    assert result["SUP_1"] == 400
    assert result["UNIT_1"] == 200
    assert result["DEV_1"] == 500
    assert "DRUG_1" not in result

    class_result = {
        row["concept_code"]: row["drug_class_id"]
        for row in relationship_to_concept.to_dict("records")
    }
    assert class_result["ING_1"] == "Ingredient"
    assert class_result["BR_1"] == "Brand Name"
    assert class_result["DF_1"] == "Dose Form"
    assert class_result["SUP_1"] == "Supplier"
    assert class_result["UNIT_1"] == "Unit"
    assert class_result["DEV_1"] == "Device"


def test_drug_mapper_requires_exact_class_config(tmp_path):
    config = _build_test_config(tmp_path)
    del config.mapping_per_concept_class["brand_name"]
    mapper = DrugMapper(config=config)

    df = pd.DataFrame(
        [
            {"concept_name": "Tylenol", "concept_class_id": "Brand Name", "concept_code": "BR_1"},
        ]
    )

    try:
        mapper.map_drug_concepts(df)
        raise AssertionError("Expected ValueError for missing class config")
    except ValueError as err:
        assert "brand_name" in str(err)


def test_brand_name_rows_matching_ingredient_terms_are_removed(monkeypatch, tmp_path):
    config = _build_test_config(tmp_path)
    mapper = DrugMapper(config=config)

    class FakeVerbatimMapper:
        def __init__(self, settings):
            self.settings = settings

        def map_terms(self, source_terms, term_column, mapped_concept_id_column, mapped_concept_name_column):
            mapped = source_terms.copy()
            mapped[mapped_concept_id_column] = -1
            mapped[mapped_concept_name_column] = ""
            index_name = Path(self.settings.verbatim_mapping_index_file).name
            if "ingredient" in index_name:
                mapped.loc[mapped[term_column] == "Aspirin", mapped_concept_id_column] = 100
                mapped.loc[mapped[term_column] == "Aspirin", mapped_concept_name_column] = "Aspirin"
                return mapped
            mapped.loc[mapped[term_column] == "Tylenol", mapped_concept_id_column] = 300
            mapped.loc[mapped[term_column] == "Tylenol", mapped_concept_name_column] = "Tylenol"
            return mapped

    def fake_download_terms(settings):
        return None

    monkeypatch.setattr("ariadne.llm_mapping.drug_mapper.download_terms", fake_download_terms)
    monkeypatch.setattr("ariadne.llm_mapping.drug_mapper.VocabVerbatimTermMapper", FakeVerbatimMapper)

    df = pd.DataFrame(
        [
            {"concept_name": "Aspirin", "concept_class_id": "Brand Name", "concept_code": "BR_ASP"},
            {"concept_name": "Tylenol", "concept_class_id": "Brand Name", "concept_code": "BR_TYL"},
        ]
    )

    mapped = mapper.map_drug_concepts(df)

    assert set(mapped["concept_code"]) == {"BR_TYL"}
    assert mapped.iloc[0]["mapped_concept_id"] == 300


def test_brand_name_mapping_requires_ingredient_config_for_prefilter(tmp_path):
    config = _build_test_config(tmp_path)
    del config.mapping_per_concept_class["ingredient"]
    mapper = DrugMapper(config=config)

    df = pd.DataFrame(
        [
            {"concept_name": "Tylenol", "concept_class_id": "Brand Name", "concept_code": "BR_1"},
        ]
    )

    try:
        mapper.map_drug_concepts(df)
        raise AssertionError("Expected ValueError when ingredient config is missing for brand pre-filter")
    except ValueError as err:
        assert "ingredient" in str(err)

