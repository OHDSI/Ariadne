from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ariadne.llm_mapping.drug_mapper import DrugMapper
from ariadne.utils.settings import (
    ConceptClassSettings,
    ConceptContextSettings,
    LlmMapperSettings,
    StandardConceptFilter,
    VerbatimMappingSettings,
    VectorSearchSettings,
)


def _build_test_config(tmp_path):
    shared_context = ConceptContextSettings(
        include_target_parents=False,
        include_target_children=False,
        include_target_synonyms=False,
        include_target_domain=False,
        include_target_class=False,
        include_target_vocabulary=False,
        re_insert_target_details=False,
    )

    concept_classes = {}
    class_key_to_label = {
        "ingredient": "Ingredient",
        "brand_name": "Brand Name",
        "dose_form": "Dose Form",
        "supplier": "Supplier",
        "unit": "Unit",
        "device": "Device",
    }
    for class_key, label in class_key_to_label.items():
        concept_classes[class_key] = ConceptClassSettings(
            verbatim_mapping=VerbatimMappingSettings(
                terms_folder=str(Path(tmp_path) / f"terms_{label.replace(' ', '_').lower()}"),
                verbatim_mapping_index_file=str(Path(tmp_path) / f"index_{label.replace(' ', '_').lower()}.pkl"),
                download_batch_size=1000,
                log_folder=str(Path(tmp_path) / "logs"),
                substrings_to_remove=[],
                standard_concept_filter=StandardConceptFilter(
                    domain_ids=["Drug"],
                    standard_concept=True,
                    concept_class_ids=[label],
                ),
            ),
            llm_mapping=LlmMapperSettings(
                llm_mapper_responses_folder=str(Path(tmp_path) / "responses"),
                context=shared_context,
                system_prompts=[f"Prompt for {label}", "global-step-2"],
            ),
        )

    return SimpleNamespace(
        verbatim_mapping=VerbatimMappingSettings(
            terms_folder=str(Path(tmp_path) / "terms"),
            verbatim_mapping_index_file=str(Path(tmp_path) / "index.pkl"),
            download_batch_size=1000,
            log_folder=str(Path(tmp_path) / "logs"),
            substrings_to_remove=[],
        ),
        vector_search=VectorSearchSettings(max_candidates=25),
        llm_mapping=LlmMapperSettings(
            llm_mapper_responses_folder=str(Path(tmp_path) / "responses"),
            context=shared_context,
            system_prompts=["global-step-1", "global-step-2"],
        ),
        concept_classes=concept_classes,
    )


def test_drug_mapper_runs_class_specific_pipeline(monkeypatch, tmp_path):
    config = _build_test_config(tmp_path)
    mapper = DrugMapper(config=config)

    download_calls = []

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
        def __init__(self, *args, **kwargs):
            pass

        def search_terms(self, df, term_column, **kwargs):
            rows = []
            for _, row in df.iterrows():
                rows.append(
                    {
                        "cleaned_term": row[term_column],
                        "source_concept_id": row["source_concept_id"],
                        "source_term": row["source_term"],
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

        def map_terms(self, source_target_concepts, term_column, source_id_column, source_term_column):
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

    assert len(download_calls) == 6
    assert set(relationship_to_concept.columns) == {"concept_code_1", "concept_id"}

    result = {
        row["concept_code_1"]: row["concept_id"]
        for row in relationship_to_concept.to_dict("records")
    }
    assert result["ING_1"] == 100
    assert result["BR_1"] == 300
    assert result["DF_1"] is None
    assert result["SUP_1"] == 400
    assert result["UNIT_1"] == 200
    assert result["DEV_1"] == 500
    assert "DRUG_1" not in result


def test_drug_mapper_requires_exact_class_config(tmp_path):
    config = _build_test_config(tmp_path)
    del config.concept_classes["brand_name"]
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
