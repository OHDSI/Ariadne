from pathlib import Path

import pytest

from ariadne.utils.config_drug_mapping import ConfigDrugMapping
from ariadne.utils.utils import get_project_root


def test_config_drug_mapping_parses_mapping_per_concept_class(tmp_path):
    config_path = tmp_path / "drug_config.yaml"
    config_path.write_text(
        (
            "drug_structuring:\n"
            "  llm_mapper_responses_folder: data/responses\n"
            "  drug_device_system_prompt: classify\n"
            "  ingredient_system_prompt: ingredient\n"
            "  drug_system_prompt: drug\n"
            "  device_system_prompt: device\n"
            "mapping_per_concept_class:\n"
            "  ingredient:\n"
            "    verbatim_mapping:\n"
            "      terms_folder: data/terms_ing\n"
            "      verbatim_mapping_index_file: data/index_ing.pkl\n"
            "      preferred_vocabulary_ids:\n"
            "        - RxNorm\n"
            "        - SNOMED\n"
            "      substrings_to_remove:\n"
            "        - hydrochloride\n"
            "      include_synonyms: true\n"
            "      filter:\n"
            "        domain_ids:\n"
            "          - Drug\n"
            "        standard_concept:\n"
            "          - S\n"
            "        concept_class_ids:\n"
            "          - Ingredient\n"
            "    hecate_search:\n"
            "      max_candidates: 12\n"
            "    llm_mapping:\n"
            "      llm_mapper_responses_folder: data/responses/ingredient\n"
            "      context:\n"
            "        include_target_parents: false\n"
            "        include_target_children: false\n"
            "        include_target_synonyms: false\n"
            "        include_target_domain: false\n"
            "        include_target_class: false\n"
            "        include_target_vocabulary: false\n"
            "        re_insert_target_details: false\n"
            "      system_prompts:\n"
            "        - ingredient prompt\n"
        ),
        encoding="utf-8",
    )

    config = ConfigDrugMapping(filename=str(config_path))

    assert "ingredient" in config.mapping_per_concept_class
    cc = config.mapping_per_concept_class["ingredient"]

    # Verbatim mapping settings parsed correctly
    assert cc.verbatim_mapping.substrings_to_remove == ["hydrochloride"]
    assert cc.verbatim_mapping.preferred_vocabulary_ids == ["RxNorm", "SNOMED"]
    assert Path(cc.verbatim_mapping.terms_folder) == get_project_root() / "data" / "terms_ing"
    assert cc.verbatim_mapping.filter.standard_concept == ["S"]
    assert cc.verbatim_mapping.filter.concept_class_ids == ["Ingredient"]

    # Class-scoped vector search and LLM settings are parsed directly
    assert cc.hecate_search is not None
    assert cc.hecate_search.max_candidates == 12
    assert cc.pgvector_search is None
    assert cc.tfidf_search is None
    assert cc.llm_mapping.system_prompts == ["ingredient prompt"]
    assert Path(cc.llm_mapping.llm_mapper_responses_folder) == get_project_root() / "data" / "responses" / "ingredient"
    assert cc.llm_mapping.context.include_target_parents is False

    assert config.drug_structuring.drug_device_system_prompt == "classify"
    assert not hasattr(config, "vector_search")
    assert not hasattr(config, "verbatim_mapping")
    assert not hasattr(config, "llm_mapping")


def test_concept_class_uses_defaults_when_fields_omitted(tmp_path):
    """Class-scoped settings fall back to dataclass defaults when omitted."""
    config_path = tmp_path / "drug_config.yaml"
    config_path.write_text(
        (
            "drug_structuring:\n"
            "  llm_mapper_responses_folder: data/responses\n"
            "  drug_device_system_prompt: classify\n"
            "  ingredient_system_prompt: ingredient\n"
            "  drug_system_prompt: drug\n"
            "  device_system_prompt: device\n"
            "mapping_per_concept_class:\n"
            "  ingredient:\n"
            "    verbatim_mapping:\n"
            "      terms_folder: data/terms_ing\n"
            "      verbatim_mapping_index_file: data/index_ing.pkl\n"
            "      filter:\n"
            "        domain_ids:\n"
            "          - Drug\n"
            "        standard_concept:\n"
            "          - S\n"
            "        concept_class_ids:\n"
            "          - Ingredient\n"
            "    hecate_search:\n"
            "      filter:\n"
            "        standard_concept:\n"
            "          - S\n"
        ),
        encoding="utf-8",
    )

    config = ConfigDrugMapping(filename=str(config_path))
    cc = config.mapping_per_concept_class["ingredient"]

    assert cc.hecate_search is not None
    assert cc.hecate_search.max_candidates == 25
    assert cc.pgvector_search is None
    assert cc.tfidf_search is None
    assert cc.llm_mapping.system_prompts == []


def test_concept_class_requires_exactly_one_search_block(tmp_path):
    config_path = tmp_path / "drug_config.yaml"
    config_path.write_text(
        (
            "drug_structuring:\n"
            "  llm_mapper_responses_folder: data/responses\n"
            "mapping_per_concept_class:\n"
            "  ingredient:\n"
            "    verbatim_mapping:\n"
            "      filter:\n"
            "        standard_concept:\n"
            "          - S\n"
            "    hecate_search:\n"
            "      max_candidates: 10\n"
            "    tfidf_search:\n"
            "      terms_folder: data/terms\n"
            "      tfidf_index_file: data/index.pkl\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Exactly one vector search block"):
        ConfigDrugMapping(filename=str(config_path))


def test_invalid_standard_concept_value_fails_fast(tmp_path):
    config_path = tmp_path / "drug_config.yaml"
    config_path.write_text(
        (
            "drug_structuring:\n"
            "  llm_mapper_responses_folder: data/responses\n"
            "mapping_per_concept_class:\n"
            "  ingredient:\n"
            "    verbatim_mapping:\n"
            "      filter:\n"
            "        standard_concept:\n"
            "          - X\n"
            "    hecate_search:\n"
            "      filter:\n"
            "        standard_concept:\n"
            "          - S\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        ConfigDrugMapping(filename=str(config_path))

