from pathlib import Path

from ariadne.utils.config_drug_mapping import ConfigDrugMapping
from ariadne.utils.utils import get_project_root


def test_config_drug_mapping_parses_concept_classes(tmp_path):
    config_path = tmp_path / "drug_config.yaml"
    config_path.write_text(
        (
            "verbatim_mapping:\n"
            "  terms_folder: data/terms\n"
            "  verbatim_mapping_index_file: data/index.pkl\n"
            "  download_batch_size: 100\n"
            "  log_folder: logs\n"
            "vector_search:\n"
            "  max_candidates: 25\n"
            "llm_mapping:\n"
            "  llm_mapper_responses_folder: data/responses\n"
            "  context:\n"
            "    include_target_parents: false\n"
            "    include_target_children: false\n"
            "    include_target_synonyms: false\n"
            "    include_target_domain: false\n"
            "    include_target_class: false\n"
            "    include_target_vocabulary: false\n"
            "    re_insert_target_details: false\n"
            "  system_prompts:\n"
            "    - p1\n"
            "    - p2\n"
            "drug_structuring:\n"
            "  llm_mapper_responses_folder: data/responses\n"
            "  drug_device_system_prompt: classify\n"
            "  ingredient_system_prompt: ingredient\n"
            "  drug_system_prompt: drug\n"
            "  device_system_prompt: device\n"
            "concept_classes:\n"
            "  ingredient:\n"
            "    verbatim_mapping:\n"
            "      terms_folder: data/terms_ing\n"
            "      verbatim_mapping_index_file: data/index_ing.pkl\n"
            "      substrings_to_remove:\n"
            "        - hydrochloride\n"
            "      standard_concept_filter:\n"
            "        domain_ids:\n"
            "          - Drug\n"
            "        standard_concept: true\n"
            "        concept_class_ids:\n"
            "          - Ingredient\n"
            "    llm_mapping:\n"
            "      system_prompts:\n"
            "        - ingredient prompt\n"
        ),
        encoding="utf-8",
    )

    config = ConfigDrugMapping(filename=str(config_path))

    assert "ingredient" in config.concept_classes
    cc = config.concept_classes["ingredient"]

    # Verbatim mapping settings parsed correctly
    assert cc.verbatim_mapping.substrings_to_remove == ["hydrochloride"]
    assert Path(cc.verbatim_mapping.terms_folder) == get_project_root() / "data" / "terms_ing"
    assert cc.verbatim_mapping.standard_concept_filter.standard_concept is True
    assert cc.verbatim_mapping.standard_concept_filter.concept_class_ids == ["Ingredient"]

    # Shared values merged from top-level verbatim_mapping
    assert Path(cc.verbatim_mapping.log_folder) == get_project_root() / "logs"
    assert cc.verbatim_mapping.download_batch_size == 100

    # include_classification_concepts derived from standard_concept
    assert cc.verbatim_mapping.standard_concept_filter.include_classification_concepts is False

    # LLM mapping: class-specific prompts are kept as-is (no top-level merge)
    assert cc.llm_mapping.system_prompts == ["ingredient prompt"]
    assert Path(cc.llm_mapping.llm_mapper_responses_folder) == get_project_root() / "data" / "responses"
    assert cc.llm_mapping.context.include_target_parents is False

    # Verify top-level shared settings were parsed
    assert config.llm_mapping.system_prompts == ["p1", "p2"]
    assert config.drug_structuring.drug_device_system_prompt == "classify"
    assert config.vector_search.max_candidates == 25


def test_concept_class_inherits_prompts_when_none_defined(tmp_path):
    """When a concept class defines no system_prompts, all top-level prompts are inherited."""
    config_path = tmp_path / "drug_config.yaml"
    config_path.write_text(
        (
            "verbatim_mapping:\n"
            "  terms_folder: data/terms\n"
            "  verbatim_mapping_index_file: data/index.pkl\n"
            "  download_batch_size: 100\n"
            "  log_folder: logs\n"
            "vector_search:\n"
            "  max_candidates: 25\n"
            "llm_mapping:\n"
            "  llm_mapper_responses_folder: data/responses\n"
            "  context:\n"
            "    include_target_parents: false\n"
            "    include_target_children: false\n"
            "    include_target_synonyms: false\n"
            "    include_target_domain: false\n"
            "    include_target_class: false\n"
            "    include_target_vocabulary: false\n"
            "    re_insert_target_details: false\n"
            "  system_prompts:\n"
            "    - p1\n"
            "    - p2\n"
            "drug_structuring:\n"
            "  llm_mapper_responses_folder: data/responses\n"
            "  drug_device_system_prompt: classify\n"
            "  ingredient_system_prompt: ingredient\n"
            "  drug_system_prompt: drug\n"
            "  device_system_prompt: device\n"
            "concept_classes:\n"
            "  ingredient:\n"
            "    verbatim_mapping:\n"
            "      terms_folder: data/terms_ing\n"
            "      verbatim_mapping_index_file: data/index_ing.pkl\n"
            "      standard_concept_filter:\n"
            "        domain_ids:\n"
            "          - Drug\n"
            "        standard_concept: true\n"
            "        concept_class_ids:\n"
            "          - Ingredient\n"
        ),
        encoding="utf-8",
    )

    config = ConfigDrugMapping(filename=str(config_path))
    cc = config.concept_classes["ingredient"]

    # No class-specific prompts → inherit all from top-level
    assert cc.llm_mapping.system_prompts == ["p1", "p2"]
