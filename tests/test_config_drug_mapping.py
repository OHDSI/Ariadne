from pathlib import Path

from ariadne.utils.config_drug_mapping import ConfigDrugMapping
from ariadne.utils.utils import get_project_root


def test_config_drug_mapping_parses_concept_classes(tmp_path):
    config_path = tmp_path / "drug_config.yaml"
    config_path.write_text(
        (
            "system:\n"
            "  log_folder: logs\n"
            "  terms_folder: data/terms\n"
            "  verbatim_mapping_index_file: data/index.pkl\n"
            "  llm_mapper_responses_folder: data/responses\n"
            "  download_batch_size: 100\n"
            "  max_cores: 1\n"
            "vector_search:\n"
            "  max_candidates: 25\n"
            "llm_mapping:\n"
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
            "  drug_device_system_prompt: classify\n"
            "  ingredient_system_prompt: ingredient\n"
            "  drug_system_prompt: drug\n"
            "  device_system_prompt: device\n"
            "concept_classes:\n"
            "  ingredient:\n"
            "    terms_folder: data/terms_ing\n"
            "    verbatim_mapping_index_file: data/index_ing.pkl\n"
            "    substrings_to_remove:\n"
            "      - hydrochloride\n"
            "    domain_ids:\n"
            "      - Drug\n"
            "    standard_concept: true\n"
            "    vocabularies:\n"
            "    concept_class_ids:\n"
            "      - Ingredient\n"
            "    system_prompt: ingredient prompt\n"
        ),
        encoding="utf-8",
    )

    config = ConfigDrugMapping(filename=str(config_path))

    assert "ingredient" in config.concept_classes
    ingredient_config = config.concept_classes["ingredient"]
    assert ingredient_config.standard_concept is True
    assert ingredient_config.concept_class_ids == ["Ingredient"]
    assert ingredient_config.substrings_to_remove == ["hydrochloride"]
    assert Path(ingredient_config.terms_folder) == get_project_root() / "data" / "terms_ing"
