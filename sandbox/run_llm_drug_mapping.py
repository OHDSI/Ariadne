from pathlib import Path

import pandas as pd

from ariadne.llm_mapping.drug_mapper import DrugMapper
from ariadne.llm_mapping.llm_drug_structurer import (
    LlmDrugStructurer,
    normalize_structured_drugs,
)
from ariadne.utils.config_drug_mapping import ConfigDrugMapping


INPUT_CSV = Path(r"E:\git\Ariadne\data\sample_data\drug_codes_2_sample.csv")
INPUT_DICT_MD = Path(r"E:\git\Ariadne\data\sample_data\drug_codes_2_dictionary.md")
DRUG_RESULTS_FOLDER = Path(r"E:\git\Ariadne\sandbox\drug_results")
DRUG_CODE_COLUMN = "code"


def main() -> None:
    configDrugMapping = ConfigDrugMapping()

    # Read inpput
    source_df = pd.read_csv(INPUT_CSV, dtype=str)

    # Stage 1: Structure drug data and save to files
    structurer = LlmDrugStructurer(
        settings=configDrugMapping.drug_structuring,
        dictionary_markdown_file=INPUT_DICT_MD,
    )
    result = structurer.structure_drugs(source_df, drug_code_column=DRUG_CODE_COLUMN)

    DRUG_RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    classify_path = DRUG_RESULTS_FOLDER / "classification.csv"
    ingredient_path = DRUG_RESULTS_FOLDER / "ingredients.csv"
    drug_path = DRUG_RESULTS_FOLDER / "drugs.csv"
    device_path = DRUG_RESULTS_FOLDER / "devices.csv"
    result.classification_df.to_csv(classify_path, index=False)
    result.ingredient_df.to_csv(ingredient_path, index=False)
    result.drug_df.to_csv(drug_path, index=False)
    result.device_df.to_csv(device_path, index=False)

    # Stage 2: Normalize drug attributes and save to files
    normalized_result = normalize_structured_drugs(result)

    drug_concept_stage_path = DRUG_RESULTS_FOLDER / "drug_concept_stage.csv"
    internal_relationship_stage_path = DRUG_RESULTS_FOLDER / "internal_relationship_stage.csv"
    ds_stage_path = DRUG_RESULTS_FOLDER / "ds_stage.csv"
    normalized_result.drug_concept_stage.to_csv(drug_concept_stage_path, index=False)
    normalized_result.internal_relationship_stage.to_csv(internal_relationship_stage_path, index=False)
    normalized_result.ds_stage.to_csv(ds_stage_path, index=False)

    # Stage 3: Map to standard concepts
    drug_mapper = DrugMapper(configDrugMapping)
    mapped_drugs = drug_mapper.map_drug_concepts(normalized_result.drug_concept_stage)
    mapped_drugs_path = DRUG_RESULTS_FOLDER / "relationship_to_concept.csv"
    mapped_drugs.to_csv(mapped_drugs_path, index=False)


if __name__ == "__main__":
    main()




