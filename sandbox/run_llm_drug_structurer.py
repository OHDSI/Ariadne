from pathlib import Path

import pandas as pd

from ariadne.llm_mapping.llm_drug_structurer import (
    LlmDrugStructurer,
    normalize_structured_drugs,
)
from ariadne.utils.config import Config
from ariadne.utils.config_drug_mapping import ConfigDrugMapping
from ariadne.verbatim_mapping.term_downloader import download_terms
from ariadne.verbatim_mapping.vocab_verbatim_term_mapper import VocabVerbatimTermMapper


INPUT_CSV = Path(r"E:\git\Ariadne\data\sample_data\drug_codes_sample.csv")
DRUG_RESULTS_FOLDER = Path(r"E:\git\Ariadne\sandbox\drug_results")
DRUG_CODE_COLUMN = "code"


def main() -> None:
    configDrugMapping = ConfigDrugMapping()
    source_df = pd.read_csv(INPUT_CSV, dtype=str)
 
    structurer = LlmDrugStructurer(config=configDrugMapping)
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

    normalized_result = normalize_structured_drugs(result)
    drug_concept_stage_path = DRUG_RESULTS_FOLDER / "drug_concept_stage.csv"
    internal_relationship_stage_path = DRUG_RESULTS_FOLDER / "internal_relationship_stage.csv"
    ds_stage_path = DRUG_RESULTS_FOLDER / "ds_stage.csv"

    normalized_result.drug_concept_stage.to_csv(drug_concept_stage_path, index=False)
    normalized_result.internal_relationship_stage.to_csv(internal_relationship_stage_path, index=False)
    normalized_result.ds_stage.to_csv(ds_stage_path, index=False)

    download_terms(config=configDrugMapping)

    ingredients_and_devices = (
        normalized_result.drug_concept_stage
        .loc[normalized_result.drug_concept_stage["concept_class_id"].isin(["Ingredient", "Device"])]
        .copy(deep=True)
        .reset_index(drop=True)
    )

    verbatim_mapper = VocabVerbatimTermMapper(configDrugMapping) # Will construct the vocabulary index if needed
    verbatim_matches = verbatim_mapper.map_terms(ingredients_and_devices, term_column="concept_name")
    verbatim_matches.to_csv(DRUG_RESULTS_FOLDER / "drug_concept_stage_mapped.csv", index=False)

if __name__ == "__main__":
    main()




