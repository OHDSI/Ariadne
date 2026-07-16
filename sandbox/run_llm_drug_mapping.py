import logging
import os.path
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
DRUG_CODE_COLUMN = "APPID"
FULL_DRUG_NAME_COLUMN = "NM_AMPP"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    configDrugMapping = ConfigDrugMapping()

    DRUG_RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

    # Read inpput
    source_df = pd.read_csv(INPUT_CSV, dtype=str)

    # Stage 1: Structure drug data and save to files
    classify_path = DRUG_RESULTS_FOLDER / "classification.csv"
    ingredient_path = DRUG_RESULTS_FOLDER / "ingredients.csv"
    ingredient_debug_path = DRUG_RESULTS_FOLDER / "ingredients_with_full_drug_name.csv"
    drug_path = DRUG_RESULTS_FOLDER / "drugs.csv"
    device_path = DRUG_RESULTS_FOLDER / "devices.csv"

    if os.path.isfile(classify_path) and os.path.isfile(ingredient_path) and os.path.isfile(ingredient_debug_path) and os.path.isfile(drug_path) and os.path.isfile(device_path):
        logging.info("Structured drug files already exist. Loading from files")
        classification_df = pd.read_csv(classify_path, dtype=str)
        ingredient_df = pd.read_csv(ingredient_path, dtype=str)
        drug_df = pd.read_csv(drug_path, dtype=str)
        device_df = pd.read_csv(device_path, dtype=str)
        result = type("StructuringResult", (object,), {
            "classification_df":classification_df,
            "ingredient_df": ingredient_df,
            "drug_df": drug_df,
            "device_df": device_df})()
    else:
        structurer = LlmDrugStructurer(
            settings=configDrugMapping.drug_structuring,
            dictionary_markdown_file=INPUT_DICT_MD,
        )
        result = structurer.structure_drugs(source_df, drug_code_column=DRUG_CODE_COLUMN)
        result.classification_df.to_csv(classify_path, index=False)
        result.ingredient_df.to_csv(ingredient_path, index=False)
        result.drug_df.to_csv(drug_path, index=False)
        result.device_df.to_csv(device_path, index=False)

        # Debug export: keep ingredient rows but add NM_AMPP when available from source input.
        ingredient_debug_df = result.ingredient_df.copy()
        if FULL_DRUG_NAME_COLUMN not in ingredient_debug_df.columns and FULL_DRUG_NAME_COLUMN in source_df.columns:
            merge_columns = [DRUG_CODE_COLUMN, FULL_DRUG_NAME_COLUMN]
            ingredient_debug_df = ingredient_debug_df.merge(
                source_df[merge_columns].drop_duplicates(subset=[DRUG_CODE_COLUMN]),
                left_on="drug_code",
                right_on=DRUG_CODE_COLUMN,
                how="left",
            )
        ingredient_debug_df.to_csv(ingredient_debug_path, index=False)

    # Stage 2: Normalize drug attributes and save to files
    drug_concept_stage_path = DRUG_RESULTS_FOLDER / "drug_concept_stage.csv"
    internal_relationship_stage_path = DRUG_RESULTS_FOLDER / "internal_relationship_stage.csv"
    ds_stage_path = DRUG_RESULTS_FOLDER / "ds_stage.csv"

    if os.path.isfile(internal_relationship_stage_path) and os.path.isfile(ds_stage_path) and os.path.isfile(drug_concept_stage_path):
        logging.info("Normalized drug files already exist. Loading from files")
        # Load from existing files
        drug_concept_stage = pd.read_csv(drug_concept_stage_path, dtype=str)
        internal_relationship_stage = pd.read_csv(internal_relationship_stage_path, dtype=str)
        ds_stage = pd.read_csv(ds_stage_path, dtype=str)
        normalized_result = type("NormalizedResult", (object,), {
            "drug_concept_stage": drug_concept_stage,
            "internal_relationship_stage": internal_relationship_stage,
            "ds_stage": ds_stage})()
    else:
        normalized_result = normalize_structured_drugs(result)
        normalized_result.drug_concept_stage.to_csv(drug_concept_stage_path, index=False)
        normalized_result.internal_relationship_stage.to_csv(internal_relationship_stage_path, index=False)
        normalized_result.ds_stage.to_csv(ds_stage_path, index=False)

    # Stage 3: Map to standard concepts
    mapped_drugs_path = DRUG_RESULTS_FOLDER / "relationship_to_concept.csv"

    if os.path.isfile(mapped_drugs_path):
        mapped_drugs_df = pd.read_csv(mapped_drugs_path, dtype=str)
    else:
        drug_mapper = DrugMapper(configDrugMapping)
        mapped_drugs = drug_mapper.map_drug_concepts(normalized_result.drug_concept_stage)
        mapped_drugs.to_csv(mapped_drugs_path, index=False)


if __name__ == "__main__":
    main()
