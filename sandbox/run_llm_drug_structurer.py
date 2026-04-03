from pathlib import Path

import pandas as pd

from ariadne.llm_mapping.llm_drug_structurer import LlmDrugStructurer
from ariadne.utils.config import Config


INPUT_CSV = Path(r"E:\git\Ariadne\data\sample_data\drug_codes_sample.csv")
OUTPUT_CSV = Path(r"E:\git\Ariadne\sandbox\drug_codes_sample_structured.csv")
CACHE_FOLDER = Path(r"E:\git\Ariadne\sandbox\drug_structurer_responses")
DRUG_CODE_COLUMN = "code"


def main() -> None:
    input_path = INPUT_CSV.resolve()
    output_path = OUTPUT_CSV.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    source_df = pd.read_csv(input_path, dtype=str)
    if DRUG_CODE_COLUMN not in source_df.columns:
        raise ValueError(
            f"drug code column '{DRUG_CODE_COLUMN}' not found. Available columns: {list(source_df.columns)}"
        )

    config = Config()
    config.system.llm_mapper_responses_folder = CACHE_FOLDER
    structurer = LlmDrugStructurer(config=config)
    result = structurer.structure_drugs(source_df, drug_code_column=DRUG_CODE_COLUMN)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_stem = output_path.with_suffix("")
    classify_path = output_stem.with_name(f"{output_stem.name}_classification.csv")
    ingredient_path = output_stem.with_name(f"{output_stem.name}_ingredients.csv")
    drug_path = output_stem.with_name(f"{output_stem.name}_drugs.csv")
    device_path = output_stem.with_name(f"{output_stem.name}_devices.csv")

    result.classification_df.to_csv(classify_path, index=False)
    result.ingredient_df.to_csv(ingredient_path, index=False)
    result.drug_df.to_csv(drug_path, index=False)
    result.device_df.to_csv(device_path, index=False)

    print(f"Input rows: {len(source_df)}")
    print(f"Classification rows: {len(result.classification_df)}")
    print(f"Ingredient rows: {len(result.ingredient_df)}")
    print(f"Drug rows: {len(result.drug_df)}")
    print(f"Device rows: {len(result.device_df)}")
    print(f"Total LLM cost (USD): {structurer.get_total_cost():.6f}")
    print(f"Wrote: {classify_path}")
    print(f"Wrote: {ingredient_path}")
    print(f"Wrote: {drug_path}")
    print(f"Wrote: {device_path}")


if __name__ == "__main__":
    main()




