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
    result_df = structurer.structure_drugs(source_df, drug_code_column=DRUG_CODE_COLUMN)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    print(f"Input rows: {len(source_df)}")
    print(f"Output rows: {len(result_df)}")
    print(f"Total LLM cost (USD): {structurer.get_total_cost():.6f}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()




