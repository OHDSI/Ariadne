import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import logging

from ariadne.utils.config import load_hierarchy_settings
from ariadne.hierarchy.attribute_ref_table_builder import build_attribute_reference_tables
from ariadne.hierarchy.runner import process_hierarchy
from ariadne.hierarchy.evaluator import build_prediction_rows
from ariadne.hierarchy.searchers import SnomedAttributeSearcher, SnomedReferenceConceptVectorSearcher
from ariadne.hierarchy.evaluator import evaluate_results

load_dotenv()

def main():
    project_root = Path.cwd().parent
    hierarchy_settings = load_hierarchy_settings(project_root / "config_condition_mapping.yaml")

    # Build attribute table if it doesn't exist
    build_attribute_reference_tables(hierarchy_settings, if_exists="skip")

    # Load gold standard
    attribute_gs_path = project_root / "data" / "gold_standards" / "hierarchy_attributes_snomed_gs.csv"
    attribute_gs = pd.read_csv(attribute_gs_path)

    # Define attributes
    raw_results_file = project_root / "data" / "notebook_results" / "hierarchy_results_raw.json"

    if raw_results_file.exists():
        with open(raw_results_file, "r") as f:
            results = json.load(f)
        print(f"Loaded {len(results)} cached results from file.")
    else:
        # Exclude gold standard terms from reference examples to prevent data leakage
        gs_concept_ids = set(attribute_gs["concept_id_1"].unique())
        attr_searcher = SnomedAttributeSearcher(cfg=hierarchy_settings)
        ref_searcher = SnomedReferenceConceptVectorSearcher(cfg=hierarchy_settings, exclude_concept_ids=gs_concept_ids)
        terms = (
            attribute_gs[["concept_id_1", "concept_name_1"]]
            .drop_duplicates()
            .rename(columns={"concept_id_1": "concept_id", "concept_name_1": "concept_name"})
        )
        results = process_hierarchy(
            terms,
            attr_searcher,
            reference_searcher=ref_searcher,
            hierarchy_settings=hierarchy_settings,
        )
        # Save raw results for debugging
        with open(raw_results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Processed {len(results)} terms. Results saved.")

    total_cost = sum(r.get("cost", {}).get("total_cost", 0.0) for r in results if "cost" in r)
    print(f"Total API cost: ${total_cost:.4f}")

    # Save flat predictions CSV (one row per attribute) — input for RF2 export
    results_df = pd.DataFrame(build_prediction_rows(results))
    results_df.to_csv(project_root / "data" / "notebook_results" / "attribute_results.csv", index=False)


    # Set output directory relative to project root
    hierarchy_settings.evaluation.output_dir = str(project_root / "data" / "notebook_results")

    eval_df = evaluate_results(results, str(attribute_gs_path), hierarchy_settings=hierarchy_settings)
    eval_df.head(20)


if __name__ == "__main__":
    main()
