import json
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from ariadne.hierarchy.attribute_table_builder import build_attribute_table
from ariadne.hierarchy.llm_attribute_runner import process_hierarchy
from ariadne.evaluation.hierarchy_attribute_evaluator import build_prediction_rows
from ariadne.hierarchy.searchers import SnomedAttributeSearcher, SnomedReferenceConceptVectorSearcher
from ariadne.evaluation.hierarchy_attribute_evaluator import evaluate_results
from ariadne.hierarchy.mrcm import MrcmRules
from ariadne.hierarchy.mrcm_validator import validate_role_groups
from ariadne.hierarchy.role_grouper import regroup_results
from ariadne.utils.config import Config

load_dotenv()

def main():
    project_root = Path.cwd().parent
    os.chdir(project_root)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    config = Config()
    hierarchy_settings = config.hierarchy

    # Build attribute table if it doesn't exist
    # build_attribute_table(hierarchy_settings, if_exists="skip")

    # Load gold standard
    attribute_gs_path = project_root / "data" / "gold_standards" / "hierarchy_attributes_snomed_gs.csv"
    attribute_gs = pd.read_csv(attribute_gs_path)
    unique_terms = attribute_gs[
        ["source_concept_id", "source_concept_code", "source_vocabulary_id", "source_term"]
    ].drop_duplicates()

    # Define attributes
    # raw_results_file = project_root / "data" / "notebook_results" / "hierarchy_results_raw_debug.json" # For debugging
    raw_results_file = project_root / "data" / "notebook_results" / "hierarchy_results_raw.json"

    if raw_results_file.exists():
        with open(raw_results_file, "r") as f:
            results = json.load(f)
        print(f"Loaded {len(results)} cached results from file.")
    else:
        # Exclude gold standard terms from reference examples to prevent data leakage.
        # concept_id may be absent for new vocabularies; only exclude rows that have one.
        gs_concept_ids = set(attribute_gs["source_concept_id"].dropna().unique())
        attr_searcher = SnomedAttributeSearcher(hierarchy_settings=hierarchy_settings)
        ref_searcher = SnomedReferenceConceptVectorSearcher(hierarchy_settings=hierarchy_settings, exclude_concept_ids=gs_concept_ids)
        # terms = terms[terms['source_term'] == 'Neonatal hypoglycemia'].head(1) # For debugging

        results = process_hierarchy(
            unique_terms,
            attr_searcher,
            reference_searcher=ref_searcher,
            hierarchy_settings=hierarchy_settings,
            source_term_column="source_term",
            source_code_column="source_concept_code",
            source_vocab_column="source_vocabulary_id",
            source_id_column="source_concept_id",
        )
        # Save raw results for debugging
        with open(raw_results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Processed {len(results)} terms. Results saved.")

    total_cost = sum(r.get("cost", {}).get("total_cost", 0.0) for r in results if "cost" in r)
    print(f"Total API cost: ${total_cost:.4f}")

    # Role groups: (re)group cached results deterministically (LLM groups are kept when present)
    mrcm_rules = MrcmRules.load(hierarchy_settings.mrcm, hierarchy_settings.attributes)
    results = regroup_results(results, hierarchy_settings.mrcm, mrcm_rules)
    violations = validate_role_groups(results, mrcm_rules)
    violations.to_csv(project_root / "data" / "notebook_results" / "hierarchy_mrcm_violations.csv", index=False)
    print(f"MRCM violations: {len(violations)} rows for {violations['source_key'].nunique()} terms")

    # Save flat predictions CSV (one row per attribute, with relationship_groups) — input for RF2 export
    results_df = pd.DataFrame(build_prediction_rows(results, hierarchy_settings))
    results_df.to_csv(project_root / "data" / "notebook_results" / "hierarchy_attribute_results.csv", index=False)

    # Set output directory relative to project root
    output_dir = project_root / "data" / "notebook_results"

    eval_df = evaluate_results(results, str(attribute_gs_path), output_dir, hierarchy_settings)
    eval_df.head(20)


if __name__ == "__main__":
    main()
