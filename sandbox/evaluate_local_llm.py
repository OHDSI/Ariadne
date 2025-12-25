from pathlib import Path
import pandas as pd
import os


from ariadne.llm_mapping.llm_mapper import LlmMapper
from ariadne.utils.config import Config
from ariadne.evaluation.concept_selection_evaluator import evaluate

project_root = Path.cwd().parent
gold_standard_path = project_root / "data" / "gold_standards" / "exact_matching_gs.csv"

gold_standard = pd.read_csv(gold_standard_path)
context_file_name = project_root / "data" / "notebook_results" / "exact_matching_vector_search_context.csv"
vector_search_results_context = pd.read_csv(context_file_name)

# Point environmental variables to local LLM:
os.environ["GENAI_PROVIDER"] = "lm-studio"
os.environ["LLM_MODEL"] = "nvidia/nemotron-3-nano"
os.environ["LM_STUDIO_ENDPOINT"] = "http://localhost:1234/v1"

# Specify a custom folder for caching the LLM responses:
config = Config()
config.system.llm_mapper_responses_folder = project_root / "data" / "nemotron_responses"
llm_mapper = LlmMapper(config)

# Limit to a set of hard cases, and use the LLM to map:
hard_cases = [
    "9724",
    "1423175",
    "1569915",
    "37606067",
    "1102778",
    "1102631",
    "1102646",
    "1102832",
    "1102837",
    "1411507",
    "1434052",
    "1567318",
    "35208218",
    "37083340",
    "37091346",
    "45563320",
    "42489572",
    "42618665",
    "4160345",
    "1413053",
]
mapped_terms = llm_mapper.map_terms(vector_search_results_context,
                                    source_ids=hard_cases)


# Combine verbatim matches and LLM matches
verbatim_match_file = project_root / "data" / "notebook_results" / "exact_matching_verbatim_maps.csv"
verbatim_matches = pd.read_csv(verbatim_match_file)
final_mapped_terms = verbatim_matches[verbatim_matches["mapped_concept_id"] != -1][
    ["source_concept_id", "source_term", "cleaned_term", "mapped_concept_id", "mapped_concept_name"]
].copy()
final_mapped_terms["map_method"] = "verbatim"
llm_mapped_terms_filtered = mapped_terms[
    ["source_concept_id", "source_term", "cleaned_term", "mapped_concept_id", "mapped_concept_name", "mapped_rationale"]
].copy()
llm_mapped_terms_filtered["map_method"] = "llm"
final_mapped_terms = pd.concat([final_mapped_terms, llm_mapped_terms_filtered], ignore_index=True)

# Evaluate
final_evaluation_results = evaluate(final_mapped_terms, source_ids=hard_cases)
final_evaluation_results.to_csv(project_root / "data" / "notebook_results" / "nemotron_3_final_evaluation.csv", index=False)
overall_accuracy = final_evaluation_results["is_correct"].mean()
print(f"Overall accuracy: {overall_accuracy:.4f}")
