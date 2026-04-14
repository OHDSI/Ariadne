from pathlib import Path

import pandas as pd

from ariadne.evaluation.concept_selection_evaluator import evaluate
from ariadne.llm_mapping import LlmMapper
from ariadne.llm_mapping.concept_context_retriever import add_concept_context
from ariadne.term_cleanup.term_cleaner import TermCleaner
from ariadne.utils.config import Config
from ariadne.vector_search.hecate_concept_searcher import HecateConceptSearcher
from ariadne.verbatim_mapping.term_downloader import download_terms
from ariadne.verbatim_mapping.vocab_verbatim_term_mapper import VocabVerbatimTermMapper

PROCEDURE_RESULTS_FOLDER = Path("E:/git/Ariadne/sandbox/procedure_results")
PROCEDURE_TERM_COLUMN = "source_term"
PROCEDURE_ORIGINAL_TERM_COLUMN = "original_source_term"
PROCEDURE_CODE_COLUMN = "source_code"
MATCHED_CONCEPT_ID_COLUMN = "matched_concept_id"


def main() -> None:
    config = Config("config_procedure_mapping.yaml")

    project_root = Path.cwd().parent

    # Load input file and make unique
    gold_standard_path = project_root / "data" / "gold_standards" / "procedure_mapping_train_set.csv"
    gold_standard = pd.read_csv(gold_standard_path)
    unique_terms = (
        gold_standard.loc[:, [PROCEDURE_CODE_COLUMN, PROCEDURE_ORIGINAL_TERM_COLUMN, PROCEDURE_TERM_COLUMN]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Clean terms
    cleaned_terms_file = project_root / "sandbox" / "procedure_results" / "cleaned_terms.csv"
    if cleaned_terms_file.exists():
        cleaned_terms = pd.read_csv(cleaned_terms_file)
        print("Loaded cleaned terms from file.")
    else:
        term_cleaner = TermCleaner(config.term_cleaning)
        cleaned_terms = term_cleaner.clean_terms(unique_terms,
                                                 term_column=PROCEDURE_TERM_COLUMN)
        print(f"Total LLM cost: ${term_cleaner.get_total_cost():.6f}")
    cleaned_terms.to_csv(cleaned_terms_file, index=False)

    # Verbatim matching
    verbatim_match_file = project_root / "sandbox" / "procedure_results" / "verbatim_maps.csv"
    if verbatim_match_file.exists():
        verbatim_matches = pd.read_csv(verbatim_match_file)
        print("Loaded verbatim matches from file.")
    else:
        download_terms(settings=config.verbatim_mapping) # Downloads the terms as Parquet files to the folder specified in config.yaml.
        verbatim_mapper = VocabVerbatimTermMapper(settings=config.verbatim_mapping) # Will construct the vocabulary index if needed
        verbatim_matches = verbatim_mapper.map_terms(cleaned_terms)
        verbatim_matches.to_csv(verbatim_match_file, index=False)

    # Embedding vector search
    vector_search_results_file = project_root / "sandbox" / "procedure_results"  / "vector_search_results.csv"
    if vector_search_results_file.exists():
        vector_search_results = pd.read_csv(vector_search_results_file)
        print("Loaded vector search results from file.")
    else:
        concept_searcher = HecateConceptSearcher(
            standard_concept="S",
            concept_class_ids=config.verbatim_mapping.standard_concept_filter.concept_class_ids,
            domain_ids=config.verbatim_mapping.standard_concept_filter.domain_ids
        )
        unmatched_terms = cleaned_terms.copy()
        unmatched_terms = unmatched_terms[
            unmatched_terms[PROCEDURE_CODE_COLUMN].isin(
                verbatim_matches[PROCEDURE_CODE_COLUMN][verbatim_matches["mapped_concept_id"] == -1]
            )
        ]
        vector_search_results = concept_searcher.search_terms(unmatched_terms, term_column="cleaned_term")
        vector_search_results.to_csv(vector_search_results_file, index=False)

    # Embedding vector search - Original term
    vector_search_ot_results_file = project_root / "sandbox" / "procedure_results"  / "vector_search_results_ot.csv"
    if vector_search_ot_results_file.exists():
        vector_search_ot_results = pd.read_csv(vector_search_ot_results_file)
        print("Loaded vector search results for the original term from file.")
    else:
        concept_searcher = HecateConceptSearcher(
            standard_concept="S",
            concept_class_ids=config.verbatim_mapping.standard_concept_filter.concept_class_ids,
            domain_ids=config.verbatim_mapping.standard_concept_filter.domain_ids
        )
        unmatched_terms = cleaned_terms.copy()
        unmatched_terms = unmatched_terms[
            unmatched_terms[PROCEDURE_CODE_COLUMN].isin(
                verbatim_matches[PROCEDURE_CODE_COLUMN][verbatim_matches["mapped_concept_id"] == -1]
            )
        ]
        vector_search_ot_results = concept_searcher.search_terms(unmatched_terms, term_column=PROCEDURE_ORIGINAL_TERM_COLUMN)
        vector_search_ot_results.to_csv(vector_search_ot_results_file, index=False)

    # Combine vector search outputs and de-duplicate duplicate target concepts per source.
    vector_search_combined_results_file = project_root / "sandbox" / "procedure_results" / "vector_search_results_combined.csv"
    vector_search_combined_results = pd.concat(
        [vector_search_results, vector_search_ot_results],
        ignore_index=True
    )

    dedup_subset = [PROCEDURE_CODE_COLUMN, MATCHED_CONCEPT_ID_COLUMN]
    if all(column in vector_search_combined_results.columns for column in dedup_subset):
        vector_search_combined_results = vector_search_combined_results.drop_duplicates(subset=dedup_subset)
    else:
        vector_search_combined_results = vector_search_combined_results.drop_duplicates()

    sort_column = PROCEDURE_TERM_COLUMN if PROCEDURE_TERM_COLUMN in vector_search_combined_results.columns else "cleaned_term"
    if sort_column in vector_search_combined_results.columns:
        vector_search_combined_results = vector_search_combined_results.sort_values(by=sort_column).reset_index(drop=True)

    vector_search_combined_results.to_csv(vector_search_combined_results_file, index=False)

    # LLM exact matching
    context_file_name = project_root / "sandbox" / "procedure_results" / "vector_search_combined_context.csv"
    if context_file_name.exists():
        vector_search_results_context = pd.read_csv(context_file_name)
        print("Loaded vector search context from file.")
    else:
        vector_search_results_context = add_concept_context(vector_search_combined_results)
        vector_search_results_context.to_csv(context_file_name, index=False)

    llm_mapper = LlmMapper(settings=config.llm_mapping)
    mapped_terms = llm_mapper.map_terms(vector_search_results_context,
                                        source_id_column=PROCEDURE_CODE_COLUMN,
                                        source_term_column=PROCEDURE_TERM_COLUMN,
                                        source_context_columns=[PROCEDURE_ORIGINAL_TERM_COLUMN],
                                        allow_multiple_targets=True)

    llm_mapped_terms_file = project_root / "sandbox" / "procedure_results" / "llm_mapped_terms.csv"
    mapped_terms.to_csv(llm_mapped_terms_file, index=False)
    print(f"Total LLM cost: ${llm_mapper.get_total_cost():.6f}")

    # Combine verbatim matches and LLM matches
    # First take all verbatim matches that were successful
    final_mapped_terms = verbatim_matches[verbatim_matches["mapped_concept_id"] != -1][
        [
            PROCEDURE_CODE_COLUMN,
            PROCEDURE_TERM_COLUMN,
            PROCEDURE_ORIGINAL_TERM_COLUMN,
            "cleaned_term",
            "mapped_concept_id",
            "mapped_concept_name"
         ]
    ].copy()
    final_mapped_terms["map_method"] = "verbatim"

    # Then add the LLM matches (for the terms that were not verbatim matched)
    llm_mapped_terms_filtered = mapped_terms[
        [
            PROCEDURE_CODE_COLUMN,
            PROCEDURE_TERM_COLUMN,
            PROCEDURE_ORIGINAL_TERM_COLUMN,
            "cleaned_term",
            "mapped_concept_id",
            "mapped_concept_name",
            "mapped_rationale",
        ]
    ].copy()
    llm_mapped_terms_filtered["map_method"] = "llm"
    final_mapped_terms = pd.concat([final_mapped_terms, llm_mapped_terms_filtered], ignore_index=True)
    final_mapped_terms_file = project_root / "sandbox" / "procedure_results" / "final_mapped_terms.csv"
    final_mapped_terms.to_csv(final_mapped_terms_file, index=False)

    # Evaluate
    final_evaluation_results = evaluate(final_mapped_terms,
                                        gold_standard_file=gold_standard_path)
    final_evaluation_results.to_csv(
        project_root / "sandbox" / "procedure_results" / "exact_matching_final_evaluation.csv", index=False
    )


if __name__ == "__main__":
    main()
