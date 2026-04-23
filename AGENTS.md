# AGENTS.md

## Project Scope
- Ariadne maps source clinical terms to OMOP standard concepts using a staged pipeline: cleanup -> candidate search -> LLM selection -> evaluation.
- Core packages are under `src/ariadne/`: `term_cleanup`, `vector_search`, `llm_mapping`, `verbatim_mapping`, `evaluation`, `utils`.
- Runtime behavior is mostly configuration-driven via `config_condition_mapping.yaml` loaded by `Config` in `src/ariadne/utils/config.py`.

## Architecture and Data Flow
- Fast exact matching path: `VocabVerbatimTermMapper` (`src/ariadne/verbatim_mapping/vocab_verbatim_term_mapper.py`) normalizes source terms and looks up a local pickle index (`data/verbatim_mapping_index.pkl`).
- Candidate generation path: `HecateConceptSearcher` (`src/ariadne/vector_search/hecate_concept_searcher.py`) queries external Hecate API and returns ranked candidates (`matched_concept_*`, `match_rank`).
- Context enrichment: `add_concept_context(...)` in `src/ariadne/llm_mapping/concept_context_retriever.py` joins OMOP DB metadata (parents/children/synonyms) onto candidate rows.
- Final selection: `LlmMapper.map_terms(...)` (`src/ariadne/llm_mapping/llm_mapper.py`) runs multi-step prompts from `config_condition_mapping.yaml` and outputs `mapped_concept_id`, `mapped_concept_name`, `mapped_rationale`.
- Evaluation: `evaluate(...)` and `evaluate_concept_search(...)` in `src/ariadne/evaluation/` compare outputs against `data/gold_standards/exact_matching_gs.csv`.

## Critical Runtime Dependencies
- Required env vars are documented in `.env.example` (`VOCAB_CONNECTION_STRING`, `VOCAB_SCHEMA`, `GENAI_PROVIDER`, model/API settings).
- OMOP vocabulary DB access is required for `term_downloader` and `concept_context_retriever`.
- External services:
  - Hecate HTTP API (`https://hecate.pantheon-hds.com/api/search_standard`) for vector/candidate search.
  - OpenAI/Azure/LM Studio via `src/ariadne/utils/gen_ai_api.py` for embeddings + LLM calls.
- spaCy model `en_core_web_sm` must be installed for term normalization (`TermNormalizer`).

## Developer Workflows (Observed)
- Install package in editable mode from repo root: `pip install -e .` (see `README.md`).
- No formal CLI exists; run modules/scripts directly (many files have `if __name__ == "__main__":` examples).
- Build verbatim term corpus/index:
  1. Run `download_terms()` from `src/ariadne/verbatim_mapping/term_downloader.py` to create parquet files in `data/terms/`.
  2. Instantiate `VocabVerbatimTermMapper` to auto-create/load `data/verbatim_mapping_index.pkl`.
- Docs workflow is in `sandbox/documentation.sh` (`mkdocs serve`, `mkdocs gh-deploy`).
- Pytest is configured in `pyproject.toml` for `tests/test_*.py`, but current repo examples are mostly in `sandbox/`.

## Project-Specific Conventions
- Sentinel "no match" concept ID is `-1` (used in `LlmMapper` and evaluators), not `None`.
- LLM responses are cached on disk by source ID + prompt step: `data/llm_mapper_responses/response_<source_id>_s<step>.txt`.
- Column naming is stable and pipeline-oriented; reuse existing names to avoid merge bugs:
  - source: `source_concept_id`, `source_term`, `cleaned_term`
  - candidates: `matched_concept_id`, `matched_concept_name`, `match_rank`, `matched_* context columns`
  - final: `mapped_concept_id`, `mapped_concept_name`, `mapped_rationale`, optional `map_method`
- Paths in config are relative to project root through `resolve_path(...)` in `src/ariadne/utils/utils.py`.

## Guardrails for Changes
- Keep config compatibility with `Config.from_dict(...)` dataclass field names in `src/ariadne/utils/config.py`.
- Preserve caching behavior in `LlmMapper` when changing prompt logic (it is relied on for cost control and reproducibility).
- If you add tests, place them under `tests/` so existing pytest discovery (`pyproject.toml`) picks them up.

