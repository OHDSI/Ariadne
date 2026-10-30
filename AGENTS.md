# AGENTS.md

## Project Scope
- Ariadne maps source clinical terms to OMOP standard concepts using a staged pipeline: cleanup -> candidate search -> LLM selection -> evaluation.
- Core packages are under `src/ariadne/`: `term_cleanup`, `vector_search`, `llm_mapping`, `verbatim_mapping`, `evaluation`, `utils`.
- Runtime behavior is mostly configuration-driven via `config_condition_mapping.yaml` loaded by `Config` in `src/ariadne/utils/config.py`.

## Architecture and Data Flow
- Fast exact matching path: `VocabVerbatimTermMapper` (`src/ariadne/verbatim_mapping/vocab_verbatim_term_mapper.py`) normalizes source terms and looks up a local pickle index (`verbatim_mapping.verbatim_mapping_index_file`, e.g. `data/verbatim_condition_index.pkl`).
- Candidate generation path: `HecateConceptSearcher` (`src/ariadne/vector_search/hecate_concept_searcher.py`) queries external Hecate API and returns ranked candidates (`matched_concept_*`, `match_rank`).
- Context enrichment: `add_concept_context(...)` in `src/ariadne/llm_mapping/concept_context_retriever.py` joins OMOP DB metadata (parents/children/synonyms) onto candidate rows.
- Final selection: `LlmMapper.map_terms(...)` (`src/ariadne/llm_mapping/llm_mapper.py`) runs multi-step prompts from `config_condition_mapping.yaml` and outputs `mapped_concept_id`, `mapped_concept_name`, `mapped_rationale`.
- Evaluation: `evaluate(...)` and `evaluate_concept_search(...)` in `src/ariadne/evaluation/` compare outputs against `data/gold_standards/exact_matching_gs.csv`.

## Critical Runtime Dependencies
- Required env vars are documented in `.env.example` (`VOCAB_CONNECTION_STRING`, `VOCAB_SCHEMA`, `GENAI_PROVIDER`, model/API settings).
- OHDSI Vocabularies DB access is required for `term_downloader` and `concept_context_retriever`.
- External services:
  - Hecate HTTP API (`https://hecate.pantheon-hds.com/api/search_standard`) for vector/candidate search.
  - OpenAI/Azure/LM Studio via `src/ariadne/utils/gen_ai_api.py` for embeddings + LLM calls.
- spaCy model `en_core_web_sm` must be installed for term normalization (`TermNormalizer`).
- SNOMED CT MRCM refset snapshots (Domain, AttributeDomain, AttributeRange) in `data/mrcm/` (copied from the SNOMED RF2 release `Snapshot/Refset/Metadata/`) for attribute validation.

## Developer Workflows (Observed)
- Install package in editable mode from repo root: `pip install -e .` (see `README.md`).
- No formal CLI exists; run modules/scripts directly (many files have `if __name__ == "__main__":` examples).
- Build verbatim term corpus/index:
  1. Run `download_terms()` from `src/ariadne/verbatim_mapping/term_downloader.py` to create parquet files in `verbatim_mapping.terms_folder` (e.g. `data/terms_condition/`).
  2. Instantiate `VocabVerbatimTermMapper` to auto-create/load the index at `verbatim_mapping.verbatim_mapping_index_file`.
- Docs workflow is in `sandbox/documentation.sh` (`mkdocs serve`, `mkdocs gh-deploy`).
- Tests live in `tests/` (pytest, configured in `pyproject.toml`); `sandbox/` holds run scripts.

## Project-Specific Conventions
- Sentinel "no match" concept ID is `-1` (used in `LlmMapper` and evaluators), not `None`.
- LLM responses are cached on disk by source ID + prompt step: `data/llm_mapper_responses/response_<source_id>_s<step>.txt`.
- Column naming is stable and pipeline-oriented; reuse existing names to avoid merge bugs:
  - source: `source_concept_id`, `source_term`, `cleaned_term`
  - candidates: `matched_concept_id`, `matched_concept_name`, `match_rank`, `matched_* context columns`
  - final: `mapped_concept_id`, `mapped_concept_name`, `mapped_rationale`, optional `map_method`
- Paths in config are relative to project root through `resolve_path(...)` in `src/ariadne/utils/utils.py`.
- Hierarchy role groups: `result["role_groups"] = [{"group": n, "members": [{attribute, concept_id, concept_name, concept_code}]}]` built by `group_attributes(...)` (`src/ariadne/hierarchy/role_grouper.py`); `result["attributes"]` stays flat for evaluation, and evaluator rows carry `relationship_groups` (e.g. `"1;2"`) used by the RF2 exporter. Grouping rules live in `hierarchy.mrcm.grouping`; `regroup_results(...)` regroups cached results without LLM calls; `validate_role_groups(...)` (`mrcm_validator.py`) reports violations.

## Guardrails for Changes
- Keep YAML keys in sync with the settings dataclass field names in `src/ariadne/utils/settings.py` (built by `Config` in `src/ariadne/utils/config.py` via `build_dataclass`).
- Preserve caching behavior in `LlmMapper` when changing prompt logic (it is relied on for cost control and reproducibility).
- If you add tests, place them under `tests/` so existing pytest discovery (`pyproject.toml`) picks them up.
- Tests needing local data are marked `data` and excluded by default; run them with `pytest -m data` (e.g. `tests/test_mrcm_compliance.py`, results via `ARIADNE_HIERARCHY_RESULTS`).

