from types import SimpleNamespace
from typing import Any

import pandas as pd

from ariadne.llm_mapping.concept_context_retriever import add_concept_context
from ariadne.llm_mapping.llm_mapper import LlmMapper
from ariadne.utils.config_drug_mapping import ConfigDrugMapping, DrugMapperConceptClassConfig
from ariadne.vector_search.hecate_concept_searcher import HecateConceptSearcher
from ariadne.verbatim_mapping.term_downloader import download_terms
from ariadne.verbatim_mapping.vocab_verbatim_term_mapper import VocabVerbatimTermMapper


_SUPPORTED_CONCEPT_CLASSES = [
    "Ingredient",
    "Brand Name",
    "Dose Form",
    "Supplier",
    "Unit",
    "Device",
]

_CONCEPT_CLASS_TO_CONFIG_KEY = {
    "Ingredient": "ingredient",
    "Brand Name": "brand_name",
    "Dose Form": "dose_form",
    "Supplier": "supplier",
    "Unit": "unit",
    "Device": "device",
}


class DrugMapper:
    def __init__(self, config: ConfigDrugMapping = ConfigDrugMapping()):
        self.config = config

    @staticmethod
    def _validate_input_columns(df: pd.DataFrame) -> None:
        required_columns = {"concept_name", "concept_class_id", "concept_code"}
        missing_columns = sorted(required_columns - set(df.columns))
        if missing_columns:
            raise ValueError(f"drug_concept_stage is missing required columns: {missing_columns}")

    def _build_class_config(self, class_config: DrugMapperConceptClassConfig):
        class_prompts = self._class_system_prompts(class_config.system_prompt)
        standard_filter = SimpleNamespace(
            vocabularies=class_config.vocabularies,
            domain_ids=class_config.domain_ids,
            concept_class_ids=class_config.concept_class_ids,
            include_classification_concepts=not class_config.standard_concept,
            include_synonyms=class_config.include_synonyms,
            standard_concept=class_config.standard_concept,
        )

        return SimpleNamespace(
            system=SimpleNamespace(
                log_folder=self.config.system.log_folder,
                terms_folder=class_config.terms_folder,
                verbatim_mapping_index_file=class_config.verbatim_mapping_index_file,
                llm_mapper_responses_folder=self.config.system.llm_mapper_responses_folder,
                download_batch_size=self.config.system.download_batch_size,
                max_cores=self.config.system.max_cores,
            ),
            verbatim_mapping=SimpleNamespace(
                substrings_to_remove=class_config.substrings_to_remove,
                standard_concept_filter=standard_filter,
            ),
            vector_search=self.config.vector_search,
            llm_mapping=SimpleNamespace(
                context=self.config.llm_mapping.context,
                system_prompts=class_prompts,
            ),
        )

    def _class_system_prompts(self, class_prompt: str) -> list[str]:
        default_prompts = list(self.config.llm_mapping.system_prompts)
        if not default_prompts:
            return [class_prompt]
        if len(default_prompts) == 1:
            return [class_prompt]
        return [class_prompt, *default_prompts[1:]]

    def _map_class_rows(self, class_rows: pd.DataFrame, class_config: DrugMapperConceptClassConfig) -> pd.DataFrame:
        scoped_config = self._build_class_config(class_config)

        download_terms(config=scoped_config)
        verbatim_mapper = VocabVerbatimTermMapper(config=scoped_config)

        work_df = class_rows[["concept_code", "concept_name"]].copy()
        work_df = verbatim_mapper.map_terms(
            source_terms=work_df,
            term_column="concept_name",
            mapped_concept_id_column="mapped_concept_id",
            mapped_concept_name_column="mapped_concept_name",
        )

        unmatched = work_df[work_df["mapped_concept_id"] == -1].copy()
        if not unmatched.empty:
            unmatched["__row_index"] = unmatched.index
            search_input = (
                unmatched[["concept_name", "concept_code"]]
                .drop_duplicates(subset=["concept_name"])
                .rename(columns={"concept_name": "cleaned_term", "concept_code": "source_concept_id"})
            )
            search_input["source_term"] = search_input["cleaned_term"]

            standard_flag = "S" if class_config.standard_concept else "None"
            hecate = HecateConceptSearcher(
                standard_concept=standard_flag,
                domain_ids=class_config.domain_ids,
                concept_class_ids=class_config.concept_class_ids,
                vocabulary_ids=class_config.vocabularies,
            )
            candidates = hecate.search_terms(
                search_input,
                term_column="cleaned_term",
                limit=25,
                standard_concept=standard_flag,
                domain_ids=class_config.domain_ids,
                concept_class_ids=class_config.concept_class_ids,
                vocabulary_ids=class_config.vocabularies,
            )

            if not candidates.empty:
                context_cfg = self.config.llm_mapping.context
                candidates = add_concept_context(
                    concept_table=candidates,
                    add_parents=context_cfg.include_target_parents,
                    add_children=context_cfg.include_target_children,
                    add_synonyms=context_cfg.include_target_synonyms,
                )

                mapper = LlmMapper(config=scoped_config)
                llm_matches = mapper.map_terms(
                    source_target_concepts=candidates,
                    term_column="cleaned_term",
                    source_id_column="source_concept_id",
                    source_term_column="source_term",
                )
                if not llm_matches.empty:
                    llm_term_match = llm_matches[["cleaned_term", "mapped_concept_id"]].rename(
                        columns={"cleaned_term": "concept_name"}
                    )
                    unmatched = unmatched.merge(llm_term_match, on="concept_name", how="left", suffixes=("", "_llm"))
                    unmatched["mapped_concept_id"] = unmatched["mapped_concept_id_llm"].fillna(-1)
                    unmatched.drop(columns=["mapped_concept_id_llm"], inplace=True)

            work_df.loc[unmatched["__row_index"], "mapped_concept_id"] = unmatched["mapped_concept_id"].values

        work_df["concept_id"] = work_df["mapped_concept_id"].apply(
            lambda value: int(value) if pd.notna(value) and int(value) != -1 else None
        )
        return work_df[["concept_code", "concept_id"]]

    def map_drug_concepts(self, drug_concept_stage: pd.DataFrame) -> pd.DataFrame:
        self._validate_input_columns(drug_concept_stage)

        relevant_rows = drug_concept_stage[drug_concept_stage["concept_class_id"].isin(_SUPPORTED_CONCEPT_CLASSES)].copy()
        if relevant_rows.empty:
            return pd.DataFrame(columns=["concept_code_1", "concept_id"])

        mapped_batches = []
        for concept_class_id in _SUPPORTED_CONCEPT_CLASSES:
            class_rows = relevant_rows[relevant_rows["concept_class_id"] == concept_class_id]
            if class_rows.empty:
                continue

            config_key = _CONCEPT_CLASS_TO_CONFIG_KEY[concept_class_id]
            if config_key not in self.config.concept_classes:
                raise ValueError(f"Missing concept_classes config for '{config_key}' ({concept_class_id})")

            class_config = self.config.concept_classes[config_key]
            class_mapped = self._map_class_rows(class_rows, class_config)
            mapped_batches.append(class_mapped)

        if not mapped_batches:
            return pd.DataFrame(columns=["concept_code_1", "concept_id"])

        relationship_to_concept = pd.concat(mapped_batches, ignore_index=True)
        relationship_to_concept = relationship_to_concept.rename(columns={"concept_code": "concept_code_1"})
        relationship_to_concept = relationship_to_concept[["concept_code_1", "concept_id"]]
        return relationship_to_concept
