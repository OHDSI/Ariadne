from dataclasses import replace
import logging

import pandas as pd

from ariadne.llm_mapping.concept_context_retriever import add_concept_context
from ariadne.llm_mapping.llm_mapper import LlmMapper
from ariadne.utils.config_drug_mapping import ConfigDrugMapping
from ariadne.utils.settings import MappingPerConceptClassSettings
from ariadne.vector_search.abstract_concept_searcher import AbstractConceptSearcher
from ariadne.vector_search.hecate_concept_searcher import HecateConceptSearcher
from ariadne.vector_search.pgvector_concept_searcher import PgvectorConceptSearcher
from ariadne.vector_search.tfidf_concept_searcher import TfidfConceptSearcher
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

_LOGGER = logging.getLogger(__name__)


class DrugMapper:
    def __init__(self, config: ConfigDrugMapping = ConfigDrugMapping()):
        self.config = config

    @staticmethod
    def _validate_input_columns(df: pd.DataFrame) -> None:
        required_columns = {"concept_name", "concept_class_id", "concept_code"}
        missing_columns = sorted(required_columns - set(df.columns))
        if missing_columns:
            raise ValueError(f"drug_concept_stage is missing required columns: {missing_columns}")

    def _filter_brand_rows_matching_ingredients(self, brand_rows: pd.DataFrame) -> pd.DataFrame:
        if brand_rows.empty:
            return brand_rows
        if "ingredient" not in self.config.mapping_per_concept_class:
            raise ValueError(
                "Brand Name mapping requires 'ingredient' config to pre-filter ingredient-like brand names."
            )

        ingredient_vm_settings = self.config.mapping_per_concept_class["ingredient"].verbatim_mapping
        download_terms(settings=ingredient_vm_settings)
        ingredient_verbatim_mapper = VocabVerbatimTermMapper(settings=ingredient_vm_settings)

        brand_probe = brand_rows[["concept_code", "concept_name"]].copy()
        brand_probe = ingredient_verbatim_mapper.map_terms(
            source_terms=brand_probe,
            term_column="concept_name",
            mapped_concept_id_column="mapped_concept_id",
            mapped_concept_name_column="mapped_concept_name",
        )
        to_remove_codes = set(brand_probe.loc[brand_probe["mapped_concept_id"] != -1, "concept_code"])
        if to_remove_codes:
            _LOGGER.info(
                "Removed %d Brand Name rows that matched ingredient verbatim index.",
                len(to_remove_codes),
            )
            return brand_rows[~brand_rows["concept_code"].isin(to_remove_codes)].copy()
        return brand_rows

    def _map_class_rows(self, class_rows: pd.DataFrame, class_settings: MappingPerConceptClassSettings) -> pd.DataFrame:
        vm_settings = class_settings.verbatim_mapping
        llm_settings = class_settings.llm_mapping

        download_terms(settings=vm_settings)
        verbatim_mapper = VocabVerbatimTermMapper(settings=vm_settings)

        work_df = class_rows[["concept_code", "concept_name"]].copy()
        work_df = verbatim_mapper.map_terms(
            source_terms=work_df,
            term_column="concept_name",
            mapped_concept_id_column="mapped_concept_id",
            mapped_concept_name_column="mapped_concept_name",
        )

        unmatched = work_df[work_df["mapped_concept_id"] == -1].copy()
        if not unmatched.empty:
            searcher = self._create_concept_searcher(class_settings)
            try:
                candidates = searcher.search_terms(
                    unmatched,
                    term_column="concept_name",
                )
            finally:
                close_fn = getattr(searcher, "close", None)
                if callable(close_fn):
                    close_fn()

            if not candidates.empty:
                context_cfg = llm_settings.context
                candidates = add_concept_context(
                    concept_table=candidates,
                    add_parents=True,
                    add_children=False,
                    add_synonyms=True,
                    add_clinical_drug_form_child_count=(
                        context_cfg.include_target_clinical_drug_form_child_count
                    ),
                )
                mapper_settings = replace(
                    llm_settings,
                    context=replace(llm_settings.context, include_target_children=False),
                )
                mapper = LlmMapper(settings=mapper_settings)
                llm_matches = mapper.map_terms(
                    source_target_concepts=candidates,
                    source_id_column="concept_code",
                    term_column="concept_name",
                    source_term_column="concept_name",
                    children_column=None
                )
                if not llm_matches.empty:
                    work_df = pd.concat([
                        work_df[work_df["mapped_concept_id"] != -1],
                        llm_matches[["concept_code", "concept_name", "mapped_concept_id", "mapped_concept_name"]]
                    ])
        work_df["mapped_concept_id"] = work_df["mapped_concept_id"].apply(
            lambda value: int(value) if pd.notna(value) and int(value) != -1 else None
        )
        return work_df

    @staticmethod
    def _create_concept_searcher(class_settings: MappingPerConceptClassSettings) -> AbstractConceptSearcher:
        if class_settings.hecate_search is not None:
            return HecateConceptSearcher(settings=class_settings.hecate_search)
        if class_settings.pgvector_search is not None:
            return PgvectorConceptSearcher(settings=class_settings.pgvector_search)
        if class_settings.tfidf_search is not None:
            return TfidfConceptSearcher(settings=class_settings.tfidf_search)
        raise ValueError(
            "Exactly one vector search block must be configured per concept class: "
            "hecate_search, pgvector_search, or tfidf_search."
        )

    def map_drug_concepts(self, drug_concept_stage: pd.DataFrame) -> pd.DataFrame:
        self._validate_input_columns(drug_concept_stage)

        relevant_rows = drug_concept_stage[drug_concept_stage["concept_class_id"].isin(_SUPPORTED_CONCEPT_CLASSES)].copy()
        if relevant_rows.empty:
            return pd.DataFrame(columns=["concept_code", "source_name", "mapped_concept_id", "mapped_concept_name", "drug_class_id"])

        mapped_batches = []
        for concept_class_id in _SUPPORTED_CONCEPT_CLASSES:
            class_rows = relevant_rows[relevant_rows["concept_class_id"] == concept_class_id]
            if class_rows.empty:
                continue

            config_key = _CONCEPT_CLASS_TO_CONFIG_KEY[concept_class_id]
            if config_key not in self.config.mapping_per_concept_class:
                raise ValueError(
                    f"Missing mapping_per_concept_class config for '{config_key}' ({concept_class_id})"
                )

            if concept_class_id == "Brand Name":
                class_rows = self._filter_brand_rows_matching_ingredients(class_rows)
                if class_rows.empty:
                    continue

            class_settings = self.config.mapping_per_concept_class[config_key]
            class_mapped = self._map_class_rows(class_rows, class_settings)
            class_mapped["drug_class_id"] = concept_class_id
            mapped_batches.append(class_mapped)

        if not mapped_batches:
            return pd.DataFrame(columns=["concept_code", "source_name", "mapped_concept_id", "mapped_concept_name", "drug_class_id"])

        relationship_to_concept = pd.concat(mapped_batches, ignore_index=True)
        relationship_to_concept = relationship_to_concept.rename(columns={
            "concept_name": "source_name",
        })
        return relationship_to_concept
