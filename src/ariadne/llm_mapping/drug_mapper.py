import pandas as pd

from ariadne.llm_mapping.concept_context_retriever import add_concept_context
from ariadne.llm_mapping.llm_mapper import LlmMapper
from ariadne.utils.config_drug_mapping import ConfigDrugMapping
from ariadne.utils.settings import MappingPerConceptClassSettings, StandardConceptFilter
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

    @staticmethod
    def _hecate_kwargs(scf: StandardConceptFilter) -> dict:
        """Build keyword arguments for :class:`HecateConceptSearcher` from a filter."""
        return dict(
            standard_concept="S" if scf.standard_concept else "None",
            domain_ids=scf.domain_ids,
            concept_class_ids=scf.concept_class_ids,
            vocabulary_ids=scf.vocabularies,
        )

    def _map_class_rows(self, class_rows: pd.DataFrame, cc: MappingPerConceptClassSettings) -> pd.DataFrame:
        vm_settings = cc.verbatim_mapping
        vs_settings = cc.vector_search
        llm_settings = cc.llm_mapping

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
            hecate_kwargs = self._hecate_kwargs(vm_settings.standard_concept_filter)
            hecate = HecateConceptSearcher(**hecate_kwargs)
            candidates = hecate.search_terms(
                unmatched,
                term_column="concept_name",
                limit=vs_settings.max_candidates,
                standard_concept=hecate_kwargs["standard_concept"],
                domain_ids=hecate_kwargs["domain_ids"],
                concept_class_ids=hecate_kwargs["concept_class_ids"],
                vocabulary_ids=hecate_kwargs["vocabulary_ids"],
            )

            if not candidates.empty:
                context_cfg = llm_settings.context
                candidates = add_concept_context(
                    concept_table=candidates,
                    add_parents=True,
                    add_children=False,
                    add_synonyms=True,
                )
                llm_settings.context.include_target_children = False
                mapper = LlmMapper(settings=llm_settings)
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
            if config_key not in self.config.mapping_per_concept_class:
                raise ValueError(
                    f"Missing mapping_per_concept_class config for '{config_key}' ({concept_class_id})"
                )

            cc = self.config.mapping_per_concept_class[config_key]
            class_mapped = self._map_class_rows(class_rows, cc)
            mapped_batches.append(class_mapped)

        if not mapped_batches:
            return pd.DataFrame(columns=["concept_code", "source_name", "concept_id", "concept_name"])

        relationship_to_concept = pd.concat(mapped_batches, ignore_index=True)
        relationship_to_concept = relationship_to_concept.rename(columns={
            "concept_name": "source_name",
            "mappend_concept_id": "concept_id",
            "mappend_concept_name": "concept_name",
        })
        return relationship_to_concept
