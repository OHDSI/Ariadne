"""SNOMED CT attribute extraction pipeline (hierarchy sub-package).

Public API re-exports for convenient imports::

    from ariadne.hierarchy import (
        HierarchySettings,
        SnomedAttributeSearcher,
        SnomedReferenceSearcher,
        find_attributes_two_stage,
        process_hierarchy,
        build_prediction_rows,
        evaluate_results,
    )
"""

from ariadne.hierarchy.classifier import (
    classify_delta,
    classification_summary,
    parse_classification_results,
    pre_classification_checks,
    resolve_parent_names,
)
from ariadne.hierarchy.evaluator import build_prediction_rows, evaluate_results
from ariadne.hierarchy.attribute_ref_table_builder import (
    build_attribute_reference_tables as build_indexes,
    check_populated,
)
from ariadne.hierarchy.pipeline import ContentFilterError, find_attributes_two_stage
from ariadne.hierarchy.rf2_exporter import export_to_rf2
from ariadne.hierarchy.runner import process_hierarchy
from ariadne.hierarchy.searchers import (
    ATTR_KEY_TO_GS_CATEGORY,
    ATTR_KEY_TO_SNOMED_CATEGORY,
    SNOMED_CATEGORY_TO_ATTR_KEY,
    AbstractSnomedSearcher,
    SnomedAttributeSearcher,
    SnomedReferenceSearcher,
)
from ariadne.hierarchy.types import (
    ExtractionResult,
    INTERPRETS_PAIRED_KEYS,
    LlmResult,
    ReferenceRetrievalResult,
    ReferenceSearchResult,
    SearchBatchResult,
    SearchResult,
)
from ariadne.utils.settings import (
    EvaluationConfig,
    HierarchySettings,
    PromptsConfig,
    RetrievalConfig,
    ScoringConfig,
)


__all__ = [
    # Config
    "HierarchySettings",
    "RetrievalConfig",
    "ScoringConfig",
    "EvaluationConfig",
    "PromptsConfig",
    # Searchers
    "AbstractSnomedSearcher",
    "SnomedAttributeSearcher",
    "SnomedReferenceSearcher",
    # Mappings
    "ATTR_KEY_TO_SNOMED_CATEGORY",
    "ATTR_KEY_TO_GS_CATEGORY",
    "SNOMED_CATEGORY_TO_ATTR_KEY",
    # Types
    "LlmResult",
    "ExtractionResult",
    "SearchResult",
    "SearchBatchResult",
    "ReferenceSearchResult",
    "ReferenceRetrievalResult",
    "INTERPRETS_PAIRED_KEYS",
    # Pipeline
    "ContentFilterError",
    "find_attributes_two_stage",
    # Evaluation
    "process_hierarchy",
    "build_prediction_rows",
    "evaluate_results",
    # Index builder
    "build_indexes",
    "check_populated",
    # RF2 exporter
    "export_to_rf2",
    # Classifier
    "classify_delta",
    "parse_classification_results",
    "pre_classification_checks",
    "resolve_parent_names",
    "classification_summary",
]
