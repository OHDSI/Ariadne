# Copyright 2025 Observational Health Data Sciences and Informatics
#
# This file is part of Ariadne
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Shared, per-component settings dataclasses used by processing classes.

Top-level config classes (Config, ConfigDrugMapping) are *composed* of these
settings so the same structures are reused across different workflows.
Each processing class accepts only the settings it actually needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Type, get_type_hints

from ariadne.utils.utils import resolve_path


_DEFAULT_SNOMED_RELATIONSHIPS: list[str] = [
    "Has asso morph",
    "Has finding site",
    "Has causative agent",
    "Has clinical course",
    "Has finding context",
    "Has interpretation",
    "Has interprets",
    "Has occurrence",
    "Has pathology",
    "Has relat context",
    "Has severity",
    "Has temporal context",
    "Finding asso with",
    "During",
    "Occurs after",
    "Has due to"
]


# ── generic helper ────────────────────────────────────────────────────────────


def build_dataclass(cls: Type[Any], data: Dict[str, Any] | None) -> Any:
    """Recursively build a dataclass instance from a plain dict."""
    if data is None:
        data = {}
    if not is_dataclass(cls):
        return data
    kw: dict[str, Any] = {}
    # Resolve string annotations to actual types
    try:
        resolved_hints = get_type_hints(cls)
    except Exception:
        resolved_hints = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        # Use resolved type hint if available, else fall back to f.type
        raw_type = resolved_hints.get(f.name, f.type)
        # Unwrap Optional[X] / X | None to get the inner type
        origin = getattr(raw_type, "__origin__", None)
        if origin is not None:
            args = getattr(raw_type, "__args__", ())
            # For Optional[X], args is (X, NoneType); pick the non-None arg
            non_none = [a for a in args if a is not type(None)]
            raw_type = non_none[0] if non_none else raw_type
        if is_dataclass(raw_type) and isinstance(value, dict):
            kw[f.name] = build_dataclass(raw_type, value)
        else:
            kw[f.name] = value
    return cls(**kw)


def serialize_dataclass(obj: Any) -> Any:
    """Recursively serialize a dataclass (or list/dict of dataclasses) to plain dicts."""
    if is_dataclass(obj):
        result = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            result[f.name] = serialize_dataclass(value)
        return result
    elif isinstance(obj, dict):
        return {k: serialize_dataclass(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_dataclass(item) for item in obj]
    else:
        return obj


# ── reusable filter ───────────────────────────────────────────────────────────


@dataclass
class ConceptFilterSettings:
    """Shared concept filters used by vocabulary download and vector search."""

    standard_concept: List[str] = field(default_factory=lambda: ["S"])
    domain_ids: Optional[List[str]] = None
    concept_class_ids: Optional[List[str]] = None
    vocabulary_ids: Optional[List[str]] = None
    exclude_concept_class_ids: Optional[List[str]] = None
    exclude_vocabulary_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        allowed_values = {"S", "C", "None"}
        if not self.standard_concept:
            raise ValueError("filter.standard_concept must contain at least one value.")
        invalid_values = [value for value in self.standard_concept if value not in allowed_values]
        if invalid_values:
            raise ValueError(
                "Invalid filter.standard_concept values: "
                f"{invalid_values}. Allowed values are {sorted(allowed_values)}."
            )


# ── per-component settings ────────────────────────────────────────────────────


@dataclass
class TermCleanerSettings:
    """Everything :class:`TermCleaner` needs."""

    system_prompt: str = ""


@dataclass
class VerbatimMappingSettings:
    """Everything :func:`download_terms`, :class:`VocabVerbatimTermMapper`, and
    :class:`TermNormalizer` need."""

    terms_folder: str = "data/terms"
    verbatim_mapping_index_file: str = "data/verbatim_mapping_index.pkl"
    download_batch_size: int = 100_000
    log_folder: str = "logs"
    substrings_to_remove: List[str] = field(default_factory=list)
    include_synonyms: bool = True
    preferred_vocabulary_ids: List[str] = field(default_factory=list)
    filter: ConceptFilterSettings = field(default_factory=ConceptFilterSettings)

    def __post_init__(self) -> None:
        self.terms_folder = resolve_path(self.terms_folder)
        self.verbatim_mapping_index_file = resolve_path(self.verbatim_mapping_index_file)
        self.log_folder = resolve_path(self.log_folder)


@dataclass
class VectorSearchSettings:
    """Everything concept searchers read from config."""

    max_candidates: int = 25
    substrings_to_remove: List[str] = field(default_factory=list)
    filter: ConceptFilterSettings = field(default_factory=ConceptFilterSettings)
    include_synonyms: bool = True
    include_mapped_terms: bool = True


@dataclass
class ConceptContextSettings:
    """Controls which context columns :func:`add_concept_context` adds."""

    include_target_parents: bool = True
    include_target_children: bool = True
    include_target_synonyms: bool = True
    include_target_domain: bool = True
    include_target_class: bool = True
    include_target_vocabulary: bool = True
    include_target_clinical_drug_form_child_count: bool = False
    re_insert_source_target_details: bool = True


@dataclass
class LlmMapperSettings:
    """Everything :class:`LlmMapper` needs."""

    llm_mapper_responses_folder: str = "data/llm_mapper_responses"
    context: ConceptContextSettings = field(default_factory=ConceptContextSettings)
    system_prompts: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.llm_mapper_responses_folder = resolve_path(self.llm_mapper_responses_folder)


@dataclass
class DrugStructuringSettings:
    """Everything :class:`LlmDrugStructurer` needs."""

    llm_mapper_responses_folder: str = "data/llm_drug_mapper_responses"
    drug_device_system_prompt: str = ""
    ingredient_system_prompt: str = ""
    drug_system_prompt: str = ""
    device_system_prompt: str = ""

    def __post_init__(self) -> None:
        self.llm_mapper_responses_folder = resolve_path(self.llm_mapper_responses_folder)


@dataclass
class MappingPerConceptClassSettings:
    """Per-concept-class settings used by the drug-mapping pipeline."""

    verbatim_mapping: VerbatimMappingSettings = field(
        default_factory=VerbatimMappingSettings
    )
    vector_search: VectorSearchSettings = field(default_factory=VectorSearchSettings)
    llm_mapping: LlmMapperSettings = field(default_factory=LlmMapperSettings)


@dataclass
class RetrievalConfig:
    """Retrieval-stage hyper-parameters for hierarchy extraction."""

    num_reference_examples: int = 5
    top_k_per_category: int = 20
    hnsw_ef_search: int = 200


@dataclass
class ScoringConfig:
    """Similarity score overrides used by hierarchy ranking."""

    reference_similarity: float = 0.9
    hierarchy_similarity: float = 0.85


@dataclass
class EvaluationConfig:
    """Output  for hierarchy evaluation."""

    output_dir: str = "data/notebook_results"


@dataclass
class PromptsConfig:
    """Prompt templates for hierarchy extraction and candidate selection."""

    extraction: str = ""
    selection: str = ""


@dataclass
class IndexBuildConfig:
    """Batch/index build settings used by the hierarchy pgvector builder."""

    reference_sample_size: int = 10_000
    embedding_batch_size: int = 500
    embedding_cache_folder: str = "data/hierarchy_embedding_cache"

    def __post_init__(self) -> None:
        self.embedding_cache_folder = resolve_path(self.embedding_cache_folder)


@dataclass
class HierarchySettings:
    """Settings block loaded from the optional top-level ``hierarchy`` config key."""

    index_build: IndexBuildConfig = field(default_factory=IndexBuildConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    snomed_relationships: List[str] = field(default_factory=lambda: list(_DEFAULT_SNOMED_RELATIONSHIPS))
