"""Configuration dataclasses for the SNOMED CT attribute extraction pipeline.

Loaded from the ``hierarchy`` section of ``config.yaml``::

    cfg = HierarchyConfig.from_yaml("config.yaml")
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ariadne.utils.utils import get_project_root

logger = logging.getLogger(__name__)

_DEFAULT_SNOMED_RELATIONSHIPS: list[str] = [
    'Has asso morph', 'Has finding site', 'Has causative agent',
    'Has clinical course', 'Has finding context', 'Has interpretation',
    'Has interprets', 'Has occurrence', 'Has pathology',
    'Has relat context', 'Has severity', 'Has temporal context',
    'Finding asso with',
]


@dataclass
class ModelsConfig:
    """LLM / embedding model identifiers."""

    embedding: str = "text-embedding-3-large"
    extraction: str = "o3"
    selection: str = "o3"


@dataclass
class RetrievalConfig:
    """Retrieval-stage hyper-parameters."""

    num_reference_examples: int = 5
    top_k_per_category: int = 20
    hnsw_ef_search: int = 200


@dataclass
class ScoringConfig:
    """Similarity score thresholds / overrides."""

    reference_similarity: float = 0.9
    hierarchy_similarity: float = 0.85


@dataclass
class EvaluationConfig:
    """Paths used by the evaluation harness."""

    gold_standard_path: str = "./data/gold_standards/hierarchy_attributes_train_test_gs.csv"
    output_dir: str = "./data/gold_standards"


@dataclass
class PromptsConfig:
    """Prompt templates for extraction and selection steps."""

    extraction: str = ""
    selection: str = ""


@dataclass
class HierarchyConfig:
    """Central configuration for the SNOMED CT attribute extraction pipeline.

    Load from YAML (recommended)::

        cfg = HierarchyConfig.from_yaml("config.yaml")

    A bare ``HierarchyConfig()`` will raise ``ValueError`` because the
    extraction and selection prompts must be non-empty.
    """

    models: ModelsConfig = field(default_factory=ModelsConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    snomed_relationships: list[str] = field(default_factory=lambda: list(_DEFAULT_SNOMED_RELATIONSHIPS))

    @classmethod
    def from_dict(cls, data: dict) -> "HierarchyConfig":
        """Build a ``HierarchyConfig`` from an already-parsed dict.

        This is the shared construction logic used by both :meth:`from_yaml`
        and ``Config`` (which reads the YAML once for the whole toolkit).

        Args:
            data: The ``hierarchy`` section of the parsed YAML (a plain dict).
        """
        return cls(
            models=ModelsConfig(**data.get("models", {})),
            retrieval=RetrievalConfig(**data.get("retrieval", {})),
            scoring=ScoringConfig(**data.get("scoring", {})),
            evaluation=EvaluationConfig(**data.get("evaluation", {})),
            prompts=PromptsConfig(**data.get("prompts", {})),
            snomed_relationships=data.get("snomed_relationships", list(_DEFAULT_SNOMED_RELATIONSHIPS)),
        )

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> "HierarchyConfig":
        """Load hierarchy config from the ``hierarchy`` section of a YAML file.

        Searches CWD first, then project root.
        """
        p = Path(path)
        if not p.exists():
            p = get_project_root() / path
        if not p.exists():
            raise FileNotFoundError(
                f"Config file '{path}' not found in CWD or project root. "
                f"The hierarchy pipeline requires a config.yaml with prompts."
            )

        with p.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        # Support both nested (config.yaml with 'hierarchy' key) and flat layout
        raw = raw.get("hierarchy", raw)
        return cls.from_dict(raw)

    def __post_init__(self):
        """Validate that required fields are populated."""
        if not self.prompts.extraction or not self.prompts.extraction.strip():
            raise ValueError(
                "HierarchyConfig.prompts.extraction is empty. "
                "Load config via HierarchyConfig.from_yaml() to get prompts from YAML."
            )
        if not self.prompts.selection or not self.prompts.selection.strip():
            raise ValueError(
                "HierarchyConfig.prompts.selection is empty. "
                "Load config via HierarchyConfig.from_yaml() to get prompts from YAML."
            )
