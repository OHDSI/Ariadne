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

from pathlib import Path
from typing import Any, Dict

import yaml

from ariadne.utils.settings import (
    ConceptClassSettings,
    DrugStructuringSettings,
    LlmMapperSettings,
    VectorSearchSettings,
    VerbatimMappingSettings,
    build_dataclass,
    serialize_dataclass,
)
from ariadne.utils.utils import get_project_root, resolve_path


class ConfigDrugMapping:
    """
    Configuration class for the drug-mapping workflow.

    Loads settings from a YAML file and provides structured access via
    per-component settings dataclasses shared with :class:`Config`.
    """

    verbatim_mapping: VerbatimMappingSettings
    vector_search: VectorSearchSettings
    llm_mapping: LlmMapperSettings
    drug_structuring: DrugStructuringSettings
    concept_classes: Dict[str, ConceptClassSettings]

    def __init__(self, filename: str = "config_drug_mapping.yaml"):
        """
        Initializes the ConfigDrugMapping object by loading settings from the specified YAML file.

        Args:
            filename: The path to the YAML configuration file. Defaults to 'config_drug_mapping.yaml' in the current
                        working directory or project root.
        """
        path = Path.cwd() / filename
        if not path.exists():
            path = get_project_root() / filename
            if not path.exists():
                raise FileNotFoundError(f"Could not find {filename} in {Path.cwd()} or project root.")
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        self.verbatim_mapping = build_dataclass(VerbatimMappingSettings, raw.get("verbatim_mapping", {}))
        self.vector_search = build_dataclass(VectorSearchSettings, raw.get("vector_search", {}))
        self.llm_mapping = build_dataclass(LlmMapperSettings, raw.get("llm_mapping", {}))
        self.drug_structuring = build_dataclass(DrugStructuringSettings, raw.get("drug_structuring", {}))
        self.concept_classes = {
            name: self._merge_shared(build_dataclass(ConceptClassSettings, cc_raw))
            for name, cc_raw in (raw.get("concept_classes") or {}).items()
        }

    # -- private ---------------------------------------------------------------

    def _merge_shared(self, cc: ConceptClassSettings) -> ConceptClassSettings:
        """Fill shared top-level values into a per-concept-class settings object.

        Values that are always inherited from the top-level sections:
        - ``verbatim_mapping``: ``log_folder``, ``download_batch_size``
        - ``llm_mapping``: ``llm_mapper_responses_folder``, ``context``,
          ``system_prompts`` (only when the concept class defines none)
        - ``standard_concept_filter.include_classification_concepts`` is derived
          from ``standard_concept``.
        """
        vm = cc.verbatim_mapping
        vm.log_folder = self.verbatim_mapping.log_folder
        vm.download_batch_size = self.verbatim_mapping.download_batch_size
        vm.standard_concept_filter.include_classification_concepts = (
            not vm.standard_concept_filter.standard_concept
        )

        lm = cc.llm_mapping
        default_folder = resolve_path("data/llm_mapper_responses")
        if lm.llm_mapper_responses_folder == default_folder:
            lm.llm_mapper_responses_folder = self.llm_mapping.llm_mapper_responses_folder
        lm.context = self.llm_mapping.context
        if not lm.system_prompts:
            lm.system_prompts = list(self.llm_mapping.system_prompts)

        return cc

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verbatim_mapping": serialize_dataclass(self.verbatim_mapping),
            "vector_search": serialize_dataclass(self.vector_search),
            "llm_mapping": serialize_dataclass(self.llm_mapping),
            "drug_structuring": serialize_dataclass(self.drug_structuring),
            "concept_classes": {
                name: serialize_dataclass(cc)
                for name, cc in self.concept_classes.items()
            },
        }


if __name__ == "__main__":
    config = ConfigDrugMapping()
    print(config.to_dict())
