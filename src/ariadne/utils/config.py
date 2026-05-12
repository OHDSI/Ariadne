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
from typing import Any, Dict, cast

import yaml

from ariadne.utils.settings import (
    HierarchySettings,
    LlmMapperSettings,
    TermCleanerSettings,
    VectorSearchSettings,
    VerbatimMappingSettings,
    build_dataclass,
    serialize_dataclass,
)
from ariadne.utils.utils import get_project_root


class Config:
    """
    Configuration class for the exact-matching workflow.

    Loads settings from a YAML file and provides structured access via
    per-component settings dataclasses that are shared with other config
    classes (e.g. :class:`ConfigDrugMapping`).
    """

    verbatim_mapping: VerbatimMappingSettings
    term_cleaning: TermCleanerSettings
    vector_search: VectorSearchSettings
    llm_mapping: LlmMapperSettings
    hierarchy: HierarchySettings | None

    def __init__(self, filename: str = "config_condition_mapping.yaml"):
        """
        Initializes the Config object by loading settings from the specified YAML file.

        Args:
            filename: The path to the YAML configuration file. Defaults to
                        'config_condition_mapping.yaml' in the current working
                        directory or project root.
        """
        path = Path.cwd() / filename
        if not path.exists():
            path = get_project_root() / filename
            if not path.exists():
                raise FileNotFoundError(f"Could not find {filename} in {Path.cwd()} or project root.")
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        self.verbatim_mapping = build_dataclass(VerbatimMappingSettings, raw.get("verbatim_mapping", {}))
        self.term_cleaning = build_dataclass(TermCleanerSettings, raw.get("term_cleaning", {}))
        self.vector_search = build_dataclass(VectorSearchSettings, raw.get("vector_search", {}))
        self.llm_mapping = build_dataclass(LlmMapperSettings, raw.get("llm_mapping", {}))
        hierarchy_raw = raw.get("hierarchy")
        self.hierarchy = build_dataclass(HierarchySettings, hierarchy_raw) if hierarchy_raw is not None else None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "verbatim_mapping": serialize_dataclass(self.verbatim_mapping),
            "term_cleaning": serialize_dataclass(self.term_cleaning),
            "vector_search": serialize_dataclass(self.vector_search),
            "llm_mapping": serialize_dataclass(self.llm_mapping),
        }
        if self.hierarchy is not None:
            result["hierarchy"] = serialize_dataclass(self.hierarchy)
        return result


if __name__ == "__main__":
    config = Config()
    print(config.to_dict())
