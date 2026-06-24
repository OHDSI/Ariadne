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
    HecateSearchSettings,
    HierarchySettings,
    LlmMapperSettings,
    PgvectorSearchSettings,
    TermCleanerSettings,
    TfidfSearchSettings,
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
    hecate_search: HecateSearchSettings | None
    pgvector_search: PgvectorSearchSettings | None
    tfidf_search: TfidfSearchSettings | None
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
        hecate_raw = raw.get("hecate_search")
        pgvector_raw = raw.get("pgvector_search")
        tfidf_raw = raw.get("tfidf_search")
        self.hecate_search = build_dataclass(HecateSearchSettings, hecate_raw) if hecate_raw is not None else None
        self.pgvector_search = build_dataclass(PgvectorSearchSettings, pgvector_raw) if pgvector_raw is not None else None
        self.tfidf_search = build_dataclass(TfidfSearchSettings, tfidf_raw) if tfidf_raw is not None else None
        self.llm_mapping = build_dataclass(LlmMapperSettings, raw.get("llm_mapping", {}))
        hierarchy_raw = raw.get("hierarchy")
        self.hierarchy = build_dataclass(HierarchySettings, hierarchy_raw) if hierarchy_raw is not None else None

        configured_searchers = [
            name
            for name, value in (
                ("hecate_search", self.hecate_search),
                ("pgvector_search", self.pgvector_search),
                ("tfidf_search", self.tfidf_search),
            )
            if value is not None
        ]
        if len(configured_searchers) != 1:
            raise ValueError(
                "Exactly one top-level vector search block must be configured: "
                "hecate_search, pgvector_search, or tfidf_search. "
                f"Configured: {configured_searchers or 'none'}."
            )

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "verbatim_mapping": serialize_dataclass(self.verbatim_mapping),
            "term_cleaning": serialize_dataclass(self.term_cleaning),
            "llm_mapping": serialize_dataclass(self.llm_mapping),
        }
        if self.hecate_search is not None:
            result["hecate_search"] = serialize_dataclass(self.hecate_search)
        if self.pgvector_search is not None:
            result["pgvector_search"] = serialize_dataclass(self.pgvector_search)
        if self.tfidf_search is not None:
            result["tfidf_search"] = serialize_dataclass(self.tfidf_search)
        if self.hierarchy is not None:
            result["hierarchy"] = serialize_dataclass(self.hierarchy)
        return result


if __name__ == "__main__":
    config = Config()
    print(config.to_dict())
