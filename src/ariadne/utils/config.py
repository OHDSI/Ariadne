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

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import List, Optional, Any, Type, Dict
import yaml

from ariadne.utils.utils import get_project_root, resolve_path
from spacy.util import from_dict


@dataclass
class SystemConfig:
    log_folder: Path = Path("logs")
    terms_folder: Path = Path("data/terms")
    verbatim_mapping_index_file: Path = Path("data/verbatim_mapping_index.pkl")
    download_batch_size: int = 100000
    max_cores: int = 10

    def __post_init__(self):
        self.log_folder = resolve_path(self.log_folder)
        self.terms_folder = resolve_path(self.terms_folder)
        self.verbatim_mapping_index_file = resolve_path(
            self.verbatim_mapping_index_file
        )


@dataclass
class StandardConceptFilter:
    vocabularies: Optional[List[str]]
    domain_ids: List[str]
    include_classification_concepts: bool
    include_synonyms: bool


@dataclass
class VerbatimMapping:
    substrings_to_remove: List[str]
    standard_concept_filter: StandardConceptFilter


@dataclass
class TermCleaning:
    term_clean_system_prompt: str


@dataclass
class VectorSearch:
    max_candidates: int


class Config:
    system: SystemConfig = field(default_factory=SystemConfig)
    verbatim_mapping: VerbatimMapping = field(default_factory=VerbatimMapping)
    term_cleaning: TermCleaning = field(default_factory=TermCleaning)
    vector_search: VectorSearch = field(default_factory=VectorSearch)

    def __init__(self, filename: str = "config.yaml"):
        path = Path.cwd() / filename
        if not path.exists():
            path = get_project_root() / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"Could not find {filename} in {Path.cwd()} or project root."
                )
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        self.system = self.from_dict(SystemConfig, raw["system"])
        self.verbatim_mapping = self.from_dict(VerbatimMapping, raw["verbatim_mapping"])
        self.term_cleaning = self.from_dict(TermCleaning, raw["term_cleaning"])
        self.vector_search = self.from_dict(VectorSearch, raw["vector_search"])

    def from_dict(self, cls: Type["Config"], data: Dict[str, Any]) -> "Config":
        def build(dc_type: Type[Any], subdata: Dict[str, Any]) -> Any:
            if not is_dataclass(dc_type):
                return subdata
            kw = {}
            for f in fields(dc_type):
                if subdata is None or f.name not in subdata:
                    continue
                value = subdata[f.name]
                if is_dataclass(f.type):
                    kw[f.name] = build(f.type, value or {})
                else:
                    kw[f.name] = value
            return dc_type(**kw)

        return build(cls, data)

    def to_dict(self) -> Dict[str, Any]:
        def serialize(obj: Any) -> Any:
            if is_dataclass(obj):
                result = {}
                for f in fields(obj):
                    value = getattr(obj, f.name)
                    result[f.name] = serialize(value)
                return result
            elif isinstance(obj, list):
                return [serialize(item) for item in obj]
            else:
                return obj

        return {
            "system": serialize(self.system),
            "verbatim_mapping": serialize(self.verbatim_mapping),
            "term_cleaning": serialize(self.term_cleaning),
            "vector_search": serialize(self.vector_search),
        }


if __name__ == "__main__":
    config = Config()
    print(config.to_dict())
