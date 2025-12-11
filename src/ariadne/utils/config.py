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
    log_folder: Path
    terms_folder: Path
    verbatim_mapping_index_file: Path
    llm_mapper_responses_folder: Path
    download_batch_size: int
    max_cores: int

    def __post_init__(self):
        self.log_folder = resolve_path(self.log_folder)
        self.terms_folder = resolve_path(self.terms_folder)
        self.verbatim_mapping_index_file = resolve_path(self.verbatim_mapping_index_file)
        self.llm_mapper_responses_folder = resolve_path(self.llm_mapper_responses_folder)


@dataclass
class StandardConceptFilter:
    vocabularies: Optional[List[str]]
    domain_ids: Optional[List[str]]
    include_classification_concepts: bool
    include_synonyms: bool


@dataclass
class VerbatimMapping:
    substrings_to_remove: List[str]
    standard_concept_filter: StandardConceptFilter


@dataclass
class TermCleaning:
    system_prompt: str


@dataclass
class VectorSearch:
    max_candidates: int


@dataclass
class Context:
    include_target_parents: bool
    include_target_children: bool
    include_target_synonyms: bool
    include_target_domain: bool
    include_target_class: bool
    include_target_vocabulary: bool
    re_insert_target_details: bool


@dataclass
class Llm_mapping:
    context: Context
    system_prompts: List[str]


class Config:
    """
    Configuration class for the Ariadne toolkit. Loads settings from a YAML file and provides structured access to
    configuration parameters.
    """

    system: SystemConfig = field(default_factory=SystemConfig)
    verbatim_mapping: VerbatimMapping = field(default_factory=VerbatimMapping)
    term_cleaning: TermCleaning = field(default_factory=TermCleaning)
    vector_search: VectorSearch = field(default_factory=VectorSearch)
    llm_mapping: Llm_mapping = field(default_factory=Llm_mapping)

    def __init__(self, filename: str = "config.yaml"):
        """
        Initializes the Config object by loading settings from the specified YAML file.

        Args:
            filename: The path to the YAML configuration file. Defaults to 'config.yaml' in the current working
                        directory or project root.
        """

        path = Path.cwd() / filename
        if not path.exists():
            path = get_project_root() / filename
            if not path.exists():
                raise FileNotFoundError(f"Could not find {filename} in {Path.cwd()} or project root.")
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        self.system = self.from_dict(SystemConfig, raw["system"])
        self.verbatim_mapping = self.from_dict(VerbatimMapping, raw["verbatim_mapping"])
        self.term_cleaning = self.from_dict(TermCleaning, raw["term_cleaning"])
        self.vector_search = self.from_dict(VectorSearch, raw["vector_search"])
        self.llm_mapping = self.from_dict(Llm_mapping, raw["llm_mapping"])

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
            "llm_mapping": serialize(self.llm_mapping),
        }


if __name__ == "__main__":
    config = Config()
    print(config.to_dict())
