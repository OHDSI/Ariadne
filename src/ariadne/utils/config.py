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

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import yaml

from ariadne.utils.utils import get_project_root, resolve_path


def load_config(filename) -> Dict[str, Any]:
    # 1. Check User's Current Working Directory (Override)
    user_config = Path.cwd() / filename
    if user_config.exists():
        with open(user_config, "r") as f:
            return yaml.safe_load(f)

    # 2. Check Project Root (Development Mode)
    dev_root_config = get_project_root() / filename
    if dev_root_config.exists():
        with open(dev_root_config, "r") as f:
            return yaml.safe_load(f)

    raise FileNotFoundError(
        f"Could not find {filename} in {Path.cwd()} or project root."
    )


@dataclass
class Config:
    """
    Configuration settings for Ariadne components. By default, loads from 'config.yaml' in the project root or current
    working directory.
    """

    log_folder: str
    terms_folder: str
    download_batch_size: int
    verbatim_mapping_index_file: str
    max_cores: int

    vocabularies: List[str]
    domain_ids: List[str]
    include_classification_concepts: bool
    include_synonyms: bool

    def __init__(self, filename: str = "config.yaml"):
        config = load_config(filename)
        if config is None:
            return
        system = config["system"]
        for key, value in system.items():
            setattr(self, key, value)
        vector_store = config["verbatim_mapping"]
        for key, value in vector_store.items():
            setattr(self, key, value)

        self.log_folder = resolve_path(self.log_folder)
        self.terms_folder = resolve_path(self.terms_folder)
        self.verbatim_mapping_index_file = resolve_path(
            self.verbatim_mapping_index_file
        )

    def __post_init__(self):
        if self.download_batch_size <= 0:
            raise ValueError(f"download_batch_size must be a positive integer")
