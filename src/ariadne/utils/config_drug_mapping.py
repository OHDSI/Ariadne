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
    DrugStructuringSettings,
    MappingPerConceptClassSettings,
    build_dataclass,
    serialize_dataclass,
)
from ariadne.utils.utils import get_project_root


class ConfigDrugMapping:
    """
    Configuration class for the drug-mapping workflow.

    Loads settings from a YAML file and provides structured access via
    per-component settings dataclasses shared with :class:`Config`.
    """

    drug_structuring: DrugStructuringSettings
    mapping_per_concept_class: Dict[str, MappingPerConceptClassSettings]

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

        self.drug_structuring = build_dataclass(DrugStructuringSettings, raw.get("drug_structuring", {}))
        self.mapping_per_concept_class = {
            name: build_dataclass(MappingPerConceptClassSettings, cc_raw)
            for name, cc_raw in (raw.get("mapping_per_concept_class") or {}).items()
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drug_structuring": serialize_dataclass(self.drug_structuring),
            "mapping_per_concept_class": {
                name: serialize_dataclass(cc)
                for name, cc in self.mapping_per_concept_class.items()
            },
        }


if __name__ == "__main__":
    config = ConfigDrugMapping()
    print(config.to_dict())
