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
import os


def get_project_root() -> Path:
    """Returns the path to the project root directory.

    Assumes this file is at src/ariadne/utils/config.py,
    so the root is 3 levels up.
    """
    return Path(__file__).resolve().parent.parent.parent.parent


def get_environment_variable(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise EnvironmentError(f"Environment variable '{name}' is not set.")
    return value


def resolve_path(path: str) -> str:
    """If the path is relative, makes it absolute by prepending the project root."""
    p = Path(path)
    if not p.is_absolute():
        p = get_project_root() / p
    return str(p.resolve())
