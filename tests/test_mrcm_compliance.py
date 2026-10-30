"""MRCM compliance of real pipeline outputs — one test per term.

Marked ``data`` (excluded by default). Run with::

    pytest -m data tests/test_mrcm_compliance.py
    ARIADNE_HIERARCHY_RESULTS=path/to/results.json pytest -m data tests/test_mrcm_compliance.py

Inputs:
    * MRCM refsets from ``hierarchy.mrcm.folder`` (``data/mrcm``).
    * Results from ``$ARIADNE_HIERARCHY_RESULTS`` (JSON list file or folder of per-term
      JSON files), else ``data/notebook_results/hierarchy_results_raw.json``, else the
      per-term cache ``hierarchy.evaluation.llm_response_folder``.
    * Optional range checks: ``$ARIADNE_SNOMED_RELATIONSHIP_FILE`` or the RF2 relationship
      snapshot found under ``data/SnomedCT_*``.

A failing test is a term whose role groups break an MRCM / grouping rule; the failure
message lists the violated rules. Warnings (e.g. ambiguous grouping) are reported as
pytest warnings without failing.
"""

import os
import warnings
from pathlib import Path

import pytest
import yaml

from ariadne.hierarchy.mrcm import MrcmRules, Rf2IsAIndex
from ariadne.hierarchy.llm_attribute_runner import load_hierarchy_results
from ariadne.hierarchy.mrcm_validator import ERROR, validate_result
from ariadne.utils.settings import HierarchySettings, build_dataclass

pytestmark = pytest.mark.data

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _settings() -> HierarchySettings:
    raw = yaml.safe_load((PROJECT_ROOT / "config_condition_mapping.yaml").read_text())
    return build_dataclass(HierarchySettings, raw["hierarchy"])


def _results_path(settings: HierarchySettings) -> Path | None:
    candidates = [
        os.environ.get("ARIADNE_HIERARCHY_RESULTS"),
        PROJECT_ROOT / "data" / "notebook_results" / "hierarchy_results_raw.json",
        settings.evaluation.llm_response_folder,
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists() and (Path(candidate).is_file() or any(Path(candidate).glob("*.json"))):
            return Path(candidate)
    return None


def _relationship_file() -> Path | None:
    explicit = os.environ.get("ARIADNE_SNOMED_RELATIONSHIP_FILE")
    if explicit:
        return Path(explicit)
    found = sorted(PROJECT_ROOT.glob("data/SnomedCT_*/Snapshot/Terminology/sct2_Relationship_Snapshot_*.txt"))
    return found[-1] if found else None


def _data_tests_selected(config: pytest.Config) -> bool:
    """Whether ``-m`` selects ``data`` tests; the default ``-m 'not data'`` from pyproject's addopts does not."""
    markexpr = config.getoption("markexpr")
    return not markexpr or ("data" in markexpr and "not data" not in markexpr)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """One test per cached result; files are only read when ``data`` tests are selected."""
    if "result" not in metafunc.fixturenames:
        return
    results_path = _results_path(_settings()) if _data_tests_selected(metafunc.config) else None
    results = [r for r in load_hierarchy_results(results_path) if not r.get("error")] if results_path else []
    if results:
        metafunc.parametrize("result", results, ids=[f"{r.get('source_key')}|{r.get('source_term')}" for r in results])
    else:
        skip = pytest.mark.skip(reason="no hierarchy results found (set ARIADNE_HIERARCHY_RESULTS)")
        metafunc.parametrize("result", [pytest.param(None, marks=skip)])


@pytest.fixture(scope="module")
def rules() -> MrcmRules:
    settings = _settings()
    try:
        return MrcmRules.load(settings.mrcm, settings.attributes)
    except FileNotFoundError as exc:
        pytest.skip(f"MRCM files not available: {exc}")


@pytest.fixture(scope="module")
def is_a():
    path = _relationship_file()
    return Rf2IsAIndex.from_relationship_file(path) if path and path.exists() else None


def test_term_complies_with_mrcm(result, rules, is_a):
    violations = validate_result(result, rules, is_a)
    for warning in [v for v in violations if v["severity"] != ERROR]:
        warnings.warn(f"{result.get('source_term')}: {warning['rule']} — {warning['detail']}", stacklevel=1)

    errors = [v for v in violations if v["severity"] == ERROR]
    assert not errors, f"{result.get('source_term')} ({result.get('source_key')}) breaks MRCM rules:\n" + "\n".join(
        f"  - {v['rule']}: {v['attribute'] or ''} {v['concept_name'] or ''} — {v['detail']}" for v in errors
    )
