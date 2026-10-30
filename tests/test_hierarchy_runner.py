import json
import threading
import time
from types import SimpleNamespace

import openai
import pandas as pd
import pytest

import ariadne.hierarchy.llm_attribute_runner as runner
from ariadne.utils.concurrency import run_concurrently
from ariadne.utils.settings import EvaluationConfig


# ---------------------------------------------------------------------------
# run_concurrently
# ---------------------------------------------------------------------------

def test_run_concurrently_sequential_preserves_order():
    assert list(run_concurrently([3, 1, 2], lambda x: x * 10, max_workers=1)) == [(0, 30), (1, 10), (2, 20)]


def test_run_concurrently_parallel_returns_every_index():
    barrier = threading.Barrier(4, timeout=5)

    def work(x):
        barrier.wait()
        return x * 2

    out = dict(run_concurrently(list(range(4)), work, max_workers=4))
    assert out == {0: 0, 1: 2, 2: 4, 3: 6}


def test_run_concurrently_propagates_errors_and_cancels_pending():
    started = []

    def work(x):
        started.append(x)
        if x == 0:
            raise RuntimeError("boom")
        time.sleep(0.05)
        return x

    with pytest.raises(RuntimeError, match="boom"):
        list(run_concurrently(list(range(50)), work, max_workers=2))
    assert len(started) < 50


# ---------------------------------------------------------------------------
# process_hierarchy
# ---------------------------------------------------------------------------

def _terms(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_term": [f"term {i}" for i in range(n)],
            "source_concept_code": [str(i) for i in range(n)],
            "source_vocabulary_id": ["ICD10CM"] * n,
            "source_concept_id": list(range(1000, 1000 + n)),
        }
    )


def _settings(tmp_path, max_workers: int):
    evaluation = EvaluationConfig(llm_response_folder=str(tmp_path / "responses"), max_workers=max_workers)
    return SimpleNamespace(evaluation=evaluation)


def test_evaluation_config_default_max_workers_is_parallel():
    assert EvaluationConfig().max_workers > 1


def test_process_hierarchy_parallel_caches_results_in_input_order(tmp_path, monkeypatch):
    barrier = threading.Barrier(3, timeout=5)

    def fake_extract(source_term, attribute_searcher, reference_searcher=None, hierarchy_settings=None):
        barrier.wait()  # only passes if three terms run at the same time
        return {"attributes": {}, "cost": {"total_cost": 0.25}}

    monkeypatch.setattr(runner, "extract_attributes", fake_extract)
    results = runner.process_hierarchy(
        _terms(3), attribute_searcher=object(), reference_searcher=object(),
        hierarchy_settings=_settings(tmp_path, max_workers=3),
    )

    assert [r["source_term"] for r in results] == ["term 0", "term 1", "term 2"]
    assert [r["source_concept_id"] for r in results] == [1000, 1001, 1002]
    files = sorted(p.name for p in (tmp_path / "responses").glob("*.json"))
    assert files == ["ICD10CM_0.json", "ICD10CM_1.json", "ICD10CM_2.json"]
    assert json.loads((tmp_path / "responses" / "ICD10CM_1.json").read_text())["source_key"] == "ICD10CM:1"


def test_process_hierarchy_does_not_cache_failed_terms_and_resumes(tmp_path, monkeypatch):
    calls = []

    def fake_extract(source_term, attribute_searcher, reference_searcher=None, hierarchy_settings=None):
        calls.append(source_term)
        if source_term == "term 1":
            raise openai.APIConnectionError(request=None)
        return {"attributes": {}, "cost": {"total_cost": 0.1}}

    monkeypatch.setattr(runner, "extract_attributes", fake_extract)
    settings = _settings(tmp_path, max_workers=2)
    results = runner.process_hierarchy(_terms(3), object(), object(), hierarchy_settings=settings)

    assert {r["source_term"] for r in results} == {"term 0", "term 1", "term 2"}
    assert "error" in next(r for r in results if r["source_term"] == "term 1")
    assert not (tmp_path / "responses" / "ICD10CM_1.json").exists()

    calls.clear()
    results = runner.process_hierarchy(_terms(3), object(), object(), hierarchy_settings=settings)
    assert calls == ["term 1"]  # only the failed term is retried
    assert [r["source_term"] for r in results][-1] == "term 1"


def test_process_hierarchy_max_workers_argument_overrides_settings(tmp_path, monkeypatch):
    thread_ids = set()

    def fake_extract(source_term, attribute_searcher, reference_searcher=None, hierarchy_settings=None):
        thread_ids.add(threading.get_ident())
        return {"attributes": {}}

    monkeypatch.setattr(runner, "extract_attributes", fake_extract)
    runner.process_hierarchy(
        _terms(4), object(), object(), hierarchy_settings=_settings(tmp_path, max_workers=4), max_workers=1
    )
    assert thread_ids == {threading.get_ident()}


def test_process_hierarchy_drops_non_candidate_values_from_cached_results(tmp_path, monkeypatch):
    cached = {
        "source_key": "ICD10CM:0", "source_term": "term 0",
        "attributes": {"finding_site": [{"concept_id": "1", "concept_name": "Offered"},
                                        {"concept_id": "9", "concept_name": "Copied"}]},
        "role_groups": [{"group": 1, "members": [{"attribute": "finding_site", "concept_id": "9"}]}],
        "retrieved_candidates": [{"concept_id": "1", "attribute_key": "finding_site"}],
    }
    folder = tmp_path / "responses"
    folder.mkdir()
    (folder / "ICD10CM_0.json").write_text(json.dumps(cached))
    monkeypatch.setattr(runner, "extract_attributes", lambda *a, **k: pytest.fail("cached term re-extracted"))

    [result] = runner.process_hierarchy(_terms(1), object(), object(), hierarchy_settings=_settings(tmp_path, 1))

    assert [c["concept_id"] for c in result["attributes"]["finding_site"]] == ["1"]
    assert result["role_groups"] == []
    assert result["dropped_selections"] == [{"attribute": "finding_site", "concept_id": "9", "concept_name": "Copied"}]
    assert json.loads((folder / "ICD10CM_0.json").read_text()) == cached  # cache file is not rewritten


def test_load_hierarchy_results_skips_unreadable_files_and_results_without_source_key(tmp_path):
    folder = tmp_path / "responses"
    folder.mkdir()
    (folder / "a.json").write_text(json.dumps({"source_key": "ICD10CM:1"}))
    (folder / "b.json").write_text("{not json")
    (folder / "c.json").write_text(json.dumps({"source_term": "no key"}))
    list_file = tmp_path / "results.json"
    list_file.write_text(json.dumps([{"source_key": "ICD10CM:2"}, {"source_term": "no key"}]))

    assert runner.load_hierarchy_results(folder) == [{"source_key": "ICD10CM:1"}]
    assert runner.load_hierarchy_results(list_file) == [{"source_key": "ICD10CM:2"}]
