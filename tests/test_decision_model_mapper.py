import json

import pandas as pd
import pytest
import requests

from ariadne.evaluation.decision_model_evaluator import sweep_decision_thresholds
from ariadne.llm_mapping import decision_model_mapper as dmm
from ariadne.llm_mapping.decision_model_mapper import DecisionModelMapper
from ariadne.utils.settings import ConceptContextSettings, DecisionModelMapperSettings


class FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or json.dumps(payload)

    def json(self):
        return self._payload


def _choice_response(probabilities, confidence=0.9):
    return {
        "model": "jev-1.13.0",
        "answers": {
            "mapping": {
                "type": "choice",
                "choice": max(probabilities, key=probabilities.get),
                "confidence": confidence,
                "probabilities": probabilities,
            }
        },
        "usage": {"input_tokens": 1000, "output_tokens": 10},
    }


def _make_mapper(tmp_path, **kwargs):
    kwargs.setdefault("prune_threshold", 0.1)
    settings = DecisionModelMapperSettings(
        responses_folder=str(tmp_path),
        instructions="Pick the exact match",
        max_retries=1,
        **kwargs,
    )
    context = ConceptContextSettings(
        include_target_parents=True,
        include_target_children=False,
        include_target_synonyms=False,
        include_target_domain=False,
        include_target_class=False,
        include_target_vocabulary=False,
    )
    return DecisionModelMapper(settings, context)


def _candidates():
    return pd.DataFrame(
        {
            "source_code": ["A", "A", "A", "B", "B"],
            "source_term": ["Term A"] * 3 + ["Term B"] * 2,
            "cleaned_term": ["term a"] * 3 + ["term b"] * 2,
            "matched_concept_id": [1, 2, 3, 4, 5],
            "matched_concept_name": ["One", "Two", "Three", "Four", "Five"],
            "matched_parents": ["P1", None, "P3", "P4", "P5"],
        }
    )


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(dmm.time, "sleep", lambda s: None)
    state = {"calls": [], "responses": {}}

    def fake_post(url, headers=None, json=None, timeout=None):
        state["calls"].append({"url": url, "headers": headers, "json": json})
        result = state["responses"][json["state"]["source_term"]]
        if isinstance(result, list):
            result = result.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(dmm.requests, "post", fake_post)
    return state


def test_confident_match_and_pruning(tmp_path, api):
    api["responses"]["term a"] = FakeResponse(200, _choice_response({"1": 0.95, "2": 0.03, "3": 0.01, "no_match": 0.01}))
    api["responses"]["term b"] = FakeResponse(200, _choice_response({"4": 0.6, "5": 0.05, "no_match": 0.35}, 0.3))
    mapper = _make_mapper(tmp_path)

    mapped, remaining = mapper.map_terms(_candidates())

    assert mapped["source_code"].tolist() == ["A"]
    assert mapped.iloc[0]["mapped_concept_id"] == 1
    assert mapped.iloc[0]["mapped_concept_name"] == "One"
    assert mapped.iloc[0]["mapped_rationale"] == "Decision model p=0.95 (confidence 0.90)"
    assert remaining["matched_concept_id"].tolist() == [4]
    assert list(remaining.columns) == list(_candidates().columns)
    assert mapper.get_total_tokens() == (2000, 20)
    assert mapper.get_total_cost() == pytest.approx(2000 * 0.042 / 1e6)


def test_request_payload(tmp_path, api):
    api["responses"]["term a"] = FakeResponse(200, _choice_response({"1": 0.95, "2": 0.03, "3": 0.01, "no_match": 0.01}))
    mapper = _make_mapper(tmp_path)

    candidates = _candidates()
    candidates["source_parents"] = ["SP", "SP", "SP", None, None]
    mapper.map_terms(candidates, source_ids=["A"], source_context_columns=["source_parents"])

    call = api["calls"][0]
    assert call["url"] == "https://api.typesafe.ai/v1/systemone"
    assert call["headers"]["Authorization"] == "Bearer test-key"
    body = call["json"]
    assert body["model"] == "jev-1.13.0"
    assert body["state"] == {"source_term": "term a", "source_parents": "SP"}
    question = body["questions"]["mapping"]
    assert question["type"] == "choice"
    assert question["instructions"] == "Pick the exact match"
    assert question["criteria"]["1"] == {"name": "One", "parents": "P1"}
    assert question["criteria"]["2"] == {"name": "Two"}
    assert "no_match" in question["criteria"]
    assert (tmp_path / "response_A.json").exists()
    assert (tmp_path / "request_A.json").exists()


def test_confident_no_match_and_all_pruned(tmp_path, api):
    api["responses"]["term a"] = FakeResponse(200, _choice_response({"1": 0.02, "2": 0.02, "3": 0.01, "no_match": 0.95}))
    api["responses"]["term b"] = FakeResponse(200, _choice_response({"4": 0.08, "5": 0.07, "no_match": 0.85}))
    mapper = _make_mapper(tmp_path)

    mapped, remaining = mapper.map_terms(_candidates())

    assert mapped["mapped_concept_id"].tolist() == [-1, -1]
    assert mapped["mapped_concept_name"].tolist() == ["no_match", "no_match"]
    assert "all candidates" in mapped.iloc[1]["mapped_rationale"]
    assert remaining.empty


def test_failure_falls_back_to_all_candidates_and_is_not_cached(tmp_path, api):
    api["responses"]["term a"] = FakeResponse(503, text="unavailable")
    api["responses"]["term b"] = requests.ConnectionError("down")
    mapper = _make_mapper(tmp_path)

    mapped, remaining = mapper.map_terms(_candidates())

    assert mapped.empty
    assert remaining["matched_concept_id"].tolist() == [1, 2, 3, 4, 5]
    assert len(api["calls"]) == 4  # one retry each
    assert not list(tmp_path.glob("response_*.json"))


def test_retry_then_success(tmp_path, api):
    api["responses"]["term a"] = [
        FakeResponse(429, text="slow down"),
        FakeResponse(200, _choice_response({"1": 0.95, "2": 0.03, "3": 0.01, "no_match": 0.01})),
    ]
    mapper = _make_mapper(tmp_path)

    mapped, _ = mapper.map_terms(_candidates(), source_ids=["A"])

    assert mapped.iloc[0]["mapped_concept_id"] == 1
    assert len(api["calls"]) == 2


def test_missing_options_in_response_falls_back(tmp_path, api):
    api["responses"]["term a"] = FakeResponse(200, _choice_response({"1": 0.95, "no_match": 0.05}))
    mapper = _make_mapper(tmp_path)

    mapped, remaining = mapper.map_terms(_candidates(), source_ids=["A"])

    assert mapped.empty
    assert remaining["matched_concept_id"].tolist() == [1, 2, 3]


def test_cache_is_reused_and_thresholds_applied_after_cache(tmp_path, api):
    api["responses"]["term a"] = FakeResponse(200, _choice_response({"1": 0.85, "2": 0.12, "3": 0.02, "no_match": 0.01}))
    _make_mapper(tmp_path).map_terms(_candidates(), source_ids=["A"])
    assert len(api["calls"]) == 1

    mapper = _make_mapper(tmp_path, match_threshold=0.8)
    mapped, remaining = mapper.map_terms(_candidates(), source_ids=["A"])

    assert len(api["calls"]) == 1
    assert mapped.iloc[0]["mapped_concept_id"] == 1
    assert remaining.empty
    assert mapper.get_total_cost() == 0


def test_invalid_thresholds():
    with pytest.raises(ValueError):
        DecisionModelMapperSettings(match_threshold=0.1, prune_threshold=0.5)


def test_score_terms_and_threshold_sweep(tmp_path, api):
    api["responses"]["term a"] = FakeResponse(200, _choice_response({"1": 0.95, "2": 0.03, "3": 0.01, "no_match": 0.01}))
    api["responses"]["term b"] = FakeResponse(200, _choice_response({"4": 0.05, "5": 0.6, "no_match": 0.35}))
    scores = _make_mapper(tmp_path).score_terms(_candidates())
    assert len(scores) == 7
    assert set(scores.loc[scores["source_id"] == "B", "concept_id"]) == {4, 5, -1}

    gold_file = tmp_path / "gs.csv"
    pd.DataFrame(
        {
            "source_concept_code": ["A", "B"],
            "target_concept_id": [1, 4],
            "predicate": ["exactMatch", "exactMatch"],
            "target_concept_id_b": [None, None],
            "predicate_b": [None, None],
        }
    ).to_csv(gold_file, index=False)

    sweep = sweep_decision_thresholds(scores, gold_file, match_thresholds=[0.9], prune_thresholds=[0.01, 0.1])

    assert sweep["n_terms"].tolist() == [2, 2]
    assert sweep["fraction_resolved"].tolist() == [0.5, 0.5]
    assert sweep["resolved_accuracy"].tolist() == [1.0, 1.0]
    assert sweep["gold_pruned_rate"].tolist() == [0.0, 1.0]
    assert sweep["mean_llm_candidates"].tolist() == [2.0, 1.0]
