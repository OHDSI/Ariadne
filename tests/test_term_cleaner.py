import json
import threading

import pandas as pd
import pytest

from ariadne.term_cleanup.term_cleaner import TermCleaner
from ariadne.utils.settings import TermCleanerSettings


def _make_settings() -> TermCleanerSettings:
    return TermCleanerSettings(system_prompt="test prompt")


def test_clean_terms_batch_uses_structured_output(monkeypatch):
    cleaner = TermCleaner(settings=_make_settings())

    def fake_get_llm_response(**kwargs):
        assert kwargs["json_schema"] is not None
        return {
            "content": '{"results":[{"row_number":0,"cleaned_term":"synovitis"}]}',
            "parsed_json": {
                "results": [
                    {"row_number": 0, "cleaned_term": "synovitis"}
                ]
            },
            "usage": {"total_cost_usd": 0.01},
        }

    monkeypatch.setattr("ariadne.term_cleanup.term_cleaner.get_llm_response", fake_get_llm_response)

    cleaned = cleaner._clean_terms_batch(["Unspecified synovitis"])

    assert cleaned == ["synovitis"]
    assert cleaner.get_total_cost() == 0.01



def test_clean_terms_dataframe(monkeypatch):
    cleaner = TermCleaner(settings=_make_settings())
    call_sizes = []

    def fake_get_llm_response(**kwargs):
        prompt = kwargs["prompt"]
        payload = json.loads(prompt.split("Input JSON:\n", 1)[1])
        terms = payload["terms"]
        call_sizes.append(len(terms))
        results = []
        for row in terms:
            results.append(
                {
                    "row_number": row["row_number"],
                    "cleaned_term": row["source_term"].replace("unspecified ", ""),
                }
            )
        return {
            "content": "",
            "parsed_json": {"results": results},
            "usage": {"total_cost_usd": 0.0},
        }

    monkeypatch.setattr("ariadne.term_cleanup.term_cleaner.get_llm_response", fake_get_llm_response)

    df = pd.DataFrame(
        {
            "source_term": [f"unspecified term {i}" for i in range(30)],
            "source_vocabulary_id": ["SNOMED"] * 30,
            "source_concept_code": [str(i) for i in range(30)],
        }
    )
    result = cleaner.clean_terms(df.copy())

    assert result["cleaned_term"].tolist() == [f"term {i}" for i in range(30)]
    assert sorted(call_sizes) == [5, 25]


def test_clean_terms_runs_batches_concurrently_and_keeps_row_order(monkeypatch):
    cleaner = TermCleaner(settings=TermCleanerSettings(system_prompt="test prompt", max_workers=3))
    assert cleaner.max_workers == 3
    # All three batches must be in flight at once to pass the barrier.
    barrier = threading.Barrier(3, timeout=5)
    thread_ids = set()

    def fake_get_llm_response(**kwargs):
        terms = json.loads(kwargs["prompt"].split("Input JSON:\n", 1)[1])["terms"]
        thread_ids.add(threading.get_ident())
        barrier.wait()
        results = [{"row_number": r["row_number"], "cleaned_term": r["source_term"].upper()} for r in terms]
        return {"content": "", "parsed_json": {"results": results}, "usage": {"total_cost_usd": 0.5}}

    monkeypatch.setattr("ariadne.term_cleanup.term_cleaner.get_llm_response", fake_get_llm_response)
    df = pd.DataFrame(
        {
            "source_term": [f"term {i}" for i in range(60)],
            "source_vocabulary_id": ["SNOMED"] * 60,
            "source_concept_code": [str(i) for i in range(60)],
        },
        index=range(100, 160),
    )
    result = cleaner.clean_terms(df)

    assert result["cleaned_term"].tolist() == [f"TERM {i}" for i in range(60)]
    assert len(thread_ids) == 3
    assert cleaner.get_total_cost() == pytest.approx(1.5)


def test_term_cleaner_default_max_workers_is_parallel():
    assert TermCleanerSettings().max_workers > 1


def test_clean_terms_raises_on_missing_required_columns():
    cleaner = TermCleaner(settings=_make_settings())

    df = pd.DataFrame({"source_term": ["unspecified synovitis"]})

    with pytest.raises(ValueError, match="source_vocabulary_id"):
        cleaner.clean_terms(df)



def test_clean_terms_batch_raises_on_malformed_structured_response(monkeypatch):
    cleaner = TermCleaner(settings=_make_settings())

    def fake_get_llm_response(**kwargs):
        return {
            "content": "",
            "parsed_json": {"results": [{"row_number": 0}]},
            "usage": {"total_cost_usd": 0.0},
        }

    monkeypatch.setattr("ariadne.term_cleanup.term_cleaner.get_llm_response", fake_get_llm_response)

    with pytest.raises(ValueError, match="cleaned_term"):
        cleaner._clean_terms_batch(["Unspecified synovitis"])

