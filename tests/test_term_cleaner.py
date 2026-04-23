import json

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

    df = pd.DataFrame({"source_term": [f"unspecified term {i}" for i in range(30)]})
    result = cleaner.clean_terms(df.copy())

    assert result["cleaned_term"].tolist() == [f"term {i}" for i in range(30)]
    assert call_sizes == [25, 5]


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

