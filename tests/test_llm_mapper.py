import json

import pandas as pd

from ariadne.llm_mapping.llm_mapper import LlmMapper
from ariadne.utils.settings import LlmMapperSettings, ConceptContextSettings


def _make_settings(responses_folder):
    return LlmMapperSettings(
        llm_mapper_responses_folder=responses_folder,
        context=ConceptContextSettings(
            include_target_parents=False,
            include_target_children=False,
            include_target_synonyms=False,
            include_target_domain=False,
            include_target_class=False,
            include_target_vocabulary=False,
            re_insert_target_details=False,
        ),
        system_prompts=["step1", "step2"],
    )


def _target_concepts_df():
    return pd.DataFrame(
        {
            "matched_concept_id": [111, 222],
            "matched_concept_name": ["Wrong concept", "Exact concept"],
        }
    )


def test_map_term_requests_structured_output_on_final_step(tmp_path, monkeypatch):
    mapper = LlmMapper(settings=_make_settings(str(tmp_path)))
    calls = []

    def fake_get_llm_response(prompt, system_prompt, show_reasoning=False, json_schema=None, **kwargs):
        calls.append({"prompt": prompt, "system_prompt": system_prompt, "json_schema": json_schema})
        if json_schema is None:
            return {
                "content": "intermediate response",
                "parsed_json": None,
                "usage": {"total_cost_usd": 0.2},
            }
        return {
            "content": json.dumps(
                {
                    "source_term": "Acute myocardial infarction",
                    "match_found": True,
                    "concept_id": 222,
                    "justification": "Exact concept match",
                }
            ),
            "parsed_json": {
                "source_term": "Acute myocardial infarction",
                "match_found": True,
                "concept_id": 222,
                "justification": "Exact concept match",
            },
            "usage": {"total_cost_usd": 0.3},
        }

    monkeypatch.setattr("ariadne.llm_mapping.llm_mapper.get_llm_response", fake_get_llm_response)

    mapped_id, mapped_name, rationale = mapper.map_term(
        "Acute myocardial infarction",
        source_id="42",
        target_concepts=_target_concepts_df(),
    )

    assert mapped_id == 222
    assert mapped_name == "Exact concept"
    assert rationale == "Exact concept match"
    assert len(calls) == 2
    assert calls[0]["json_schema"] is None
    assert calls[1]["json_schema"] is not None


def test_map_term_uses_cached_responses_without_api_call(tmp_path, monkeypatch):
    mapper = LlmMapper(settings=_make_settings(str(tmp_path)))

    step1 = tmp_path / "response_99_s1.txt"
    step2 = tmp_path / "response_99_s2.txt"
    step1.write_text("intermediate response", encoding="utf-8")
    step2.write_text(
        json.dumps(
            {
                "source_term": "Acute myocardial infarction",
                "match_found": False,
                "concept_id": None,
                "justification": "No exact equivalent",
            }
        ),
        encoding="utf-8",
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM should not be called when cache files are present")

    monkeypatch.setattr("ariadne.llm_mapping.llm_mapper.get_llm_response", fail_if_called)

    mapped_id, mapped_name, rationale = mapper.map_term(
        "Acute myocardial infarction",
        source_id="99",
        target_concepts=_target_concepts_df(),
    )

    assert mapped_id == -1
    assert mapped_name == "no_match"
    assert rationale == "No exact equivalent"

