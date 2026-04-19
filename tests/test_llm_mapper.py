import json

import pandas as pd

from ariadne.llm_mapping.llm_mapper import LlmMapper
from ariadne.utils.settings import LlmMapperSettings, ConceptContextSettings


def _make_settings(responses_folder, re_insert_target_details=False):
    return LlmMapperSettings(
        llm_mapper_responses_folder=responses_folder,
        context=ConceptContextSettings(
            include_target_parents=False,
            include_target_children=False,
            include_target_synonyms=False,
            include_target_domain=False,
            include_target_class=False,
            include_target_vocabulary=False,
            re_insert_source_target_details=re_insert_target_details,
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


def test_map_term_multiple_targets_drops_no_match_sentinel(tmp_path, monkeypatch):
    mapper = LlmMapper(settings=_make_settings(str(tmp_path)))

    def fake_get_llm_response(prompt, system_prompt, show_reasoning=False, json_schema=None, **kwargs):
        if json_schema is None:
            return {
                "content": "intermediate response",
                "parsed_json": None,
                "usage": {"total_cost_usd": 0.1},
            }
        return {
            "content": json.dumps(
                {
                    "source_term": "Acute myocardial infarction",
                    "match_found": True,
                    "concept_ids": [222, -1, 111],
                    "justification": "Requires combination",
                }
            ),
            "parsed_json": {
                "source_term": "Acute myocardial infarction",
                "match_found": True,
                "concept_ids": [222, -1, 111],
                "justification": "Requires combination",
            },
            "usage": {"total_cost_usd": 0.2},
        }

    monkeypatch.setattr("ariadne.llm_mapping.llm_mapper.get_llm_response", fake_get_llm_response)

    mapped_id, mapped_name, rationale = mapper.map_term(
        "Acute myocardial infarction",
        source_id="43",
        target_concepts=_target_concepts_df(),
        allow_multiple_targets=True,
    )

    assert mapped_id == [222, 111]
    assert mapped_name == ["Exact concept", "Wrong concept"]
    assert rationale == "Requires combination"


def test_map_terms_multiple_targets_duplicates_output_rows(tmp_path, monkeypatch):
    mapper = LlmMapper(settings=_make_settings(str(tmp_path)))
    source_target_concepts = pd.DataFrame(
        {
            "source_code": ["42", "42"],
            "source_term": ["Acute myocardial infarction", "Acute myocardial infarction"],
            "cleaned_term": ["Acute myocardial infarction", "Acute myocardial infarction"],
            "source_category": ["inpatient", "inpatient"],
            "matched_concept_id": [111, 222],
            "matched_concept_name": ["Wrong concept", "Exact concept"],
        }
    )

    def fake_map_term(*args, **kwargs):
        return [111, 222], ["Wrong concept", "Exact concept"], "Two concepts needed"

    monkeypatch.setattr(mapper, "map_term", fake_map_term)

    mapped = mapper.map_terms(
        source_target_concepts=source_target_concepts,
        allow_multiple_targets=True,
        source_context_columns=["source_category"],
    )

    assert len(mapped) == 2
    assert list(mapped["mapped_concept_id"]) == [111, 222]
    assert list(mapped["mapped_concept_name"]) == ["Wrong concept", "Exact concept"]
    assert set(mapped["mapped_rationale"]) == {"Two concepts needed"}
    assert list(mapped["source_category"]) == ["inpatient", "inpatient"]


def test_map_term_adds_source_context_to_step0_and_next_step_prompt(tmp_path, monkeypatch):
    mapper = LlmMapper(settings=_make_settings(str(tmp_path), re_insert_target_details=True))
    calls = []

    def fake_get_llm_response(prompt, system_prompt, show_reasoning=False, json_schema=None, **kwargs):
        calls.append({"prompt": prompt, "system_prompt": system_prompt, "json_schema": json_schema})
        if json_schema is None:
            return {
                "content": json.dumps(
                    {
                        "source_term": "Acute myocardial infarction",
                        "target_concepts": [{"id": 222}],
                    }
                ),
                "parsed_json": None,
                "usage": {"total_cost_usd": 0.1},
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
            "usage": {"total_cost_usd": 0.2},
        }

    monkeypatch.setattr("ariadne.llm_mapping.llm_mapper.get_llm_response", fake_get_llm_response)

    mapped_id, mapped_name, rationale = mapper.map_term(
        "Acute myocardial infarction",
        source_id="44",
        target_concepts=_target_concepts_df(),
        source_context={"encounter_type": "ER", "patient_age": 70},
    )

    assert mapped_id == 222
    assert mapped_name == "Exact concept"
    assert rationale == "Exact concept match"
    assert "Source details:" in calls[0]["prompt"]
    assert '"encounter_type": "ER"' in calls[0]["prompt"]

    step2_prompt = json.loads(calls[1]["prompt"])
    assert step2_prompt["source_details"]["source_term"] == "Acute myocardial infarction"
    assert step2_prompt["source_details"]["encounter_type"] == "ER"
    assert step2_prompt["source_details"]["patient_age"] == 70


def test_map_terms_passes_source_context_columns_to_map_term(tmp_path, monkeypatch):
    mapper = LlmMapper(settings=_make_settings(str(tmp_path)))
    source_target_concepts = pd.DataFrame(
        {
            "source_code": ["42", "42"],
            "source_term": ["Acute myocardial infarction", "Acute myocardial infarction"],
            "cleaned_term": ["Acute myocardial infarction", "Acute myocardial infarction"],
            "source_category": ["inpatient", "inpatient"],
            "age_band": ["65+", "65+"],
            "matched_concept_id": [111, 222],
            "matched_concept_name": ["Wrong concept", "Exact concept"],
        }
    )
    captured_kwargs = {}

    def fake_map_term(source_term, source_id, target_concepts, **kwargs):
        captured_kwargs.update(kwargs)
        return 222, "Exact concept", "Test rationale"

    monkeypatch.setattr(mapper, "map_term", fake_map_term)

    mapped = mapper.map_terms(
        source_target_concepts=source_target_concepts,
        source_context_columns=["source_category", "age_band"],
    )

    assert len(mapped) == 1
    assert captured_kwargs["source_context"] == {"source_category": "inpatient", "age_band": "65+"}
    assert "source_category" in mapped.columns
    assert "age_band" in mapped.columns
    assert mapped.iloc[0]["source_category"] == "inpatient"
    assert mapped.iloc[0]["age_band"] == "65+"


def test_map_terms_groups_by_source_id_when_source_id_column_is_provided(tmp_path, monkeypatch):
    mapper = LlmMapper(settings=_make_settings(str(tmp_path)))
    source_target_concepts = pd.DataFrame(
        {
            "source_code": ["42", "42"],
            "source_term": ["Acute myocardial infarction", "Heart attack"],
            "cleaned_term": ["acute mi", "heart attack"],
            "matched_concept_id": [111, 222],
            "matched_concept_name": ["Wrong concept", "Exact concept"],
        }
    )

    calls = []

    def fake_map_term(source_term, source_id, target_concepts, **kwargs):
        calls.append(
            {
                "source_term": source_term,
                "source_id": source_id,
                "candidate_count": len(target_concepts),
            }
        )
        return 222, "Exact concept", "Test rationale"

    monkeypatch.setattr(mapper, "map_term", fake_map_term)

    mapped = mapper.map_terms(source_target_concepts=source_target_concepts)

    assert len(calls) == 1
    assert calls[0]["source_id"] == "42"
    assert calls[0]["source_term"] == "acute mi"
    assert calls[0]["candidate_count"] == 2
    assert len(mapped) == 1


def test_map_terms_groups_by_term_when_source_id_column_is_none(tmp_path, monkeypatch):
    mapper = LlmMapper(settings=_make_settings(str(tmp_path)))
    source_target_concepts = pd.DataFrame(
        {
            "source_term": ["Acute myocardial infarction", "Heart attack"],
            "cleaned_term": ["acute mi", "heart attack"],
            "matched_concept_id": [111, 222],
            "matched_concept_name": ["Wrong concept", "Exact concept"],
        }
    )

    calls = []

    def fake_map_term(source_term, source_id, target_concepts, **kwargs):
        calls.append(
            {
                "source_term": source_term,
                "source_id": source_id,
                "candidate_count": len(target_concepts),
            }
        )
        return 222, "Exact concept", "Test rationale"

    monkeypatch.setattr(mapper, "map_term", fake_map_term)

    mapped = mapper.map_terms(
        source_target_concepts=source_target_concepts,
        source_id_column=None,
    )

    assert len(calls) == 2
    assert all(call["source_id"] is None for call in calls)
    assert {call["source_term"] for call in calls} == {"acute mi", "heart attack"}
    assert all(call["candidate_count"] == 1 for call in calls)
    assert len(mapped) == 2
