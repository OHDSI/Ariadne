from types import SimpleNamespace
from typing import cast

import pytest

import pandas as pd

import ariadne.hierarchy.llm_attribute_extractor as pipeline
from ariadne.utils.settings import HierarchySettings


def test_build_selection_prompt_returns_empty_for_empty_candidates():
    candidates_df = pd.DataFrame(columns=pipeline.CANDIDATE_COLUMNS)

    prompt = pipeline._build_selection_prompt(candidates_df)

    assert prompt == ""


def test_build_selection_prompt_groups_by_attribute_key():
    candidates_df = pd.DataFrame(
        [
            {
                "concept_id": 123,
                "concept_name": "Bone structure",
                "attribute_category": "Has finding site (SNOMED)",
                "attribute_key": "finding_site",
                "extracted_mention": "bone",
                "similarity": 0.91,
            }
        ]
    )

    prompt = pipeline._build_selection_prompt(candidates_df)

    assert "finding_site" in prompt
    assert "Bone structure" in prompt


def test_extract_attributes_skips_selection_when_no_candidates(monkeypatch):
    cfg = SimpleNamespace(
        prompts=SimpleNamespace(selection="unused", extraction="unused"),
        retrieval=SimpleNamespace(num_reference_examples=5),
    )

    fake_reference_searcher = SimpleNamespace(
        search=lambda *args, **kwargs: ([], 0.0),
    )
    monkeypatch.setattr(
        pipeline,
        "infer_attributes",
        lambda *args, **kwargs: ({"finding_site": "bone"}, 0.1),
    )
    monkeypatch.setattr(
        pipeline,
        "_retrieve_candidates",
        lambda *args, **kwargs: (pd.DataFrame(columns=pipeline.CANDIDATE_COLUMNS), 0.2),
    )

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("selection LLM should not be called when there are no candidates")

    monkeypatch.setattr(pipeline, "call_llm", _raise_if_called)

    result = pipeline.extract_attributes(
        "Late effect of rickets",
        attribute_searcher=cast(pipeline.AttributeSearcher, object()),
        reference_searcher=cast(pipeline.ReferenceSearcher, fake_reference_searcher),
        hierarchy_settings=cast(HierarchySettings, cfg),
    )

    assert result["attributes"] == {}
    assert result["retrieved_candidates"] == []
    assert result["cost"]["selection_cost"] == 0.0
    assert result["cost"]["total_cost"] == pytest.approx(0.3)




