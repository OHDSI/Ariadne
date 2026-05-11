from types import SimpleNamespace
from typing import cast

import pytest

import pandas as pd

import ariadne.hierarchy.pipeline as pipeline
from ariadne.utils.settings import HierarchySettings


def test_build_selection_prompt_returns_empty_for_empty_candidates():
    candidates_df = pd.DataFrame(columns=pipeline.CANDIDATE_COLUMNS)

    prompt = pipeline._build_selection_prompt(candidates_df)

    assert prompt == ""


def test_build_selection_prompt_falls_back_to_attribute_category():
    candidates_df = pd.DataFrame(
        [
            {
                "concept_id": 123,
                "concept_name": "Bone structure",
                "attribute_category": "Has finding site (SNOMED)",
                "extracted_mention": "bone",
                "similarity": 0.91,
            }
        ]
    )

    prompt = pipeline._build_selection_prompt(candidates_df)

    assert "finding_site" in prompt
    assert "Bone structure" in prompt


def test_find_attributes_two_stage_skips_selection_when_no_candidates(monkeypatch):
    cfg = SimpleNamespace(
        prompts=SimpleNamespace(selection="unused"),
        selection="unused",
    )

    monkeypatch.setattr(
        pipeline,
        "_retrieve_reference_examples",
        lambda *args, **kwargs: ([], "", 0.0),
    )
    monkeypatch.setattr(
        pipeline,
        "extract_components",
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

    result = pipeline.find_attributes_two_stage(
        "Late effect of rickets",
        attribute_searcher=cast(pipeline.AttributeSearcher, object()),
        reference_searcher=None,
        cfg=cast(HierarchySettings, cfg),
        verbose=False,
    )

    assert result["attributes"] == {}
    assert result["retrieved_candidates"] == []
    assert result["cost"]["selection_cost"] == 0.0
    assert result["cost"]["total_cost"] == pytest.approx(0.3)




