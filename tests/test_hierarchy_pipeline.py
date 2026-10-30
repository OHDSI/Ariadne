from types import SimpleNamespace
from typing import cast
from dataclasses import replace
from pathlib import Path
import json

import pytest
import yaml

import pandas as pd
from conftest import make_hierarchy_settings, make_mrcm_settings

import ariadne.hierarchy.llm_attribute_extractor as pipeline
from ariadne.utils.settings import HierarchySettings, PromptsConfig, RoleGroupingConfig, build_dataclass


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


def _concept(concept_id, name=None, code=None):
    return {"concept_id": str(concept_id), "concept_name": name or f"c{concept_id}", "concept_code": code}


def _candidates(*rows):
    return pd.DataFrame(
        [
            {"concept_id": str(cid), "concept_code": f"code{cid}", "concept_name": f"c{cid}",
             "attribute_category": "unused", "similarity": 0.9, "extracted_mention": "m", "attribute_key": key}
            for key, cid in rows
        ],
        columns=pipeline.CANDIDATE_COLUMNS,
    )


def test_drop_non_candidate_values_removes_values_outside_their_attribute_candidates():
    candidates = _candidates(("finding_site", 1), ("associated_morphology", 2), ("interprets", 3), ("interpretation", 4))
    result = {
        "attributes": {
            "finding_site": [_concept(1), _concept(9, "Copied from reference")],
            "associated_morphology": _concept(2),
            # Candidate for finding_site, but not offered for due_to.
            "due_to": _concept(1),
            "severity": None,
            "interprets_interpretation": [
                {"interprets": _concept(3), "interpretation": _concept(4)},
                {"interprets": _concept(3), "interpretation": _concept(8)},
            ],
        },
        "llm_role_groups": [
            {"finding_site": _concept(1), "associated_morphology": _concept(2)},
            {"finding_site": _concept(9)},
        ],
        "role_groups": [
            {"group": 1, "members": [{"attribute": "finding_site", **_concept(1)}, {"attribute": "finding_site", **_concept(9)}]},
            {"group": 2, "members": [{"attribute": "finding_site", **_concept(9)}]},
        ],
    }

    dropped = pipeline.drop_non_candidate_values(result, candidates)

    assert {(d["attribute"], d["concept_id"]) for d in dropped} == {
        ("finding_site", "9"), ("due_to", "1"), ("interpretation", "8")
    }
    attrs = result["attributes"]
    assert [c["concept_id"] for c in attrs["finding_site"]] == ["1"]
    assert attrs["associated_morphology"]["concept_id"] == "2"
    assert attrs["due_to"] is None
    assert attrs["severity"] is None
    assert attrs["interprets_interpretation"] == [{"interprets": _concept(3), "interpretation": _concept(4)}]
    assert result["llm_role_groups"] == [{"finding_site": _concept(1), "associated_morphology": _concept(2)}]
    assert [[m["concept_id"] for m in g["members"]] for g in result["role_groups"]] == [["1"]]
    assert result["dropped_selections"] == dropped

    # Re-applying (e.g. to a cached result) changes nothing.
    assert pipeline.drop_non_candidate_values(result, candidates) == []
    assert result["dropped_selections"] == dropped


def test_drop_non_candidate_values_accepts_retrieved_candidates_records_and_alias_keys():
    records = _candidates(("occurrence", 5)).to_dict("records")
    result = {"attributes": {"has_occurrence": _concept(5)}, "llm_role_groups": [{"finding_site": _concept(7)}]}

    dropped = pipeline.drop_non_candidate_values(result, records)

    assert result["attributes"]["has_occurrence"]["concept_id"] == "5"
    assert "llm_role_groups" not in result
    assert [d["concept_id"] for d in dropped] == ["7"]


def test_extract_attributes_drops_selection_not_in_candidates(monkeypatch):
    cfg = replace(
        make_hierarchy_settings(),
        prompts=PromptsConfig(selection="{reference_section}", extraction="unused"),
        mrcm=replace(make_mrcm_settings(), grouping=RoleGroupingConfig()),
    )
    reference_examples = [{
        "concept_id": 50, "concept_name": "Reference term", "similarity": 0.9,
        "attributes": [{"target_concept_id": 9, "target_concept_code": "code9",
                        "target_concept_name": "Copied site", "attribute_category": "Has finding site"}],
    }]
    monkeypatch.setattr(pipeline, "infer_attributes", lambda *a, **k: ({"finding_site": "bone"}, 0.1))
    monkeypatch.setattr(pipeline, "_retrieve_candidates", lambda *a, **k: (_candidates(("finding_site", 1)), 0.2))
    selection = {"attributes": {"finding_site": [
        {"concept_id": 1, "concept_name": "c1"}, {"concept_id": 9, "concept_name": "Copied site"},
    ]}}
    monkeypatch.setattr(pipeline, "call_llm", lambda *a, **k: (json.dumps(selection), 0.3))

    result = pipeline.extract_attributes(
        "Some term",
        attribute_searcher=cast(pipeline.AttributeSearcher, object()),
        reference_searcher=cast(pipeline.ReferenceSearcher, SimpleNamespace(search=lambda *a, **k: (reference_examples, 0.0))),
        hierarchy_settings=cast(HierarchySettings, cfg),
    )

    assert result["attributes"]["finding_site"] == [{"concept_id": 1, "concept_name": "c1", "concept_code": "code1"}]
    assert result["dropped_selections"] == [{"attribute": "finding_site", "concept_id": "9", "concept_name": "Copied site"}]


def test_reference_examples_name_every_configured_relationship_by_its_attribute_key():
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "config_condition_mapping.yaml").read_text())
    hierarchy = build_dataclass(HierarchySettings, config["hierarchy"])
    key_for = hierarchy.attribute_by_relationship()
    for relationship in hierarchy.snomed_relationships:
        example = [{"concept_id": 1, "concept_name": "Reference", "similarity": 0.9, "attributes": [
            {"target_concept_id": 2, "target_concept_name": "Value", "attribute_category": relationship}]}]
        text = pipeline.format_reference_examples(example, hierarchy, include_concept_ids=True)
        assert f"  - {key_for[relationship]}: Value (2)" in text, relationship


def test_format_reference_examples_uses_mrcm_attribute_keys():
    example = [{"concept_id": 1, "concept_name": "Ref", "similarity": 0.9, "attributes": [
        {"target_concept_id": 2, "target_concept_name": "Pathological developmental process",
         "attribute_category": "Has pathology"},
        {"target_concept_id": 3, "target_concept_name": "X", "attribute_category": "Not configured"},
    ]}]
    text = pipeline.format_reference_examples(example, make_hierarchy_settings())
    assert "  - pathological_process: Pathological developmental process" in text
    assert "  - Not configured: X" in text




