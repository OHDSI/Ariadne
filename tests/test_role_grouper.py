"""Tests for role grouping and its use downstream (evaluator rows, RF2 export, extractor)."""

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest
from conftest import concept, make_attributes, make_hierarchy_settings, make_mrcm_settings

import ariadne.hierarchy.llm_attribute_extractor as pipeline
from ariadne.evaluation.hierarchy_attribute_evaluator import build_prediction_rows
from ariadne.hierarchy.mrcm import MrcmRules
from ariadne.hierarchy.rf2_exporter import _build_owl_expression, export_to_rf2
from ariadne.hierarchy.role_grouper import (
    METHOD_DETERMINISTIC,
    METHOD_LLM,
    METHOD_NONE,
    METHOD_POSITIONAL,
    flatten_role_groups,
    group_attributes,
    regroup_results,
)
from ariadne.utils.settings import HierarchySettings, RoleGroupingConfig

SETTINGS = make_mrcm_settings()
HIERARCHY = make_hierarchy_settings()


def _groups(role_groups):
    """Role groups as {group: {(attribute, concept_id), ...}} for compact assertions."""
    return {g["group"]: {(m["attribute"], m["concept_id"]) for m in g["members"]} for g in role_groups}


def _group_sets(role_groups):
    return sorted(sorted(members) for members in _groups(role_groups).values())


def test_single_values_share_one_core_group_and_self_grouped_are_alone():
    attributes = {
        "finding_site": [concept(1)],
        "associated_morphology": [concept(2)],
        "occurrence": concept(3),                 # legacy single-dict format
        "clinical_course": [concept(4)],
        "due_to": [concept(5)],
        "severity": None,
    }
    result = group_attributes(attributes, SETTINGS)

    assert result.method == METHOD_DETERMINISTIC
    assert _groups(result.role_groups) == {
        1: {("finding_site", 1), ("associated_morphology", 2), ("occurrence", 3)},
        2: {("due_to", 5)},
        3: {("clinical_course", 4)},
    }


def test_one_multi_valued_attribute_is_split_and_singles_repeated():
    attributes = {"finding_site": [concept(1), concept(2)], "associated_morphology": [concept(3)]}
    result = group_attributes(attributes, SETTINGS)

    assert result.method == METHOD_DETERMINISTIC
    assert _group_sets(result.role_groups) == [
        [("associated_morphology", 3), ("finding_site", 1)],
        [("associated_morphology", 3), ("finding_site", 2)],
    ]


def test_several_multi_valued_attributes_use_positional_fallback():
    attributes = {
        "finding_site": [concept(1), concept(2)],
        "associated_morphology": [concept(3), concept(4)],
        "occurrence": [concept(5)],
    }
    result = group_attributes(attributes, SETTINGS)

    assert result.method == METHOD_POSITIONAL
    assert _group_sets(result.role_groups) == [
        [("associated_morphology", 3), ("finding_site", 1), ("occurrence", 5)],
        [("associated_morphology", 4), ("finding_site", 2), ("occurrence", 5)],
    ]


def test_interprets_pairs_become_groups_and_orphans_are_kept():
    attributes = {
        "interprets_interpretation": [
            {"interprets": concept(1), "interpretation": concept(2)},
            {"interprets": concept(3), "interpretation": concept(4)},
            {"interpretation": concept(5)},
        ],
    }
    result = group_attributes(attributes, SETTINGS)

    assert _group_sets(result.role_groups) == [
        [("interpretation", 2), ("interprets", 1)],
        [("interpretation", 4), ("interprets", 3)],
        [("interpretation", 5)],
    ]


def test_context_attributes_share_one_group():
    attributes = {
        "finding_context": [concept(1)],
        "temporal_context": [concept(2)],
        "subject_relationship_context": [concept(3)],
    }
    result = group_attributes(attributes, SETTINGS)
    assert len(result.role_groups) == 1


def test_llm_groups_take_precedence_and_uncovered_single_value_is_repeated():
    llm_groups = [
        {"finding_site": concept(1), "associated_morphology": concept(3)},
        {"finding_site": [concept(2)], "associated_morphology": concept(3)},
    ]
    attributes = {
        "finding_site": [concept(1), concept(2)],
        "associated_morphology": [concept(3)],
        "occurrence": [concept(5)],           # selected flat, not in any LLM group
        "clinical_course": [concept(6)],
    }
    result = group_attributes(attributes, SETTINGS, llm_groups)

    assert result.method == METHOD_LLM
    assert _group_sets(result.role_groups) == [
        [("associated_morphology", 3), ("finding_site", 1), ("occurrence", 5)],
        [("associated_morphology", 3), ("finding_site", 2), ("occurrence", 5)],
        [("clinical_course", 6)],
    ]


def test_llm_groups_ignore_non_cluster_attributes():
    llm_groups = [{"finding_site": concept(1), "severity": concept(2)}]
    result = group_attributes({}, SETTINGS, llm_groups)
    assert _group_sets(result.role_groups) == [[("finding_site", 1)], [("severity", 2)]]


def test_llm_group_with_several_values_for_one_attribute_is_split():
    llm_groups = [{"finding_site": concept(1), "associated_morphology": [concept(3), concept(4)]}]
    result = group_attributes({}, SETTINGS, llm_groups)

    assert result.method == METHOD_LLM
    assert _group_sets(result.role_groups) == [
        [("associated_morphology", 3), ("finding_site", 1)],
        [("associated_morphology", 4), ("finding_site", 1)],
    ]


def test_llm_group_with_several_multi_valued_attributes_is_flagged_positional():
    llm_groups = [{"finding_site": [concept(1), concept(2)], "associated_morphology": [concept(3), concept(4)]}]
    result = group_attributes({}, SETTINGS, llm_groups)

    assert result.method == METHOD_POSITIONAL
    assert len(result.role_groups) == 2


def test_duplicate_groups_are_removed():
    llm_groups = [{"finding_site": concept(1)}, {"finding_site": concept(1)}]
    result = group_attributes({}, SETTINGS, llm_groups)
    assert len(result.role_groups) == 1


def test_no_attributes():
    assert group_attributes({}, SETTINGS) == ([], METHOD_NONE)
    assert group_attributes(None, SETTINGS) == ([], METHOD_NONE)


def test_unconfigured_attribute_gets_its_own_group():
    result = group_attributes({"finding_site": [concept(1)], "mystery": [concept(2)]}, SETTINGS)
    assert _group_sets(result.role_groups) == [[("finding_site", 1)], [("mystery", 2)]]


def test_ungrouped_mrcm_attribute_goes_to_group_zero(mrcm_settings):
    rules = MrcmRules.load(mrcm_settings, make_attributes())
    result = group_attributes(
        {"finding_site": [concept(1)], "episodicity": [concept(2)]}, mrcm_settings, rules=rules,
        domain="clinical_finding",
    )
    assert _groups(result.role_groups) == {0: {("episodicity", 2)}, 1: {("finding_site", 1)}}


def test_flatten_role_groups_dedupes():
    flat = flatten_role_groups([{"finding_site": concept(1)}, {"finding_site": [concept(1), concept(2)]}])
    assert [c["concept_id"] for c in flat["finding_site"]] == [1, 2]


def test_regroup_results_skips_errors_and_sets_domain(mrcm_settings):
    rules = MrcmRules.load(mrcm_settings, make_attributes())
    results = [
        {"source_key": "a", "error": "boom"},
        {"source_key": "b", "attributes": {"finding_context": [concept(1)], "temporal_context": [concept(2)]}},
    ]
    regrouped = regroup_results(results, mrcm_settings, rules)

    assert regrouped[0] == results[0]
    assert regrouped[1]["mrcm_domain"] == "finding_with_explicit_context"
    assert regrouped[1]["grouping_method"] == METHOD_DETERMINISTIC
    assert "role_groups" not in results[1]  # input untouched


# ── Evaluator rows ────────────────────────────────────────────────────────────


def test_prediction_rows_carry_relationship_groups():
    result = {
        "source_key": "ICD10CM:J20",
        "source_term": "Acute tracheobronchitis",
        "attributes": {
            "finding_site": [concept(44567001), concept(955009)],
            "associated_morphology": [concept(4532008)],
            "clinical_course": [concept(424124008)],
        },
    }
    result["role_groups"] = group_attributes(result["attributes"], SETTINGS).role_groups

    rows = {(r["attribute_category"], r["predicted_target_concept_id"]): r["relationship_groups"]
            for r in build_prediction_rows([result], HIERARCHY)}
    assert rows == {
        ("Has finding site", 44567001): "1",
        ("Has finding site", 955009): "2",
        ("Has asso morph", 4532008): "1;2",
        ("Has clinical course", 424124008): "3",
    }


def test_prediction_rows_group_on_the_fly_only_when_grouping_is_configured():
    result = {"source_key": "x", "attributes": {"finding_site": [concept(1)], "due_to": [concept(2)]}}
    no_grouping = replace(HIERARCHY, mrcm=replace(SETTINGS, grouping=RoleGroupingConfig()))
    assert {r["relationship_groups"] for r in build_prediction_rows([result], no_grouping)} == {None}
    assert {r["relationship_groups"] for r in build_prediction_rows([result], HIERARCHY)} == {"1", "2"}


def test_prediction_rows_take_attribute_category_from_hierarchy_attributes():
    result = {"source_key": "x", "attributes": {
        "pathological_process": [concept(1)],
        "associated_with": [concept(2)],  # alias of finding_asso_with
        "not_configured": [concept(3)],
        "interprets_interpretation": [{"interprets": concept(4), "interpretation": concept(5)}],
    }}
    rows = {r["predicted_target_concept_id"]: r["attribute_category"] for r in build_prediction_rows([result], HIERARCHY)}
    assert rows == {1: "Has pathology", 2: "Finding asso with", 3: "not_configured",
                    4: "Has interprets", 5: "Has interpretation"}


# ── RF2 export ────────────────────────────────────────────────────────────────


def test_owl_expression_multiple_groups_and_group_zero():
    expr = _build_owl_expression(
        10, [404684003],
        [(1, 363698007, "44567001"), (1, 116676008, "4532008"), (2, 363698007, "955009"),
         (2, 116676008, "4532008"), (3, 263502005, "424124008"), (0, 246456000, "1")],
    )
    assert expr == (
        "EquivalentClasses(:10 ObjectIntersectionOf(:404684003 "
        "ObjectSomeValuesFrom(:246456000 :1) "
        "ObjectSomeValuesFrom(:609096000 ObjectIntersectionOf(ObjectSomeValuesFrom(:363698007 :44567001) "
        "ObjectSomeValuesFrom(:116676008 :4532008))) "
        "ObjectSomeValuesFrom(:609096000 ObjectIntersectionOf(ObjectSomeValuesFrom(:363698007 :955009) "
        "ObjectSomeValuesFrom(:116676008 :4532008))) "
        "ObjectSomeValuesFrom(:609096000 ObjectSomeValuesFrom(:263502005 :424124008))))"
    )


def _owl_lines(output_dir):
    owl_file = next((output_dir / "Delta" / "Refset" / "Content").glob("*OWLAxiom*"))
    return owl_file.read_text().splitlines()[1:]


def test_export_uses_relationship_groups(tmp_path):
    df = pd.DataFrame([
        {"source_concept_id": 1, "source_term": "t", "predicted_target_concept_code": "44567001",
         "attribute_category": "Has finding site", "relationship_groups": "1"},
        {"source_concept_id": 1, "source_term": "t", "predicted_target_concept_code": "955009",
         "attribute_category": "Has finding site", "relationship_groups": "2"},
        {"source_concept_id": 1, "source_term": "t", "predicted_target_concept_code": "4532008",
         "attribute_category": "Has asso morph", "relationship_groups": "1;2"},
        {"source_concept_id": 1, "source_term": "t", "predicted_target_concept_code": "1",
         "attribute_category": "Has due to", "relationship_groups": "3"},
    ])
    export_to_rf2(df, tmp_path, hierarchy_settings=HIERARCHY, date="20260301", zip_it=False)

    (line,) = _owl_lines(tmp_path)
    assert line.count(":609096000") == 3
    assert "ObjectSomeValuesFrom(:42752001 :1)" in line  # "Has due to" is now exportable


def test_export_without_relationship_groups_uses_rel_group(tmp_path):
    df = pd.DataFrame([
        {"source_concept_id": 1, "source_term": "t", "predicted_target_concept_code": "1",
         "attribute_category": "Has finding site"},
        {"source_concept_id": 1, "source_term": "t", "predicted_target_concept_code": "2",
         "attribute_category": "Has asso morph"},
    ])
    export_to_rf2(df, tmp_path / "grouped", hierarchy_settings=HIERARCHY, date="20260301", zip_it=False)
    export_to_rf2(df, tmp_path / "ungrouped", hierarchy_settings=HIERARCHY, date="20260301", zip_it=False, rel_group=0)

    assert _owl_lines(tmp_path / "grouped")[0].count(":609096000") == 1
    assert ":609096000" not in _owl_lines(tmp_path / "ungrouped")[0]


# ── Extractor integration ─────────────────────────────────────────────────────


def test_unpack_mentions_reads_role_groups_and_dedupes():
    mentions = pipeline._unpack_mentions({
        "role_groups": [
            {"associated_morphology": "Acute inflammation", "finding_site": "Tracheal structure"},
            {"associated_morphology": "Acute inflammation", "finding_site": "Bronchial structure"},
        ],
        "clinical_course": ["Acute"],
    }, HIERARCHY)
    assert mentions == [
        ("associated_morphology", "Acute inflammation", "Has asso morph"),
        ("finding_site", "Tracheal structure", "Has finding site"),
        ("finding_site", "Bronchial structure", "Has finding site"),
        ("clinical_course", "Acute", "Has clinical course"),
    ]


def test_merge_role_groups_into_attributes():
    result = {
        "role_groups": [{"finding_site": concept(1), "has_occurrence": concept(2)}],
        "attributes": {"finding_site": [concept(1)], "role_groups": [{"finding_site": concept(3)}]},
    }
    groups = pipeline._merge_role_groups_into_attributes(result)

    assert groups == [{"finding_site": concept(1), "occurrence": concept(2)}, {"finding_site": concept(3)}]
    assert [c["concept_id"] for c in result["attributes"]["finding_site"]] == [1, 3]
    assert result["attributes"]["occurrence"] == [concept(2)]
    assert "role_groups" not in result and "role_groups" not in result["attributes"]


def test_extract_attributes_returns_role_groups(monkeypatch, mrcm_settings):
    pipeline._MRCM_RULES_CACHE.clear()
    cfg = SimpleNamespace(
        prompts=SimpleNamespace(selection="S {grouping_section}", extraction="E {grouping_section}"),
        retrieval=SimpleNamespace(num_reference_examples=5),
        attributes=make_attributes(),
        mrcm=mrcm_settings,
    )
    candidates = pd.DataFrame([
        {"concept_id": "44567001", "concept_code": "44567001", "concept_name": "Tracheal structure",
         "attribute_category": "Has finding site", "similarity": 0.9, "extracted_mention": "trachea",
         "attribute_key": "finding_site"},
        {"concept_id": "955009", "concept_code": "955009", "concept_name": "Bronchial structure",
         "attribute_category": "Has finding site", "similarity": 0.9, "extracted_mention": "bronchus",
         "attribute_key": "finding_site"},
        {"concept_id": "4532008", "concept_code": "4532008", "concept_name": "Acute inflammation",
         "attribute_category": "Has asso morph", "similarity": 0.9, "extracted_mention": "acute inflammation",
         "attribute_key": "associated_morphology"},
    ])
    prompts_seen = []
    monkeypatch.setattr(pipeline, "_retrieve_candidates", lambda *a, **k: (candidates, 0.0))

    def fake_llm(system_prompt, user_prompt):
        prompts_seen.append((system_prompt, user_prompt))
        if len(prompts_seen) == 1:
            return pipeline.LlmResult(
                '{"role_groups": [{"finding_site": "trachea", "associated_morphology": "acute inflammation"},'
                ' {"finding_site": "bronchus", "associated_morphology": "acute inflammation"}]}', 0.0)
        return pipeline.LlmResult(
            '{"role_groups": ['
            '{"finding_site": {"concept_id": "44567001", "concept_name": "Tracheal structure"},'
            ' "associated_morphology": {"concept_id": "4532008", "concept_name": "Acute inflammation"}},'
            '{"finding_site": {"concept_id": "955009", "concept_name": "Bronchial structure"},'
            ' "associated_morphology": {"concept_id": "4532008", "concept_name": "Acute inflammation"}}],'
            ' "attributes": {"clinical_course": null}}', 0.0)

    monkeypatch.setattr(pipeline, "call_llm", fake_llm)
    result = pipeline.extract_attributes(
        "Acute tracheobronchitis",
        attribute_searcher=cast(pipeline.AttributeSearcher, object()),
        reference_searcher=cast(pipeline.ReferenceSearcher, SimpleNamespace(search=lambda *a, **k: ([], 0.0))),
        hierarchy_settings=cast(HierarchySettings, cfg),
    )

    assert "{grouping_section}" not in prompts_seen[0][0] and "role_groups" in prompts_seen[0][0]
    assert "Proposed role groups" in prompts_seen[1][1]
    assert result["grouping_method"] == METHOD_LLM
    assert result["mrcm_domain"] == "clinical_finding"
    assert _group_sets(result["role_groups"]) == [
        [("associated_morphology", "4532008"), ("finding_site", "44567001")],
        [("associated_morphology", "4532008"), ("finding_site", "955009")],
    ]
    # Flat attributes still available for evaluation, with injected codes.
    assert sorted(c["concept_id"] for c in result["attributes"]["finding_site"]) == ["44567001", "955009"]
    assert all(m["concept_code"] for g in result["role_groups"] for m in g["members"])


def test_extract_attributes_without_mrcm_files_still_groups(monkeypatch, tmp_path):
    pipeline._MRCM_RULES_CACHE.clear()
    cfg = SimpleNamespace(
        prompts=SimpleNamespace(selection="S", extraction="E"),
        retrieval=SimpleNamespace(num_reference_examples=5),
        mrcm=make_mrcm_settings(tmp_path / "missing"),
    )
    candidates = pd.DataFrame([
        {"concept_id": "1", "concept_code": "1", "concept_name": "Bone", "attribute_category": "Has finding site",
         "similarity": 0.9, "extracted_mention": "bone", "attribute_key": "finding_site"},
    ])
    monkeypatch.setattr(pipeline, "_retrieve_candidates", lambda *a, **k: (candidates, 0.0))
    monkeypatch.setattr(pipeline, "infer_attributes", lambda *a, **k: ({"finding_site": ["bone"]}, 0.0))
    monkeypatch.setattr(pipeline, "call_llm", lambda *a: pipeline.LlmResult(
        '{"attributes": {"finding_site": [{"concept_id": "1", "concept_name": "Bone"}]}}', 0.0))

    result = pipeline.extract_attributes(
        "Bone disorder",
        attribute_searcher=cast(pipeline.AttributeSearcher, object()),
        reference_searcher=cast(pipeline.ReferenceSearcher, SimpleNamespace(search=lambda *a, **k: ([], 0.0))),
        hierarchy_settings=cast(HierarchySettings, cfg),
    )
    assert "mrcm_domain" not in result
    assert result["grouping_method"] == METHOD_DETERMINISTIC
    assert _group_sets(result["role_groups"]) == [[("finding_site", "1")]]


@pytest.fixture(autouse=True)
def _clear_rules_cache():
    yield
    pipeline._MRCM_RULES_CACHE.clear()


def test_extract_attributes_without_mrcm_config_does_not_group(monkeypatch):
    from ariadne.utils.settings import MrcmSettings

    cfg = SimpleNamespace(
        prompts=SimpleNamespace(selection="S {grouping_section}", extraction="E"),
        retrieval=SimpleNamespace(num_reference_examples=5),
        mrcm=MrcmSettings(),
    )
    candidates = pd.DataFrame([
        {"concept_id": "1", "concept_code": "1", "concept_name": "Bone", "attribute_category": "Has finding site",
         "similarity": 0.9, "extracted_mention": "bone", "attribute_key": "finding_site"},
    ])
    prompts_seen = []
    monkeypatch.setattr(pipeline, "_retrieve_candidates", lambda *a, **k: (candidates, 0.0))
    monkeypatch.setattr(pipeline, "infer_attributes", lambda *a, **k: ({"finding_site": ["bone"]}, 0.0))

    def fake_llm(system_prompt, user_prompt):
        prompts_seen.append(system_prompt)
        return pipeline.LlmResult('{"attributes": {"finding_site": [{"concept_id": "1"}]}}', 0.0)

    monkeypatch.setattr(pipeline, "call_llm", fake_llm)
    result = pipeline.extract_attributes(
        "Bone disorder",
        attribute_searcher=cast(pipeline.AttributeSearcher, object()),
        reference_searcher=cast(pipeline.ReferenceSearcher, SimpleNamespace(search=lambda *a, **k: ([], 0.0))),
        hierarchy_settings=cast(HierarchySettings, cfg),
    )
    assert "role_groups" not in result and "grouping_method" not in result
    assert prompts_seen == ["S "]
