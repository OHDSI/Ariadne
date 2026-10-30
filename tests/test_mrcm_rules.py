"""Tests for MRCM rule loading (ariadne.hierarchy.mrcm) and config consistency."""

from pathlib import Path

import pytest
import yaml

from ariadne.hierarchy.mrcm import (
    Cardinality,
    MrcmRules,
    Rf2IsAIndex,
    build_grouping_prompt_section,
    parse_simple_ecl,
)
from ariadne.utils.settings import HierarchySettings, MrcmSettings, build_dataclass
from conftest import make_attributes

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config_condition_mapping.yaml"
ATTRIBUTES = make_attributes()


def test_cardinality_parse_and_allows():
    assert Cardinality.parse("0..*") == Cardinality(0, None)
    assert Cardinality.parse("1..1").allows(1)
    assert not Cardinality.parse("1..1").allows(2)
    assert not Cardinality.parse("1..*").allows(0)
    assert str(Cardinality.parse("0..*")) == "0..*"


def test_parse_simple_ecl():
    assert parse_simple_ecl("<< 1 |A| OR < 2 |B| OR 3 |C|") == (("<<", "1"), ("<", "2"), ("", "3"))
    assert parse_simple_ecl("(<< 1 |A|) MINUS << 2 |B|") is None
    assert parse_simple_ecl("<< 1 : 2 = 3") is None


def test_load_filters_inactive_postcoordinated_and_unconfigured(mrcm_settings):
    rules = MrcmRules.load(mrcm_settings, ATTRIBUTES)

    assert set(rules.domains) == {"clinical_finding", "situation_with_explicit_context",
                                  "finding_with_explicit_context"}
    finding_rules = rules.rules_for_domain("clinical_finding")
    assert "finding_site" in finding_rules
    assert "719722006" not in finding_rules       # inactive row
    assert "occurs_after" not in finding_rules    # postcoordinated-only row
    assert all(rule.domain_id != 129125009 for d in rules.attribute_rules.values() for rule in d.values())

    # Only the precoordinated causative agent range is kept.
    assert [r.focus for r in rules.ranges("causative_agent")] == [(("<<", "105590001"), ("<<", "410607006"))]
    assert finding_rules["finding_asso_with"].in_group_cardinality == Cardinality(0, None)
    assert finding_rules["episodicity"].grouped is False


def test_subdomain_inherits_parent_rules(mrcm_settings):
    rules = MrcmRules.load(mrcm_settings, ATTRIBUTES)
    assert set(rules.rules_for_domain("finding_with_explicit_context")) == {
        "finding_context", "associated_finding", "temporal_context", "subject_relationship_context"}
    assert set(rules.rules_for_domain("situation_with_explicit_context")) == {
        "temporal_context", "subject_relationship_context"}


def test_subdomain_without_configured_parent_does_not_inherit(mrcm_folder):
    settings = MrcmSettings(folder=str(mrcm_folder), domains={"finding_with_explicit_context": 413350009})
    rules = MrcmRules.load(settings, ATTRIBUTES)
    assert {r.attribute_id for r in rules.rules_for_domain("finding_with_explicit_context").values()} == {
        408729009, 246090004}


def test_missing_domain_raises(mrcm_folder):
    with pytest.raises(ValueError, match="not_a_domain"):
        MrcmRules.load(MrcmSettings(folder=str(mrcm_folder), domains={"not_a_domain": 1}), ATTRIBUTES)


def test_missing_files_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        MrcmRules.load(MrcmSettings(folder=str(tmp_path), domains={"clinical_finding": 404684003}), ATTRIBUTES)


def test_select_domain(mrcm_settings):
    rules = MrcmRules.load(mrcm_settings, ATTRIBUTES)
    assert rules.select_domain([]) == "clinical_finding"
    assert rules.select_domain(["finding_site", "associated_morphology"]) == "clinical_finding"
    family_history = ["finding_context", "temporal_context", "subject_relationship_context"]
    assert rules.select_domain(family_history) == "finding_with_explicit_context"
    # Only temporal/subject context: most specific domain wins the tie.
    assert rules.select_domain(["temporal_context"]) == "finding_with_explicit_context"
    # Mixed: domain covering most attributes.
    assert rules.select_domain(["finding_site", "interprets", "finding_context"]) == "clinical_finding"


def test_in_range_with_is_a(mrcm_settings):
    rules = MrcmRules.load(mrcm_settings, ATTRIBUTES)
    is_a = Rf2IsAIndex({"100": {"442083009"}, "200": {"300"}, "300": {"410607006"}})

    assert rules.in_range("finding_site", "100", is_a) is True
    assert rules.in_range("finding_site", "200", is_a) is False
    assert rules.in_range("causative_agent", "200", is_a) is True       # transitive
    assert rules.in_range("finding_asso_with", "200", is_a) is None     # unsupported ECL
    assert rules.in_range("severity", "200", is_a) is None              # no range rule


def test_rf2_is_a_index_reads_active_inferred_is_a(tmp_path):
    path = tmp_path / "sct2_Relationship_Snapshot.txt"
    header = "id\teffectiveTime\tactive\tmoduleId\tsourceId\tdestinationId\trelationshipGroup\ttypeId\t" \
             "characteristicTypeId\tmodifierId"
    rows = [
        "1\t2026\t1\tm\t10\t20\t0\t116680003\t900000000000011006\tx",   # active inferred is-a
        "2\t2026\t1\tm\t20\t30\t0\t116680003\t900000000000011006\tx",
        "3\t2026\t0\tm\t10\t40\t0\t116680003\t900000000000011006\tx",   # inactive
        "4\t2026\t1\tm\t10\t50\t1\t363698007\t900000000000011006\tx",   # not is-a
    ]
    path.write_text("\n".join([header, *rows]) + "\n")
    index = Rf2IsAIndex.from_relationship_file(path)

    assert index("10", "30") and index("10", "10")
    assert not index("10", "40") and not index("10", "50")
    assert "10" in index and "99" not in index


def test_grouping_prompt_section_lists_rules(mrcm_settings):
    text = build_grouping_prompt_section(mrcm_settings)
    assert "role_groups" in text
    assert "finding_site" in text and "due_to" in text and "interprets_interpretation" in text
    assert build_grouping_prompt_section(MrcmSettings()) == ""


# ── Config consistency ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def configured_hierarchy() -> HierarchySettings:
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    return build_dataclass(HierarchySettings, raw["hierarchy"])


@pytest.fixture(scope="module")
def configured_mrcm(configured_hierarchy) -> MrcmSettings:
    return configured_hierarchy.mrcm


def test_config_mrcm_block_is_complete(configured_hierarchy, configured_mrcm):
    assert configured_mrcm.content_types == [723596005, 723594008]  # precoordinated only
    assert {"clinical_finding", "situation_with_explicit_context"} <= set(configured_mrcm.domains)
    assert configured_mrcm.default_domain in configured_mrcm.domains

    grouping = configured_mrcm.grouping
    grouped_keys = set(grouping.self_grouped) | {k for keys in grouping.clusters.values() for k in keys}
    assert grouped_keys <= set(configured_hierarchy.attributes)
    assert set(grouping.llm_grouped_clusters) <= set(grouping.clusters)
    assert not set(grouping.self_grouped) & {k for keys in grouping.clusters.values() for k in keys}


def test_every_configured_attribute_is_complete(configured_hierarchy):
    attributes = configured_hierarchy.attributes
    assert attributes, "hierarchy.attributes is empty"
    for key, attribute in attributes.items():
        assert attribute.attribute_id and attribute.ohdsi_relationship, key
    relationships = [a.ohdsi_relationship for a in attributes.values()]
    assert len(relationships) == len(set(relationships)), "each OHDSI relationship may be configured once"


def test_relationship_lookups_come_from_hierarchy_attributes():
    settings = HierarchySettings(attributes={
        "finding_site": {"attribute_id": 363698007, "ohdsi_relationship": "Has finding site"},
        "pathological_process": {"attribute_id": 370135005, "ohdsi_relationship": "Has pathology"},
        "unmapped": {"attribute_id": 1},
    })
    assert settings.relationship_type_ids() == {"Has finding site": 363698007, "Has pathology": 370135005}
    assert settings.relationship_by_attribute() == {"finding_site": "Has finding site", "pathological_process": "Has pathology"}
    assert settings.attribute_by_relationship() == {"Has finding site": "finding_site", "Has pathology": "pathological_process"}
    assert settings.snomed_relationships == ["Has finding site", "Has pathology"]


def test_legacy_relationship_blocks_are_not_read():
    hierarchy = build_dataclass(HierarchySettings, {
        "snomed_relationships": ["Has severity"],
        "attributes": {"finding_site": {"attribute_id": 363698007, "ohdsi_relationship": "Has finding site"}},
        "mrcm": {"attributes": {"severity": {"attribute_id": 246112005, "ohdsi_relationship": "Has severity"}}},
    })
    assert hierarchy.snomed_relationships == ["Has finding site"]
