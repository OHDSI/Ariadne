"""Unit tests for MRCM / role-group validation (ariadne.hierarchy.mrcm_validator)."""

import pytest
from conftest import concept, make_attributes

from ariadne.hierarchy.mrcm import MrcmRules, Rf2IsAIndex
from ariadne.hierarchy.mrcm_validator import VIOLATION_COLUMNS, validate_role_groups


@pytest.fixture
def rules(mrcm_settings):
    return MrcmRules.load(mrcm_settings, make_attributes())


def _member(attribute, concept_id, code=None):
    return {"attribute": attribute, **concept(concept_id, code=code)}


def _result(*groups, method="deterministic", term="term", domain=None):
    result = {
        "source_key": f"TEST:{term}",
        "source_term": term,
        "attributes": {},
        "grouping_method": method,
        "role_groups": [{"group": number, "members": list(members)} for number, members in groups],
    }
    if domain:
        result["mrcm_domain"] = domain
    return result


def _rules_found(violations):
    return sorted(set(violations["rule"]))


def test_valid_result_has_no_violations(rules):
    result = _result(
        (1, [_member("finding_site", 1), _member("associated_morphology", 2)]),
        (2, [_member("finding_site", 3), _member("associated_morphology", 2)]),
        (3, [_member("clinical_course", 4)]),
        (4, [_member("interprets", 5), _member("interpretation", 6)]),
    )
    violations = validate_role_groups([result], rules)
    assert list(violations.columns) == VIOLATION_COLUMNS
    assert violations.empty


def test_attribute_not_in_domain_names_the_term(rules):
    result = _result(
        (1, [_member("finding_site", 1)]),
        (2, [_member("interprets", 4), _member("interpretation", 5)]),
        (3, [_member("finding_context", 2), _member("temporal_context", 3)]),
        term="Heart failure confirmed",
    )
    violations = validate_role_groups([result], rules)

    assert _rules_found(violations) == ["attribute_not_in_domain"]
    assert set(violations["source_term"]) == {"Heart failure confirmed"}
    assert set(violations["attribute"]) == {"finding_context", "temporal_context"}
    assert set(violations["mrcm_domain"]) == {"clinical_finding"}


def test_context_only_term_is_valid_in_situation_domain(rules):
    result = _result((1, [_member("finding_context", 1), _member("temporal_context", 2),
                          _member("subject_relationship_context", 3)]))
    assert validate_role_groups([result], rules).empty


def test_unknown_attribute(rules):
    violations = validate_role_groups([_result((1, [_member("mystery", 1)]))], rules)
    assert _rules_found(violations) == ["unknown_attribute"]


def test_in_group_cardinality(rules):
    violations = validate_role_groups(
        [_result((1, [_member("finding_site", 1), _member("finding_site", 2)]))], rules)
    assert _rules_found(violations) == ["in_group_cardinality"]


def test_associated_with_allows_many_per_group(rules):
    result = _result((1, [_member("finding_asso_with", 1), _member("finding_asso_with", 2)]))
    assert "in_group_cardinality" not in _rules_found(validate_role_groups([result], rules))


def test_attribute_cardinality(rules):
    result = _result((0, [_member("episodicity", 1), _member("episodicity", 2)]))
    assert "attribute_cardinality" in _rules_found(validate_role_groups([result], rules))


def test_grouped_mismatch_both_ways(rules):
    result = _result((0, [_member("finding_site", 1)]), (1, [_member("episodicity", 2)]))
    violations = validate_role_groups([result], rules)
    assert _rules_found(violations) == ["grouped_mismatch"]
    assert set(violations["attribute"]) == {"finding_site", "episodicity"}


def test_self_grouped_attribute_shares_group(rules):
    violations = validate_role_groups(
        [_result((1, [_member("finding_site", 1), _member("due_to", 2)]))], rules)
    assert _rules_found(violations) == ["self_grouped_shared"]
    assert violations.iloc[0]["attribute"] == "due_to"


def test_cluster_mix(rules):
    violations = validate_role_groups(
        [_result((1, [_member("finding_site", 1), _member("interprets", 2), _member("interpretation", 3)]))], rules)
    assert _rules_found(violations) == ["cluster_mix"]


def test_missing_required_partner(rules):
    violations = validate_role_groups([_result((1, [_member("interpretation", 1)]))], rules)
    assert _rules_found(violations) == ["missing_required_partner"]


def test_warnings_for_duplicates_and_ambiguous_grouping(rules):
    result = _result(
        (1, [_member("finding_site", 1)]),
        (2, [_member("finding_site", 1)]),
        method="positional_fallback",
    )
    violations = validate_role_groups([result], rules)
    assert _rules_found(violations) == ["ambiguous_grouping", "duplicate_group"]
    assert set(violations["severity"]) == {"warning"}


def test_range_check_is_opt_in(rules):
    is_a = Rf2IsAIndex({"100": {"442083009"}, "200": {"999"}})
    result = _result((1, [_member("finding_site", 1, code="100")]), (2, [_member("finding_site", 2, code="200")]),
                     (3, [_member("finding_site", 3, code="300")]))

    assert validate_role_groups([result], rules).empty
    violations = validate_role_groups([result], rules, is_a=is_a)
    assert _rules_found(violations) == ["attribute_range"]
    details = dict(zip(violations["concept_id"], violations["detail"]))
    assert "outside range" in details[2]
    assert "not active" in details[3]


def test_results_without_role_groups_are_grouped_first(rules):
    result = {
        "source_key": "k",
        "source_term": "Fracture with trauma",
        "attributes": {"finding_site": [concept(1), concept(2)], "associated_morphology": [concept(3), concept(4)]},
    }
    violations = validate_role_groups([result], rules)
    assert _rules_found(violations) == ["ambiguous_grouping"]


def test_errors_and_empty_results_are_skipped(rules):
    results = [{"source_key": "a", "error": "x"}, {"source_key": "b", "attributes": {}}]
    assert validate_role_groups(results, rules).empty
