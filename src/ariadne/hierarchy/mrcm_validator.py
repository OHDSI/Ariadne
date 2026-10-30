"""Validate role-grouped attribute predictions against SNOMED CT MRCM and grouping rules.

Usage::

    from ariadne.hierarchy.mrcm import MrcmRules
    from ariadne.hierarchy.mrcm_validator import validate_role_groups

    rules = MrcmRules.load(config.hierarchy.mrcm, config.hierarchy.attributes)
    violations = validate_role_groups(results, rules)   # one row per violation

Rules (``rule`` column):

=========================  ========  ========================================================
rule                       severity  meaning
=========================  ========  ========================================================
unknown_attribute          error     attribute key not configured in ``hierarchy.attributes``
attribute_not_in_domain    error     MRCM does not allow the attribute in the term's domain
attribute_cardinality      error     number of values of the attribute violates MRCM cardinality
in_group_cardinality       error     too many values of the attribute in one role group
grouped_mismatch           error     grouped attribute in group 0, or ungrouped attribute in a role group
self_grouped_shared        error     self-grouped attribute shares its role group
cluster_mix                error     attributes from different grouping clusters share a group
missing_required_partner   error     e.g. ``interpretation`` without ``interprets`` in its group
attribute_range            error     value outside the MRCM range (only with ``is_a``)
duplicate_group            warning   two role groups with identical content
ambiguous_grouping         warning   grouping fell back to positional pairing
=========================  ========  ========================================================
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from ariadne.hierarchy.mrcm import IsAFunction, MrcmRules
from ariadne.hierarchy.role_grouper import METHOD_POSITIONAL, regroup_result

ERROR = "error"
WARNING = "warning"

VIOLATION_COLUMNS = [
    "source_key",
    "source_term",
    "mrcm_domain",
    "rule",
    "severity",
    "attribute",
    "group",
    "concept_id",
    "concept_name",
    "detail",
]


def validate_result(result: dict, rules: MrcmRules, is_a: IsAFunction | None = None) -> list[dict]:
    """Validate one pipeline result. Results without ``role_groups`` are grouped first."""
    if not isinstance(result, dict) or result.get("error"):
        return []
    if result.get("role_groups") is None:
        result = regroup_result(result, rules.settings, rules)
    role_groups: list[dict] = result.get("role_groups") or []
    if not role_groups:
        return []

    grouping = rules.settings.grouping
    domain = result.get("mrcm_domain") or rules.select_domain(
        m.get("attribute") for g in role_groups for m in g.get("members", []))
    domain_rules = rules.rules_for_domain(domain)
    base = {"source_key": result.get("source_key"), "source_term": result.get("source_term"), "mrcm_domain": domain}
    violations: list[dict] = []

    def add(rule: str, severity: str, detail: str, attribute=None, group=None, member=None) -> None:
        violations.append({
            **base,
            "rule": rule,
            "severity": severity,
            "attribute": attribute,
            "group": group,
            "concept_id": member.get("concept_id") if member else None,
            "concept_name": member.get("concept_name") if member else None,
            "detail": detail,
        })

    # Attribute level: configured, allowed in domain, value in range.
    values_by_key: dict[str, set[str]] = {}
    checked: set[tuple[str, str]] = set()
    for group in role_groups:
        number = group.get("group")
        for member in group.get("members", []):
            key = member.get("attribute")
            values_by_key.setdefault(key, set()).add(str(member.get("concept_id")))
            if (key, str(member.get("concept_id"))) in checked:
                continue
            checked.add((key, str(member.get("concept_id"))))
            if key not in rules.attributes:
                add("unknown_attribute", ERROR, f"'{key}' is not configured in hierarchy.attributes", key, number, member)
                continue
            if key not in domain_rules:
                add("attribute_not_in_domain", ERROR, f"MRCM does not allow '{key}' in domain '{domain}'", key, number, member)
                continue
            code = member.get("concept_code")
            if is_a is not None and code not in (None, ""):
                in_range = rules.in_range(key, str(code), is_a)
                if in_range is False:
                    constraints = " OR ".join(r.range_constraint for r in rules.ranges(key))
                    inactive = hasattr(is_a, "__contains__") and str(code) not in is_a  # type: ignore[operator]
                    reason = "is not active in the RF2 release" if inactive else f"is outside range: {constraints}"
                    add("attribute_range", ERROR, f"{code} {reason}", key, number, member)

    for key, concept_ids in values_by_key.items():
        rule = domain_rules.get(key)
        if rule is not None and not rule.cardinality.allows(len(concept_ids)):
            add("attribute_cardinality", ERROR,
                f"{len(concept_ids)} values; MRCM cardinality {rule.cardinality}", key)

    # Group level.
    signatures: Counter = Counter()
    for group in role_groups:
        number = group.get("group")
        members = group.get("members", [])
        keys = [m.get("attribute") for m in members]
        signatures[frozenset((m.get("attribute"), str(m.get("concept_id"))) for m in members)] += 1

        for key, count in Counter(keys).items():
            rule = domain_rules.get(key)
            if rule is None:
                continue
            if number != 0 and not rule.in_group_cardinality.allows(count):
                add("in_group_cardinality", ERROR,
                    f"{count} values in group {number}; MRCM in-group cardinality {rule.in_group_cardinality}",
                    key, number)
            if (number == 0) == rule.grouped:
                expected = "a role group" if rule.grouped else "group 0"
                add("grouped_mismatch", ERROR, f"'{key}' must be in {expected}", key, number)

        if number == 0:
            continue
        for key in set(keys) & set(grouping.self_grouped):
            if len(members) > 1:
                add("self_grouped_shared", ERROR,
                    f"'{key}' must be alone in its role group; group {number} has {sorted(set(keys))}", key, number)
        clusters = {grouping.cluster_of(k) for k in keys if grouping.cluster_of(k) and k not in grouping.self_grouped}
        if len(clusters) > 1:
            add("cluster_mix", ERROR, f"group {number} mixes clusters {sorted(clusters)}: {sorted(set(keys))}",
                group=number)
        for key, partner in grouping.requires.items():
            if key in keys and partner not in keys:
                add("missing_required_partner", ERROR, f"'{key}' requires '{partner}' in group {number}", key, number)

    duplicates = sum(count - 1 for count in signatures.values() if count > 1)
    if duplicates:
        add("duplicate_group", WARNING, f"{duplicates} duplicated role group(s)")
    if result.get("grouping_method") == METHOD_POSITIONAL:
        add("ambiguous_grouping", WARNING,
            "several attributes have several values; groups were paired by position and may be wrong")
    return violations


def validate_role_groups(
    results: list[dict],
    rules: MrcmRules,
    is_a: IsAFunction | None = None,
) -> pd.DataFrame:
    """Validate pipeline results; returns one row per violation (``VIOLATION_COLUMNS``).

    Args:
        results: Pipeline result dicts (with or without ``role_groups``).
        rules: Loaded MRCM rules.
        is_a: Optional ``is_a(concept_code, ancestor_code)`` callable (e.g.
            :class:`~ariadne.hierarchy.mrcm.Rf2IsAIndex`) enabling MRCM range checks.
    """
    rows = [v for result in results for v in validate_result(result, rules, is_a)]
    return pd.DataFrame(rows, columns=VIOLATION_COLUMNS)
