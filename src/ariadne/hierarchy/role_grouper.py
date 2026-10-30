"""Assign extracted SNOMED CT attributes to role groups.

The grouping combines:

* LLM-proposed groups (``role_groups`` in the selection output) for the clusters listed in
  ``hierarchy.mrcm.grouping.llm_grouped_clusters``;
* ``interprets_interpretation`` pairs (each pair is one group);
* deterministic rules for everything else: self-grouped attributes get one group per value;
  for a cluster, one group when every attribute has a single value, one group per value when
  one attribute has several values (single values are repeated in every group), and a
  positional fallback when several attributes have several values (ambiguous — flagged).

Output format (``result["role_groups"]``)::

    [{"group": 1, "members": [{"attribute": "finding_site", "concept_id": ..., "concept_name": ...,
                               "concept_code": ...}, ...]}, ...]

``group`` 0 holds attributes the MRCM marks as ungrouped for the term's domain.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable, NamedTuple

from ariadne.utils.settings import MrcmSettings

INTERPRETS_PAIR_KEY = "interprets_interpretation"

METHOD_NONE = "none"
METHOD_DETERMINISTIC = "deterministic"
METHOD_LLM = "llm"
METHOD_POSITIONAL = "positional_fallback"


class GroupingResult(NamedTuple):
    role_groups: list[dict]
    method: str


Concept = dict[str, Any]
Member = dict[str, Any]


def _as_concepts(value: Any) -> list[Concept]:
    """Normalize a selected value (dict, list of dicts, or None) to a de-duplicated list."""
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    concepts, seen = [], set()
    for item in items:
        if not isinstance(item, dict) or item.get("concept_id") in (None, ""):
            continue
        cid = str(item["concept_id"])
        if cid not in seen:
            seen.add(cid)
            concepts.append(item)
    return concepts


def _member(attribute_key: str, concept: Concept) -> Member:
    return {
        "attribute": attribute_key,
        "concept_id": concept.get("concept_id"),
        "concept_name": concept.get("concept_name"),
        "concept_code": concept.get("concept_code"),
    }


def _add_values(target: dict[str, list[Concept]], key: str, concepts: Iterable[Concept]) -> None:
    existing = target.setdefault(key, [])
    seen = {str(c["concept_id"]) for c in existing}
    for concept in concepts:
        if str(concept["concept_id"]) not in seen:
            seen.add(str(concept["concept_id"]))
            existing.append(concept)
    if not existing:
        target.pop(key)


def attribute_keys(attributes: dict | None) -> list[str]:
    """Attribute keys with at least one value (``interprets_interpretation`` expanded)."""
    keys: list[str] = []
    for key, value in (attributes or {}).items():
        if key == INTERPRETS_PAIR_KEY:
            for pair in value or []:
                if isinstance(pair, dict):
                    keys += [k for k in ("interprets", "interpretation") if _as_concepts(pair.get(k))]
        elif _as_concepts(value):
            keys.append(key)
    return list(dict.fromkeys(keys))


def flatten_role_groups(llm_role_groups: list[dict] | None) -> dict[str, list[Concept]]:
    """Collapse LLM ``role_groups`` (``[{attribute_key: concept | [concepts]}]``) to flat attributes."""
    flat: dict[str, list[Concept]] = {}
    for group in llm_role_groups or []:
        if isinstance(group, dict):
            for key, value in group.items():
                _add_values(flat, key, _as_concepts(value))
    return flat


def _deterministic_cluster(values: dict[str, list[Concept]]) -> tuple[list[list[Member]], str]:
    multi = [k for k, v in values.items() if len(v) > 1]
    singles = [_member(k, v[0]) for k, v in values.items() if len(v) == 1]
    if not multi:
        return ([singles] if singles else []), METHOD_DETERMINISTIC
    if len(multi) == 1:
        key = multi[0]
        return [[_member(key, c)] + singles for c in values[key]], METHOD_DETERMINISTIC
    # Several multi-valued attributes: which value goes with which is not recoverable.
    size = max(len(values[k]) for k in multi)
    groups = [[_member(k, values[k][i]) for k in multi if i < len(values[k])] + singles for i in range(size)]
    return groups, METHOD_POSITIONAL


def _llm_cluster(
    keys: list[str],
    values: dict[str, list[Concept]],
    llm_role_groups: list[dict],
) -> tuple[list[list[Member]], str]:
    """Use LLM groups for a cluster, consuming the values they cover from ``values``.

    An LLM group with several values for one attribute is split like a deterministic cluster.
    """
    groups: list[list[Member]] = []
    method = METHOD_LLM
    covered: set[tuple[str, str]] = set()
    for raw in llm_role_groups:
        if not isinstance(raw, dict):
            continue
        group_values = {k: concepts for k in keys if (concepts := _as_concepts(raw.get(k)))}
        split, split_method = _deterministic_cluster(group_values)
        if split_method == METHOD_POSITIONAL:
            method = METHOD_POSITIONAL
        groups += split
        covered.update((m["attribute"], str(m["concept_id"])) for group in split for m in group)
    if not groups:
        return [], method
    for key in keys:
        remaining = [c for c in values.get(key, []) if (key, str(c["concept_id"])) not in covered]
        if len(remaining) == 1:
            # A single uncovered value is repeated in every LLM group that lacks this attribute.
            lacking = [g for g in groups if all(m["attribute"] != key for m in g)]
            if lacking:
                for group in lacking:
                    group.append(_member(key, remaining[0]))
                remaining = []
        if remaining:
            values[key] = remaining
        else:
            values.pop(key, None)
    return groups, method


def group_attributes(
    attributes: dict | None,
    settings: MrcmSettings,
    llm_role_groups: list[dict] | None = None,
    rules: Any = None,
    domain: str | None = None,
) -> GroupingResult:
    """Assign the selected attributes of one term to role groups.

    Args:
        attributes: ``result["attributes"]`` (``{key: concept | [concepts] | None}``, with
            optional ``interprets_interpretation`` pairs).
        settings: ``hierarchy.mrcm`` settings (grouping rules).
        llm_role_groups: Normalized LLM role groups ``[{attribute_key: concept | [concepts]}]``.
        rules: Optional :class:`~ariadne.hierarchy.mrcm.MrcmRules`; when given with ``domain``,
            attributes the MRCM marks as ungrouped are placed in group 0.
        domain: MRCM domain name of the term.

    Returns:
        ``GroupingResult(role_groups, method)``.
    """
    grouping = settings.grouping
    values: dict[str, list[Concept]] = {}
    pairs: list[tuple[Concept, Concept]] = []
    for key, value in (attributes or {}).items():
        if key == INTERPRETS_PAIR_KEY:
            for pair in value or []:
                if not isinstance(pair, dict):
                    continue
                interprets, interpretation = _as_concepts(pair.get("interprets")), _as_concepts(pair.get("interpretation"))
                if interprets and interpretation:
                    pairs.append((interprets[0], interpretation[0]))
                else:
                    _add_values(values, "interprets", interprets)
                    _add_values(values, "interpretation", interpretation)
        else:
            _add_values(values, key, _as_concepts(value))
    for key, concepts in flatten_role_groups(llm_role_groups).items():
        _add_values(values, key, concepts)

    methods: set[str] = set()
    ungrouped: list[Member] = []
    if rules is not None and domain:
        domain_rules = rules.rules_for_domain(domain)
        for key in list(values):
            rule = domain_rules.get(key)
            if rule is not None and not rule.grouped:
                ungrouped += [_member(key, c) for c in values.pop(key)]

    groups: list[list[Member]] = []
    pair_cluster = grouping.cluster_of("interprets")
    for name, keys in grouping.clusters.items():
        cluster_groups: list[list[Member]] = []
        if name == pair_cluster:
            cluster_groups += [[_member("interprets", a), _member("interpretation", b)] for a, b in pairs]
        if name in grouping.llm_grouped_clusters and llm_role_groups:
            llm_groups, llm_method = _llm_cluster(keys, values, llm_role_groups)
            if llm_groups:
                methods.add(llm_method)
                cluster_groups += llm_groups
        remaining = {k: values.pop(k) for k in keys if k in values and k not in grouping.self_grouped}
        if remaining:
            extra, method = _deterministic_cluster(remaining)
            methods.add(method)
            cluster_groups += extra
        groups += cluster_groups
    if pair_cluster is None:
        groups += [[_member("interprets", a), _member("interpretation", b)] for a, b in pairs]

    # Self-grouped attributes, then anything not covered by the grouping rules: one group per value.
    for key in [k for k in grouping.self_grouped if k in values] + [k for k in values if k not in grouping.self_grouped]:
        groups += [[_member(key, c)] for c in values[key]]
    if groups:
        methods.add(METHOD_DETERMINISTIC)

    unique: list[list[Member]] = []
    seen: set[frozenset] = set()
    for group in groups:
        signature = frozenset((m["attribute"], str(m["concept_id"])) for m in group)
        if signature not in seen:
            seen.add(signature)
            unique.append(group)

    role_groups = [{"group": 0, "members": ungrouped}] if ungrouped else []
    role_groups += [{"group": i, "members": members} for i, members in enumerate(unique, start=1)]

    if not role_groups:
        method = METHOD_NONE
    elif METHOD_POSITIONAL in methods:
        method = METHOD_POSITIONAL
    elif METHOD_LLM in methods:
        method = METHOD_LLM
    else:
        method = METHOD_DETERMINISTIC
    return GroupingResult(role_groups, method)


def group_result(result: dict, settings: MrcmSettings, rules: Any = None) -> None:
    """Set ``role_groups``, ``grouping_method`` and, when ``rules`` is given, ``mrcm_domain`` on a
    pipeline result (in place)."""
    attributes = result.get("attributes") or {}
    domain = rules.select_domain(attribute_keys(attributes)) if rules is not None else None
    if domain:
        result["mrcm_domain"] = domain
    grouping = group_attributes(attributes, settings, result.get("llm_role_groups"), rules, domain)
    result["role_groups"] = grouping.role_groups
    result["grouping_method"] = grouping.method


def regroup_result(result: dict, settings: MrcmSettings, rules: Any = None) -> dict:
    """Return a copy of a pipeline result with role groups (re)computed by :func:`group_result`.

    Results with an ``error`` key are returned unchanged.
    """
    if not isinstance(result, dict) or result.get("error"):
        return result
    updated = copy.deepcopy(result)
    group_result(updated, settings, rules)
    return updated


def regroup_results(results: list[dict], settings: MrcmSettings, rules: Any = None) -> list[dict]:
    """Apply :func:`regroup_result` to cached pipeline results (no LLM calls)."""
    return [regroup_result(r, settings, rules) for r in results]
