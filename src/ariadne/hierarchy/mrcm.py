"""SNOMED CT MRCM (Machine Readable Concept Model) rules.

Loads the RF2 MRCM refset snapshots (Domain, AttributeDomain, AttributeRange) for the
domains and content types configured in ``hierarchy.mrcm`` and exposes them keyed by the
pipeline's attribute keys (``finding_site``, ``associated_morphology``, ...) configured in
``hierarchy.attributes``.

Only pre-coordinated rules are used: rows whose ``contentTypeId`` is not listed in
``hierarchy.mrcm.content_types`` are ignored.
"""

from __future__ import annotations

import csv
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping

from ariadne.utils.settings import AttributeConfig, MrcmSettings

logger = logging.getLogger(__name__)

IS_A_TYPE_ID = "116680003"
INFERRED_CHARACTERISTIC_TYPE_ID = "900000000000011006"

_DOMAIN_GLOB = "*MRCMDomainSnapshot*.txt"
_ATTRIBUTE_DOMAIN_GLOB = "*MRCMAttributeDomainSnapshot*.txt"
_ATTRIBUTE_RANGE_GLOB = "*MRCMAttributeRangeSnapshot*.txt"

# ``is_a(concept_code, ancestor_code)`` -> True if concept is ancestor or a descendant of it.
IsAFunction = Callable[[str, str], bool]


@dataclass(frozen=True)
class Cardinality:
    """MRCM cardinality such as ``0..*`` or ``1..1``. ``max is None`` means unbounded."""

    min: int
    max: int | None

    @classmethod
    def parse(cls, text: str) -> "Cardinality":
        low, _, high = text.strip().partition("..")
        return cls(int(low), None if high in ("*", "") else int(high))

    def allows(self, count: int) -> bool:
        return count >= self.min and (self.max is None or count <= self.max)

    def __str__(self) -> str:
        return f"{self.min}..{'*' if self.max is None else self.max}"


@dataclass(frozen=True)
class AttributeRule:
    attribute_id: int
    domain_id: int
    grouped: bool
    cardinality: Cardinality
    in_group_cardinality: Cardinality


@dataclass(frozen=True)
class RangeRule:
    attribute_id: int
    range_constraint: str
    # Parsed simple ECL ``[(operator, concept_id), ...]``; ``None`` if the ECL is not supported.
    focus: tuple[tuple[str, str], ...] | None = None


@dataclass(frozen=True)
class MrcmDomain:
    name: str
    domain_id: int
    parent_domain_id: int | None


_PIPE_TERM = re.compile(r"\|[^|]*\|")
_SIMPLE_FOCUS = re.compile(r"^(<<|<|)\s*(\d+)$")


def parse_simple_ecl(constraint: str) -> tuple[tuple[str, str], ...] | None:
    """Parse an ECL disjunction of ``<<id``, ``<id`` or ``id`` terms.

    Returns ``None`` for anything more complex (refinements, MINUS, AND, nesting),
    in which case range checks are skipped for that rule.
    """
    text = _PIPE_TERM.sub("", constraint).strip()
    if not text or any(tok in text for tok in ("(", ")", ":", "{", "MINUS", " AND ")):
        return None
    parts = []
    for part in text.split(" OR "):
        match = _SIMPLE_FOCUS.match(part.strip())
        if not match:
            return None
        parts.append((match.group(1), match.group(2)))
    return tuple(parts)


def _leading_id(value: str) -> int | None:
    match = re.match(r"\s*(\d+)", value or "")
    return int(match.group(1)) if match else None


def _read_active_rows(folder: Path, pattern: str) -> Iterator[dict[str, str]]:
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No MRCM file matching '{pattern}' in {folder}")
    if len(files) > 1:
        logger.warning("Multiple MRCM files match %s in %s; using %s", pattern, folder, files[-1].name)
    with open(files[-1], encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
            if row.get("active") == "1":
                yield row


@dataclass
class MrcmRules:
    """MRCM rules for the configured domains, addressable by pipeline attribute key."""

    settings: MrcmSettings
    domains: dict[str, MrcmDomain]
    attribute_rules: dict[int, dict[int, AttributeRule]]
    range_rules: dict[int, list[RangeRule]] = field(default_factory=dict)
    # ``hierarchy.attributes``: pipeline attribute key -> SNOMED attribute concept.
    attributes: dict[str, AttributeConfig] = field(default_factory=dict)

    # ── construction ──────────────────────────────────────────────────────────

    @classmethod
    def load(cls, settings: MrcmSettings, attributes: Mapping[str, AttributeConfig]) -> "MrcmRules":
        """Load MRCM rules from the RF2 refset files in ``settings.folder``.

        Args:
            settings: ``hierarchy.mrcm``.
            attributes: ``hierarchy.attributes``; maps pipeline attribute keys to MRCM attribute IDs.
        """
        folder = Path(settings.folder)
        content_types = {int(c) for c in settings.content_types}
        wanted = {int(d): name for name, d in settings.domains.items()}

        domains: dict[str, MrcmDomain] = {}
        for row in _read_active_rows(folder, _DOMAIN_GLOB):
            domain_id = int(row["referencedComponentId"])
            if domain_id in wanted:
                domains[wanted[domain_id]] = MrcmDomain(
                    name=wanted[domain_id],
                    domain_id=domain_id,
                    parent_domain_id=_leading_id(row.get("parentDomain", "")),
                )
        missing = set(wanted.values()) - set(domains)
        if missing:
            raise ValueError(f"MRCM domains not found in {folder}: {sorted(missing)}")

        attribute_rules: dict[int, dict[int, AttributeRule]] = defaultdict(dict)
        for row in _read_active_rows(folder, _ATTRIBUTE_DOMAIN_GLOB):
            domain_id = int(row["domainId"])
            content_type = int(row["contentTypeId"])
            if domain_id not in wanted or content_type not in content_types:
                continue
            attribute_id = int(row["referencedComponentId"])
            attribute_rules[domain_id][attribute_id] = AttributeRule(
                attribute_id=attribute_id,
                domain_id=domain_id,
                grouped=row["grouped"] == "1",
                cardinality=Cardinality.parse(row["attributeCardinality"]),
                in_group_cardinality=Cardinality.parse(row["attributeInGroupCardinality"]),
            )

        relevant_attributes = {a for rules in attribute_rules.values() for a in rules}
        range_rules: dict[int, list[RangeRule]] = defaultdict(list)
        for row in _read_active_rows(folder, _ATTRIBUTE_RANGE_GLOB):
            attribute_id = int(row["referencedComponentId"])
            content_type = int(row["contentTypeId"])
            if attribute_id not in relevant_attributes or content_type not in content_types:
                continue
            range_rules[attribute_id].append(
                RangeRule(
                    attribute_id=attribute_id,
                    range_constraint=row["rangeConstraint"],
                    focus=parse_simple_ecl(row["rangeConstraint"]),
                )
            )

        return cls(settings, domains, dict(attribute_rules), dict(range_rules), dict(attributes))

    # ── attribute key helpers ─────────────────────────────────────────────────

    def attribute_id(self, attribute_key: str) -> int | None:
        config = self.attributes.get(attribute_key)
        return config.attribute_id if config else None

    def attribute_key(self, attribute_id: int) -> str | None:
        for key, config in self.attributes.items():
            if config.attribute_id == attribute_id:
                return key
        return None

    # ── domain helpers ────────────────────────────────────────────────────────

    def domain_chain(self, domain_name: str) -> list[MrcmDomain]:
        """The domain followed by its configured MRCM ancestors (nearest first)."""
        by_id = {d.domain_id: d for d in self.domains.values()}
        chain, current = [], self.domains[domain_name]
        while current is not None and current not in chain:
            chain.append(current)
            current = by_id.get(current.parent_domain_id) if current.parent_domain_id else None
        return chain

    def rules_for_domain(self, domain_name: str) -> dict[str, AttributeRule]:
        """Attribute rules (own + inherited) of a domain keyed by pipeline attribute key.

        Attributes allowed by the MRCM but without a pipeline key are keyed ``"<id>"``.
        """
        rules: dict[int, AttributeRule] = {}
        for domain in reversed(self.domain_chain(domain_name)):
            rules.update(self.attribute_rules.get(domain.domain_id, {}))
        return {(self.attribute_key(a) or str(a)): rule for a, rule in rules.items()}

    def select_domain(self, attribute_keys: Iterable[str]) -> str:
        """Pick the configured domain that allows the most of ``attribute_keys``.

        Ties go to ``default_domain`` and then to the most specific (deepest) domain.
        """
        keys = set(attribute_keys)
        default = self.settings.default_domain or next(iter(self.domains))

        def score(name: str) -> tuple[int, bool, int]:
            allowed = set(self.rules_for_domain(name))
            return len(keys & allowed), name == default, len(self.domain_chain(name))

        return max(self.domains, key=score)

    def ranges(self, attribute_key: str) -> list[RangeRule]:
        attribute_id = self.attribute_id(attribute_key)
        return self.range_rules.get(attribute_id, []) if attribute_id is not None else []

    def in_range(self, attribute_key: str, concept_code: str, is_a: IsAFunction) -> bool | None:
        """Whether ``concept_code`` satisfies any range rule of the attribute.

        Returns ``None`` when no range rule could be evaluated (unsupported ECL).
        """
        evaluable = [r for r in self.ranges(attribute_key) if r.focus is not None]
        if not evaluable:
            return None
        code = str(concept_code)
        for rule in evaluable:
            for operator, focus in rule.focus or ():
                if operator == "" and code == focus:
                    return True
                if operator == "<<" and is_a(code, focus):
                    return True
                if operator == "<" and code != focus and is_a(code, focus):
                    return True
        return False


class Rf2IsAIndex:
    """Transitive ``is a`` lookups built from an RF2 relationship snapshot (inferred, active)."""

    def __init__(self, parents: dict[str, set[str]]):
        self._parents = parents
        self._ancestors: dict[str, frozenset[str]] = {}

    @classmethod
    def from_relationship_file(cls, path: str | Path) -> "Rf2IsAIndex":
        parents: dict[str, set[str]] = defaultdict(set)
        with open(path, encoding="utf-8") as handle:
            next(handle)
            for line in handle:
                cols = line.rstrip("\n").split("\t")
                if cols[2] == "1" and cols[7] == IS_A_TYPE_ID and cols[8] == INFERRED_CHARACTERISTIC_TYPE_ID:
                    parents[cols[4]].add(cols[5])
        return cls(dict(parents))

    def ancestors(self, concept_code: str) -> frozenset[str]:
        cached = self._ancestors.get(concept_code)
        if cached is not None:
            return cached
        seen: set[str] = set()
        stack = list(self._parents.get(concept_code, ()))
        while stack:
            parent = stack.pop()
            if parent not in seen:
                seen.add(parent)
                stack.extend(self._parents.get(parent, ()))
        result = frozenset(seen)
        self._ancestors[concept_code] = result
        return result

    def __contains__(self, concept_code: object) -> bool:
        """True if the concept has an active inferred ``is a`` (i.e. is active, non-root)."""
        return str(concept_code) in self._parents

    def __call__(self, concept_code: str, ancestor_code: str) -> bool:
        concept_code, ancestor_code = str(concept_code), str(ancestor_code)
        return concept_code == ancestor_code or ancestor_code in self.ancestors(concept_code)


def build_grouping_prompt_section(settings: MrcmSettings) -> str:
    """Human-readable role-grouping rules injected into the hierarchy prompts."""
    grouping = settings.grouping
    if not grouping.enabled:
        return ""
    llm_keys = grouping.llm_grouped_keys()
    lines = ["# Role Groups (SNOMED CT concept model)"]
    if llm_keys:
        lines += [
            f"- {', '.join(llm_keys)} are reported inside \"role_groups\": a list of groups, "
            "each group describing ONE disease process (one site with its morphology/process/agent/occurrence).",
            "- Use ONE group when the term describes one process. Use SEVERAL groups only when the term names "
            "several distinct sites or lesions (e.g. 'X of A and B', 'fracture of radius and ulna').",
            "- Each group contains at most one value per attribute. Repeat a shared value (e.g. the same morphology) "
            "in every group it applies to.",
        ]
    if grouping.self_grouped:
        lines.append(
            f"- {', '.join(grouping.self_grouped)} are always reported at the top level, never inside role_groups."
        )
    other = [k for name, keys in grouping.clusters.items() if name not in grouping.llm_grouped_clusters for k in keys]
    if {"interprets", "interpretation"} & set(other):
        other = [k for k in other if k not in ("interprets", "interpretation")] + ["interprets_interpretation"]
    if other:
        lines.append(f"- {', '.join(other)} are reported at the top level, never inside role_groups.")
    return "\n".join(lines)
