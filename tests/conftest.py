"""Shared fixtures: synthetic SNOMED CT MRCM refsets and grouping settings."""

from pathlib import Path

import pytest

from ariadne.utils.settings import AttributeConfig, HierarchySettings, MrcmSettings, RoleGroupingConfig

_DOMAIN_HEADER = (
    "id\teffectiveTime\tactive\tmoduleId\trefsetId\treferencedComponentId\tdomainConstraint\tparentDomain\t"
    "proximalPrimitiveConstraint\tproximalPrimitiveRefinement\tdomainTemplateForPrecoordination\t"
    "domainTemplateForPostcoordination\tguideURL"
)
_ATTRIBUTE_DOMAIN_HEADER = (
    "id\teffectiveTime\tactive\tmoduleId\trefsetId\treferencedComponentId\tdomainId\tgrouped\t"
    "attributeCardinality\tattributeInGroupCardinality\truleStrengthId\tcontentTypeId"
)
_ATTRIBUTE_RANGE_HEADER = (
    "id\teffectiveTime\tactive\tmoduleId\trefsetId\treferencedComponentId\trangeConstraint\tattributeRule\t"
    "ruleStrengthId\tcontentTypeId"
)

ALL_CONTENT = "723596005"
PRECOORDINATED = "723594008"
POSTCOORDINATED = "723595009"
MANDATORY = "723597001"

# (attribute id, domain id, grouped, cardinality, in-group cardinality, active, content type)
_ATTRIBUTE_DOMAIN_ROWS = [
    ("363698007", "404684003", "1", "0..*", "0..1", "1", ALL_CONTENT),       # finding_site
    ("116676008", "404684003", "1", "0..*", "0..1", "1", ALL_CONTENT),       # associated_morphology
    ("246075003", "404684003", "1", "0..*", "0..1", "1", ALL_CONTENT),       # causative_agent
    ("370135005", "404684003", "1", "0..*", "0..1", "1", ALL_CONTENT),       # pathological_process
    ("246454002", "404684003", "1", "0..*", "0..1", "1", ALL_CONTENT),       # occurrence
    ("263502005", "404684003", "1", "0..*", "0..1", "1", ALL_CONTENT),       # clinical_course
    ("246112005", "404684003", "1", "0..*", "0..1", "1", ALL_CONTENT),       # severity
    ("42752001", "404684003", "1", "0..*", "0..1", "1", ALL_CONTENT),        # due_to
    ("363714003", "404684003", "1", "0..*", "0..1", "1", ALL_CONTENT),       # interprets
    ("363713009", "404684003", "1", "0..*", "0..1", "1", ALL_CONTENT),       # interpretation
    ("47429007", "404684003", "1", "0..*", "0..*", "1", ALL_CONTENT),        # finding_asso_with
    ("246456000", "404684003", "0", "0..1", "0..1", "1", ALL_CONTENT),       # episodicity (ungrouped, test only)
    ("719722006", "404684003", "1", "0..*", "0..1", "0", ALL_CONTENT),       # inactive row -> ignored
    ("255234002", "404684003", "1", "0..*", "0..1", "1", POSTCOORDINATED),   # postcoordinated -> ignored
    ("408731000", "243796009", "1", "0..*", "0..1", "1", ALL_CONTENT),       # temporal_context
    ("408732007", "243796009", "1", "0..*", "0..1", "1", ALL_CONTENT),       # subject_relationship_context
    ("408729009", "413350009", "1", "0..*", "0..1", "1", ALL_CONTENT),       # finding_context
    ("246090004", "413350009", "1", "0..*", "0..1", "1", ALL_CONTENT),       # associated_finding
    ("408730004", "129125009", "1", "0..*", "0..1", "1", ALL_CONTENT),       # procedure context (domain not configured)
]

# (attribute id, range ECL, content type)
_ATTRIBUTE_RANGE_ROWS = [
    ("363698007", "<< 442083009 |Anatomical or acquired body structure (body structure)|", ALL_CONTENT),
    ("116676008", "<< 49755003 |Morphologically abnormal structure (morphologic abnormality)|", ALL_CONTENT),
    ("246075003", "<< 105590001 |Substance (substance)| OR << 410607006 |Organism (organism)|", PRECOORDINATED),
    ("246075003", "<< 105590001 |Substance (substance)| OR 138875005 |SNOMED CT Concept|", POSTCOORDINATED),
    ("263502005", "<< 288524001 |Courses (qualifier value)|", ALL_CONTENT),
    ("47429007", "(<< 404684003 |Clinical finding|) MINUS << 1000 |Something|", PRECOORDINATED),
]


def _write(path: Path, header: str, rows: list[str]) -> None:
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")


@pytest.fixture
def mrcm_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "mrcm"
    folder.mkdir()
    _write(
        folder / "der2_sssssssRefset_MRCMDomainSnapshot_TEST_20260301.txt",
        _DOMAIN_HEADER,
        [
            "d1\t20260301\t1\tm\tr\t404684003\t<< 404684003\t\t\t\t[[template-cf]]\t\t",
            "d2\t20260301\t1\tm\tr\t243796009\t<< 243796009\t\t\t\t[[template-sit]]\t\t",
            "d3\t20260301\t1\tm\tr\t413350009\t<< 413350009\t243796009 |Situation with explicit context (situation)|"
            "\t\t\t[[template-fwc]]\t\t",
            "d4\t20260301\t1\tm\tr\t129125009\t<< 129125009\t243796009 |Situation with explicit context (situation)|"
            "\t\t\t\t\t",
        ],
    )
    _write(
        folder / "der2_cissccRefset_MRCMAttributeDomainSnapshot_TEST_20260301.txt",
        _ATTRIBUTE_DOMAIN_HEADER,
        [
            f"a{i}\t20260301\t{active}\tm\tr\t{attr}\t{domain}\t{grouped}\t{card}\t{in_group}\t{MANDATORY}\t{ct}"
            for i, (attr, domain, grouped, card, in_group, active, ct) in enumerate(_ATTRIBUTE_DOMAIN_ROWS)
        ],
    )
    _write(
        folder / "der2_ssccRefset_MRCMAttributeRangeSnapshot_TEST_20260301.txt",
        _ATTRIBUTE_RANGE_HEADER,
        [
            f"r{i}\t20260301\t1\tm\tr\t{attr}\t{ecl}\trule\t{MANDATORY}\t{ct}"
            for i, (attr, ecl, ct) in enumerate(_ATTRIBUTE_RANGE_ROWS)
        ],
    )
    return folder


def make_attributes() -> dict[str, AttributeConfig]:
    attributes = {
        "associated_morphology": (116676008, "Has asso morph"),
        "finding_site": (363698007, "Has finding site"),
        "causative_agent": (246075003, "Has causative agent"),
        "pathological_process": (370135005, "Has pathology"),
        "occurrence": (246454002, "Has occurrence"),
        "clinical_course": (263502005, "Has clinical course"),
        "severity": (246112005, "Has severity"),
        "due_to": (42752001, "Has due to"),
        "during": (371881003, "During"),
        "occurs_after": (255234002, "Occurs after"),
        "finding_asso_with": (47429007, "Finding asso with"),
        "interprets": (363714003, "Has interprets"),
        "interpretation": (363713009, "Has interpretation"),
        "associated_finding": (246090004, "Has asso finding"),
        "finding_context": (408729009, "Has finding context"),
        "temporal_context": (408731000, "Has temporal context"),
        "subject_relationship_context": (408732007, "Has relat context"),
        "episodicity": (246456000, "Has episodicity"),
    }
    return {k: AttributeConfig(attribute_id=a, ohdsi_relationship=r) for k, (a, r) in attributes.items()}


def make_mrcm_settings(folder: str | Path = "data/mrcm") -> MrcmSettings:
    return MrcmSettings(
        folder=str(folder),
        domains={
            "clinical_finding": 404684003,
            "situation_with_explicit_context": 243796009,
            "finding_with_explicit_context": 413350009,
        },
        default_domain="clinical_finding",
        grouping=RoleGroupingConfig(
            self_grouped=["due_to", "clinical_course", "severity", "during", "occurs_after", "finding_asso_with"],
            clusters={
                "core": ["finding_site", "associated_morphology", "pathological_process", "occurrence",
                         "causative_agent"],
                "interpretation": ["interprets", "interpretation"],
                "context": ["associated_finding", "finding_context", "temporal_context",
                            "subject_relationship_context"],
            },
            llm_grouped_clusters=["core"],
            requires={"interpretation": "interprets"},
        ),
    )


@pytest.fixture
def mrcm_settings(mrcm_folder: Path) -> MrcmSettings:
    return make_mrcm_settings(mrcm_folder)


def make_hierarchy_settings(folder: str | Path = "data/mrcm") -> HierarchySettings:
    return HierarchySettings(attributes=make_attributes(), mrcm=make_mrcm_settings(folder))


def concept(concept_id, name=None, code=None) -> dict:
    return {"concept_id": concept_id, "concept_name": name or f"c{concept_id}", "concept_code": code or str(concept_id)}
