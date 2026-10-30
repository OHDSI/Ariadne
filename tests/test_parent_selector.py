from pathlib import Path

import pandas as pd
import pytest
import yaml

from ariadne.evaluation import stated_parent_evaluator as evaluator
from ariadne.hierarchy import parent_selector as ps
from ariadne.hierarchy.parent_selector import (
    Neighbour,
    PrimitiveAncestry,
    PrimitiveConcept,
    proximal_primitive_parents,
    select_primitive_parents,
)
from ariadne.utils.settings import HierarchySettings, ParentSelectionConfig, build_dataclass

# Toy allowed parent set (primitives) and fully defined neighbours:
#
#   1 Clinical finding (root)
#   ├── 2 Disease
#   │   ├── 3 Disorder of larynx      ← 101, 102
#   │   └── 4 Neoplastic disease      ← 103
#   └── 5 Finding by site             ← 104
CF, DISEASE, LARYNX, NEOPLASM, BY_SITE = 1, 2, 3, 4, 5
PRIMITIVES = {
    CF: PrimitiveConcept(CF, "404684003", "Clinical finding", 0),
    DISEASE: PrimitiveConcept(DISEASE, "64572001", "Disease", 1),
    LARYNX: PrimitiveConcept(LARYNX, "1003", "Disorder of larynx", 2),
    NEOPLASM: PrimitiveConcept(NEOPLASM, "1004", "Neoplastic disease", 2),
    BY_SITE: PrimitiveConcept(BY_SITE, "1005", "Finding by site", 1),
}
ANCESTORS = {
    CF: set(),
    DISEASE: {CF},
    LARYNX: {CF, DISEASE},
    NEOPLASM: {CF, DISEASE},
    BY_SITE: {CF},
    101: {CF, DISEASE, LARYNX},
    102: {CF, DISEASE, LARYNX},
    103: {CF, DISEASE, NEOPLASM},
    104: {CF, BY_SITE},
}


@pytest.fixture
def ancestry() -> PrimitiveAncestry:
    return PrimitiveAncestry(concepts=dict(PRIMITIVES), ancestors={k: set(v) for k, v in ANCESTORS.items()})


def nb(concept_id: int, similarity: float = 0.9) -> Neighbour:
    return Neighbour(concept_id, f"c{concept_id}", similarity)


def ids(parents) -> list[int]:
    return [p.concept.concept_id for p in parents]


# ── select_primitive_parents ────────────────────────────────────────────────


def test_agreeing_neighbours_give_proximal_primitive_without_its_ancestors(ancestry):
    parents = select_primitive_parents([nb(101), nb(102)], ancestry, min_support=0.5, max_parents=2)
    assert ids(parents) == [LARYNX]
    assert parents[0].support == pytest.approx(1.0)


def test_split_neighbours_fall_back_to_shared_intermediate_primitive(ancestry):
    parents = select_primitive_parents([nb(101), nb(103)], ancestry, min_support=0.67, max_parents=2)
    assert ids(parents) == [DISEASE]


def test_low_threshold_keeps_both_branches_up_to_max_parents(ancestry):
    neighbours = [nb(101, 0.9), nb(103, 0.8)]
    assert set(ids(select_primitive_parents(neighbours, ancestry, min_support=0.4, max_parents=2))) == {
        LARYNX,
        NEOPLASM,
    }
    # Same depth: higher support wins the single slot.
    assert ids(select_primitive_parents(neighbours, ancestry, min_support=0.4, max_parents=1)) == [LARYNX]


def test_deepest_parent_comes_first(ancestry):
    parents = select_primitive_parents([nb(104, 0.9), nb(101, 0.8)], ancestry, min_support=0.4, max_parents=2)
    assert ids(parents) == [LARYNX, BY_SITE]


def test_unrelated_neighbours_only_share_the_root(ancestry):
    parents = select_primitive_parents([nb(101), nb(104)], ancestry, min_support=0.67, max_parents=2)
    assert ids(parents) == [CF]


def test_primitive_neighbour_votes_for_its_ancestors_not_itself(ancestry):
    parents = select_primitive_parents([nb(LARYNX)], ancestry, min_support=0.5, max_parents=2)
    assert ids(parents) == [DISEASE]


def test_no_neighbours_or_neighbours_outside_root_select_nothing(ancestry):
    assert select_primitive_parents([], ancestry, min_support=0.5, max_parents=2) == []
    assert select_primitive_parents([nb(999)], ancestry, min_support=0.5, max_parents=2) == []


def test_proximal_primitive_parents(ancestry):
    assert [c.concept_id for c in proximal_primitive_parents(101, ancestry)] == [LARYNX]
    assert proximal_primitive_parents(CF, ancestry) == []


# ── build_stated_parents_map ────────────────────────────────────────────────


@pytest.fixture
def patched_vocab(monkeypatch, ancestry):
    calls = {}

    def fake_descendants(ancestor_ids, candidate_ids, conn, schema):
        calls["descendants"] = (sorted(ancestor_ids), sorted(candidate_ids))
        return {500: {500, 501}}

    def fake_ancestry(concept_ids, conn, schema, settings):
        calls["ancestry"] = set(concept_ids)
        return ancestry

    monkeypatch.setattr(ps, "fetch_descendants", fake_descendants)
    monkeypatch.setattr(ps, "fetch_primitive_ancestry", fake_ancestry)
    monkeypatch.setattr(ps, "fetch_root_concept", lambda conn, schema, settings: PRIMITIVES[CF])
    return calls


def make_settings(**overrides) -> ParentSelectionConfig:
    values = {"num_neighbours": 3, "candidate_pool_size": 10, "neighbour_cache_file": ""}
    values.update(overrides)
    return ParentSelectionConfig(**values)


def test_build_map_excludes_source_and_descendants_and_falls_back(patched_vocab):
    raw_results = [
        {"source_concept_id": 500, "source_term": "larynx thing"},
        {"source_concept_id": 600, "source_term": "orphan"},
        {"source_concept_id": None, "source_key": "ICD10:X1", "source_term": "new term"},
    ]
    neighbours = {
        # Self (500) and a descendant (501) would leak the answer; 102 at 0.3 is below min_similarity.
        "larynx thing": [nb(500, 1.0), nb(501, 0.95), nb(101), nb(102, 0.3), nb(103, 0.85), nb(102, 0.84)],
        "orphan": [],
        "new term": [nb(104), nb(104)],
    }

    stated, details = ps.build_stated_parents_map(raw_results, None, "vocab", make_settings(), neighbours=neighbours)

    assert stated == {500: ["1003"], 600: ["404684003"], "ICD10:X1": ["1005"]}
    assert 500 not in patched_vocab["ancestry"] and 501 not in patched_vocab["ancestry"]
    rows = details.set_index("source_concept_id")
    assert rows.loc[500, "num_neighbours"] == 3
    assert not rows.loc[500, "is_fallback"]
    assert rows.loc[600, "is_fallback"]
    assert list(details.columns) == [
        "source_concept_id", "source_term", "parent_rank", "parent_concept_id", "parent_concept_code",
        "parent_concept_name", "parent_depth", "support", "num_neighbours", "is_fallback",
    ]


def test_build_map_without_leakage_guard_keeps_source_neighbours(patched_vocab):
    raw_results = [{"source_concept_id": 101, "source_term": "t"}]
    neighbours = {"t": [nb(LARYNX)]}

    stated, _ = ps.build_stated_parents_map(
        raw_results, None, "vocab", make_settings(exclude_source_descendants=False), neighbours=neighbours
    )

    assert "descendants" not in patched_vocab
    assert stated == {101: ["64572001"]}


# ── neighbour retrieval and cache ───────────────────────────────────────────


class FakeSearcher:
    calls: list[str] = []

    def __init__(self, settings):
        self.settings = settings

    def search_term(self, term):
        FakeSearcher.calls.append(term)
        return pd.DataFrame({"concept_id": [101], "concept_name": ["c101"], "score": [0.25]})

    def get_total_cost(self):
        return 0.0

    def close(self):
        pass


def test_retrieve_neighbours_uses_cache(monkeypatch, tmp_path: Path):
    FakeSearcher.calls = []
    monkeypatch.setattr(ps, "PgvectorConceptSearcher", FakeSearcher)
    settings = make_settings(neighbour_cache_file=str(tmp_path / "cache.json"), max_workers=2)

    first = ps.retrieve_neighbours(["a", "b", "a"], settings)
    second = ps.retrieve_neighbours(["a", "b"], settings)

    assert sorted(FakeSearcher.calls) == ["a", "b"]
    assert first["a"] == second["a"] == [Neighbour(101, "c101", 0.75)]


def test_neighbour_cache_ignored_when_pool_size_changes(monkeypatch, tmp_path: Path):
    FakeSearcher.calls = []
    monkeypatch.setattr(ps, "PgvectorConceptSearcher", FakeSearcher)
    cache_file = str(tmp_path / "cache.json")

    ps.retrieve_neighbours(["a"], make_settings(neighbour_cache_file=cache_file, candidate_pool_size=10))
    ps.retrieve_neighbours(["a"], make_settings(neighbour_cache_file=cache_file, candidate_pool_size=20))

    assert FakeSearcher.calls == ["a", "a"]


# ── evaluation ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "predicted, expected, outcome",
    [
        ({LARYNX}, {LARYNX}, "exact"),
        ({LARYNX, DISEASE}, {LARYNX}, "partial"),
        ({DISEASE}, {LARYNX}, "too_general"),
        ({CF}, {LARYNX}, "too_general"),
        ({NEOPLASM}, {LARYNX}, "wrong"),
        ({LARYNX, BY_SITE}, {LARYNX}, "wrong"),
        ({CF}, set(), "no_reference"),
    ],
)
def test_classify_parents(ancestry, predicted, expected, outcome):
    assert evaluator.classify_parents(predicted, expected, ancestry) == outcome


def test_evaluate_stated_parents(monkeypatch, ancestry):
    monkeypatch.setattr(evaluator, "fetch_primitive_ancestry", lambda ids_, conn, schema, settings: ancestry)
    details = pd.DataFrame(
        [
            {"source_concept_id": 101, "source_term": "a", "parent_concept_id": LARYNX, "parent_concept_name": "L"},
            {"source_concept_id": 103, "source_term": "b", "parent_concept_id": CF, "parent_concept_name": "CF"},
        ]
    )

    result = evaluator.evaluate_stated_parents(details, None, "vocab", make_settings())

    assert result.set_index("source_concept_id")["outcome"].to_dict() == {101: "exact", 103: "too_general"}
    assert result.set_index("source_concept_id").loc[103, "reference_parents"] == "Neoplastic disease"


# ── settings ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_neighbours": 0},
        {"num_neighbours": 20, "candidate_pool_size": 10},
        {"min_support": 0.0},
        {"min_support": 1.5},
        {"max_parents": 0},
    ],
)
def test_parent_selection_config_validation(overrides):
    with pytest.raises(ValueError):
        make_settings(**overrides)


def test_repo_config_parent_selection_block_loads():
    config_path = Path(__file__).resolve().parents[1] / "config_condition_mapping.yaml"
    hierarchy = build_dataclass(HierarchySettings, yaml.safe_load(config_path.read_text())["hierarchy"])

    settings = hierarchy.parent_selection
    assert settings.num_neighbours == 10
    assert settings.root_concept_id == 441840
    assert settings.primitive_status_concept_id == 40642538
    assert Path(settings.neighbour_cache_file).is_absolute()
