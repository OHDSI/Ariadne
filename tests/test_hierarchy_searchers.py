import numpy as np
import pandas as pd
import pytest

from ariadne.hierarchy import searchers
from conftest import make_hierarchy_settings

SETTINGS = make_hierarchy_settings()


class _FakeCursor:
    def __init__(self, connection):
        self._connection = connection

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        self._connection.executed.append((str(query), params))

    def fetchall(self):
        return self._connection.responses.pop(0)


class _FakeConnection:
    """Records executed queries; each ``fetchall`` returns the next queued response."""

    def __init__(self):
        self.executed = []
        self.responses = []
        self.closed = False

    def cursor(self):
        return _FakeCursor(self)

    def close(self):
        self.closed = True


class _FakeConceptSearcher:
    """Stands in for ``PgvectorConceptSearcher``; returns ``hits`` (cosine distances) up to ``max_candidates``."""

    hits: pd.DataFrame | None = None
    cost = 0.0

    def __init__(self, settings):
        self.settings = settings
        self.calls = []

    def search_term(self, term):
        self.calls.append(term)
        return None if self.hits is None else self.hits.head(self.settings.max_candidates)

    def get_total_cost(self):
        return self.cost

    def close(self):
        pass


@pytest.fixture
def connection(monkeypatch):
    connection = _FakeConnection()
    monkeypatch.setenv("VOCAB_CONNECTION_STRING", "postgresql://test")
    monkeypatch.setenv("VOCAB_SCHEMA", "vocab")
    monkeypatch.setattr(searchers.psycopg, "connect", lambda *args, **kwargs: connection)
    monkeypatch.setattr(searchers, "register_vector", lambda conn: None)
    return connection


def _reference_searcher(monkeypatch, hits, cost=0.0, **kwargs):
    monkeypatch.setattr(_FakeConceptSearcher, "hits", hits)
    monkeypatch.setattr(_FakeConceptSearcher, "cost", cost)
    monkeypatch.setattr(searchers, "PgvectorConceptSearcher", _FakeConceptSearcher)
    return searchers.SnomedReferenceConceptVectorSearcher(SETTINGS, **kwargs)


def _hits(*rows):
    return pd.DataFrame(rows, columns=["concept_id", "concept_name", "score"])


def test_reference_searcher_builds_examples_with_attributes(connection, monkeypatch):
    searcher = _reference_searcher(
        monkeypatch, _hits((1001, "Acute myocardial infarction", 0.02), (1002, "Myocardial infarction", 0.05)), cost=0.5
    )
    connection.responses = [[
        (1001, 2001, "12345", "Heart structure", "Has finding site"),
        (1002, 2002, "67890", "Infarction", "Has asso morph"),
    ]]

    result = searcher.search("heart attack", top_k=2)

    assert [e["concept_id"] for e in result.examples] == [1001, 1002]
    assert result.examples[0]["attributes"][0]["target_concept_id"] == 2001
    assert result.cost == 0.5
    assert searcher._concept_searcher.calls == ["heart attack"]
    assert searcher._concept_searcher.settings.max_candidates == SETTINGS.retrieval.num_reference_examples * 2
    assert searcher._concept_searcher.settings.filter.vocabulary_ids == ["SNOMED"]
    assert connection.executed[-1][1] == ([1001, 1002], SETTINGS.snomed_relationships)


def test_reference_searcher_returns_no_examples_without_hits(connection, monkeypatch):
    searcher = _reference_searcher(monkeypatch, None, cost=0.123)

    result = searcher.search("heart attack", top_k=3)

    assert result.examples == []
    assert result.cost == searcher.get_total_cost() == 0.123
    assert len(connection.executed) == 1  # only SET hnsw.ef_search; no attribute query


def test_reference_searcher_applies_exclusions(connection, monkeypatch):
    hits = _hits((1001, "Acute myocardial infarction", 0.01), (1002, "Myocardial infarction", 0.02),
                 (1003, "Cardiac infarction", 0.03))
    searcher = _reference_searcher(monkeypatch, hits, exclude_concept_ids={1001})
    connection.responses = [[]]

    result = searcher.search("heart attack", top_k=2)

    assert [e["concept_id"] for e in result.examples] == [1002, 1003]
    assert [round(e["similarity"], 2) for e in result.examples] == [0.98, 0.97]


def test_attribute_searcher_search_returns_candidates_and_tracks_cost(connection, monkeypatch):
    monkeypatch.setattr(
        searchers, "get_embedding_vectors",
        lambda texts: {"embeddings": np.array([[0.1, 0.2]]), "usage": {"total_cost_usd": 0.01}},
    )
    connection.responses = [[(1, "C1", "Heart structure", "Has finding site", 0.9)]]

    with searchers.SnomedAttributeSearcher(SETTINGS) as searcher:
        candidates, cost = searcher.search("heart", "Has finding site")
        assert searcher.get_total_cost() == 0.01

    assert connection.closed
    assert connection.executed[0] == (f"SET hnsw.ef_search = {SETTINGS.retrieval.hnsw_ef_search}", None)
    assert connection.executed[1][1] == ([0.1, 0.2], "Has finding site", SETTINGS.retrieval.top_k_per_category)
    assert candidates.to_dict("records") == [{
        "concept_id": 1, "concept_code": "C1", "concept_name": "Heart structure",
        "attribute_category": "Has finding site", "similarity": 0.9,
    }]
    assert cost == 0.01


def test_attribute_searcher_search_batch_embeds_once_and_returns_frame_per_key(connection, monkeypatch):
    embedded = []

    def fake_embeddings(texts):
        embedded.append(texts)
        return {"embeddings": np.array([[0.1], [0.2]]), "usage": {"total_cost_usd": 0.02}}

    monkeypatch.setattr(searchers, "get_embedding_vectors", fake_embeddings)
    connection.responses = [[(1, "C1", "Heart structure", "Has finding site", 0.9)], []]
    searcher = searchers.SnomedAttributeSearcher(SETTINGS)

    results, cost = searcher.search_batch(
        [("finding_site_0", "heart", "Has finding site"), ("associated_morphology_1", "tear", "Has asso morph")],
        top_k=3,
    )

    assert embedded == [["heart", "tear"]]
    assert cost == 0.02
    assert [params for _, params in connection.executed[1:]] == [
        ([0.1], "Has finding site", 3), ([0.2], "Has asso morph", 3)]
    assert results["finding_site_0"]["concept_id"].tolist() == [1]
    assert results["associated_morphology_1"].empty
    assert list(results["associated_morphology_1"].columns) == list(results["finding_site_0"].columns)
    assert searcher.search_batch([]) == ({}, 0.0)
    assert len(embedded) == 1
