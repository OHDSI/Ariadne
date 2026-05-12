from types import SimpleNamespace

from ariadne.hierarchy.searchers import SnomedReferenceConceptVectorSearcher


class _FakeCursor:
    def __init__(self, responses, executed):
        self._responses = responses
        self._executed = executed
        self._current = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        self._executed.append((str(query), params))
        self._current = self._responses.pop(0)

    def fetchall(self):
        return self._current


class _FakeConnection:
    def __init__(self, responses, executed):
        self._responses = responses
        self._executed = executed

    def cursor(self):
        return _FakeCursor(self._responses, self._executed)


class _FakeConceptSearcher:
    def __init__(self, *args, **kwargs):
        self.calls = []
        self.cost = 0.5

    def search_term(self, term, limit, vocabulary_id=None):
        self.calls.append((term, limit, vocabulary_id))
        import pandas as pd

        return pd.DataFrame(
            [
                (1001, "Acute myocardial infarction", 0.02),
                (1002, "Myocardial infarction", 0.05),
            ],
            columns=["concept_id", "concept_name", "score"],
        )

    def get_total_cost(self):
        return self.cost

    def close(self):
        return None


def test_reference_concept_vector_searcher_uses_pgvector_concept_searcher(monkeypatch):
    executed = []
    responses = [
        [
            (1001, 2001, "12345", "Heart structure", "Has finding site (SNOMED)"),
            (1002, 2002, "67890", "Infarction morphology", "Has associated morphology (SNOMED)"),
        ],
    ]
    fake_conn = _FakeConnection(responses, executed)

    def _fake_base_init(self, cfg=None):
        self.cfg = cfg or SimpleNamespace(
            retrieval=SimpleNamespace(num_reference_examples=5),
            snomed_relationships=["Has finding site", "Has asso morph"],
        )
        self.connection = fake_conn
        self.schema = "vocab"
        self._cost = 0.0

    monkeypatch.setattr("ariadne.hierarchy.searchers.AbstractSnomedSearcher.__init__", _fake_base_init)
    monkeypatch.setattr("ariadne.hierarchy.searchers.PgvectorConceptSearcher", _FakeConceptSearcher)
    searcher = SnomedReferenceConceptVectorSearcher()
    result = searcher.search("unused", top_k=2)

    assert result.cost == 0.5
    assert len(result.examples) == 2
    assert result.examples[0]["concept_id"] == 1001
    assert result.examples[0]["attributes"][0]["concept_id_2"] == 2001
    assert len(executed) == 1
    assert len(searcher._concept_searcher.calls) == 1
    term, limit, vocabulary_id = searcher._concept_searcher.calls[0]
    assert term == "unused"
    assert limit == 10
    assert vocabulary_id == "SNOMED"


def test_reference_concept_vector_searcher_embeds_when_needed(monkeypatch):
    executed = []
    responses = [[]]
    fake_conn = _FakeConnection(responses, executed)

    def _fake_base_init(self, cfg=None):
        self.cfg = cfg or SimpleNamespace(
            retrieval=SimpleNamespace(num_reference_examples=5),
            snomed_relationships=["Has finding site"],
        )
        self.connection = fake_conn
        self.schema = "vocab"
        self._cost = 0.0

    monkeypatch.setattr("ariadne.hierarchy.searchers.AbstractSnomedSearcher.__init__", _fake_base_init)
    class _FakeEmptyConceptSearcher:
        def __init__(self, *args, **kwargs):
            pass

        def search_term(self, term, limit, vocabulary_id=None):
            return None

        def get_total_cost(self):
            return 0.123

        def close(self):
            return None

    monkeypatch.setattr("ariadne.hierarchy.searchers.PgvectorConceptSearcher", _FakeEmptyConceptSearcher)
    searcher = SnomedReferenceConceptVectorSearcher()
    result = searcher.search("heart attack", top_k=3)

    assert result.examples == []
    assert result.cost == 0.123
    assert searcher.get_total_cost() == 0.123
    assert len(executed) == 0


def test_reference_concept_vector_searcher_applies_exclusions(monkeypatch):
    executed = []
    responses = [[(1002, 2002, "67890", "Infarction morphology", "Has associated morphology (SNOMED)")]]
    fake_conn = _FakeConnection(responses, executed)

    def _fake_base_init(self, cfg=None):
        self.cfg = cfg or SimpleNamespace(
            retrieval=SimpleNamespace(num_reference_examples=5),
            snomed_relationships=["Has finding site", "Has asso morph"],
        )
        self.connection = fake_conn
        self.schema = "vocab"
        self._cost = 0.0

    class _FakeConceptSearcherWithExcludedTop:
        def __init__(self, *args, **kwargs):
            import pandas as pd

            self._df = pd.DataFrame(
                [
                    (1001, "Acute myocardial infarction", 0.01),
                    (1002, "Myocardial infarction", 0.02),
                    (1003, "Cardiac infarction", 0.03),
                ],
                columns=["concept_id", "concept_name", "score"],
            )

        def search_term(self, term, limit, vocabulary_id=None):
            return self._df.head(limit)

        def get_total_cost(self):
            return 0.0

        def close(self):
            return None

    monkeypatch.setattr("ariadne.hierarchy.searchers.AbstractSnomedSearcher.__init__", _fake_base_init)
    monkeypatch.setattr("ariadne.hierarchy.searchers.PgvectorConceptSearcher", _FakeConceptSearcherWithExcludedTop)

    searcher = SnomedReferenceConceptVectorSearcher(exclude_concept_ids={1001})
    result = searcher.search("heart attack", top_k=2)

    assert [row["concept_id"] for row in result.examples] == [1002, 1003]
    assert len(executed) == 1


