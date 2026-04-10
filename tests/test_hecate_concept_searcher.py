from ariadne.vector_search.hecate_concept_searcher import (
    HecateConceptSearcher,
    _HECATE_SEARCH_STANDARD_URL,
    _HECATE_SEARCH_URL,
)


def test_non_standard_uses_search_endpoint():
    searcher = HecateConceptSearcher(standard_concept="None")
    endpoint, params = searcher._resolve_endpoint_for_standard_concept(None)

    assert endpoint == _HECATE_SEARCH_URL
    assert params.get("standard_concept") == "None"


def test_standard_override_uses_search_standard_endpoint():
    searcher = HecateConceptSearcher(standard_concept="None")
    endpoint, params = searcher._resolve_endpoint_for_standard_concept("S")

    assert endpoint == _HECATE_SEARCH_STANDARD_URL
    assert "standard_concept" not in params
