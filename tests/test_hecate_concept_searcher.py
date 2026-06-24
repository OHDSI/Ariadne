from ariadne.vector_search.hecate_concept_searcher import (
    HecateConceptSearcher,
    _HECATE_SEARCH_STANDARD_URL,
    _HECATE_SEARCH_URL,
)
from ariadne.utils.settings import ConceptFilterSettings, VectorSearchSettings


def test_non_standard_uses_search_endpoint():
    searcher = HecateConceptSearcher(
        settings=VectorSearchSettings(filter=ConceptFilterSettings(standard_concept=["None"]))
    )

    assert searcher.default_url == _HECATE_SEARCH_URL
    assert searcher.default_params.get("standard_concept") == "None"


def test_standard_override_uses_search_standard_endpoint():
    searcher = HecateConceptSearcher(
        settings=VectorSearchSettings(filter=ConceptFilterSettings(standard_concept=["S"]))
    )

    assert searcher.default_url == _HECATE_SEARCH_STANDARD_URL
    assert "standard_concept" not in searcher.default_params


def test_exclude_vocabularies_are_joined_for_hecate_query_params():
    searcher = HecateConceptSearcher(
        settings=VectorSearchSettings(
            filter=ConceptFilterSettings(
                standard_concept=["S"],
                exclude_vocabulary_ids=["ICD10", "Read"],
            )
        )
    )

    assert searcher.default_params.get("exclude_vocabulary_id") == "ICD10,Read"


def test_mixed_standard_values_use_search_endpoint_with_explicit_param():
    searcher = HecateConceptSearcher(
        settings=VectorSearchSettings(
            filter=ConceptFilterSettings(standard_concept=["S", "None"])
        )
    )

    assert searcher.default_url == _HECATE_SEARCH_URL
    assert searcher.default_params.get("standard_concept") == "S,None"

