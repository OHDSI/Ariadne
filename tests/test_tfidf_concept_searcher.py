from pathlib import Path

import pandas as pd

from ariadne.utils.settings import ConceptFilterSettings, TfidfSearchSettings
from ariadne.vector_search.tfidf_concept_searcher import TfidfConceptSearcher


def _write_terms_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_tfidf_searcher_deduplicates_concepts_and_builds_index(tmp_path):
    terms_folder = tmp_path / "tfidf_terms"
    index_file = tmp_path / "tfidf_index.pkl"

    _write_terms_parquet(
        terms_folder / "terms.parquet",
        [
            {
                "concept_id": 100,
                "term": "Acme Pharma Ltd",
                "concept_name": "Acme Pharma Ltd",
                "vocabulary_id": "RxNorm Extension",
            },
            {
                "concept_id": 100,
                "term": "Acme Pharmaceuticals",
                "concept_name": "Acme Pharma Ltd",
                "vocabulary_id": "RxNorm Extension",
            },
            {
                "concept_id": 200,
                "term": "Kent Pharma UK",
                "concept_name": "Kent Pharma UK",
                "vocabulary_id": "RxNorm Extension",
            },
        ],
    )

    settings = TfidfSearchSettings(
        terms_folder=str(terms_folder),
        tfidf_index_file=str(index_file),
        max_candidates=10,
        substrings_to_remove=["ltd"],
        filter=ConceptFilterSettings(standard_concept=["None"]),
    )

    searcher = TfidfConceptSearcher(settings=settings)
    result = searcher.search_term("Acme Pharma Ltd")

    assert index_file.exists()
    assert not result.empty
    assert result.iloc[0]["concept_id"] == 100
    # Same concept can have multiple source terms, but should appear only once.
    assert result["concept_id"].tolist().count(100) == 1


def test_tfidf_searcher_orders_ties_by_concept_id(tmp_path):
    terms_folder = tmp_path / "tfidf_terms"
    index_file = tmp_path / "tfidf_index.pkl"

    _write_terms_parquet(
        terms_folder / "terms.parquet",
        [
            {
                "concept_id": 2,
                "term": "shared supplier",
                "concept_name": "Supplier B",
                "vocabulary_id": "RxNorm Extension",
            },
            {
                "concept_id": 1,
                "term": "shared supplier",
                "concept_name": "Supplier A",
                "vocabulary_id": "RxNorm Extension",
            },
            {
                "concept_id": 3,
                "term": "different supplier",
                "concept_name": "Supplier C",
                "vocabulary_id": "RxNorm Extension",
            },
        ],
    )

    settings = TfidfSearchSettings(
        terms_folder=str(terms_folder),
        tfidf_index_file=str(index_file),
        max_candidates=2,
        filter=ConceptFilterSettings(standard_concept=["None"]),
    )

    searcher = TfidfConceptSearcher(settings=settings)
    result = searcher.search_term("shared supplier")

    assert result["concept_id"].tolist() == [1, 2]

