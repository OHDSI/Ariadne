import pandas as pd
import pyarrow.parquet as pq

from ariadne.utils.settings import VerbatimMappingSettings
from ariadne.verbatim_mapping.term_downloader import _store_in_parquet
from ariadne.verbatim_mapping.vocab_verbatim_term_mapper import VocabVerbatimTermMapper


class _IdentityNormalizer:
    def __init__(self, substrings_to_remove=None):
        self.substrings_to_remove = substrings_to_remove or []

    def normalize_terms(self, terms):
        return [term.lower() for term in terms]

    def normalize_term(self, term):
        return term.lower()


def test_map_term_prefers_first_matching_vocabulary(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "ariadne.verbatim_mapping.vocab_verbatim_term_mapper.TermNormalizer",
        _IdentityNormalizer,
    )

    terms_file = tmp_path / "terms.parquet"
    pd.DataFrame(
        {
            "term": ["Pain", "Pain"],
            "concept_id": [1, 2],
            "concept_name": ["Pain SNOMED", "Pain ICD10CM"],
            "vocabulary_id": ["SNOMED", "ICD10CM"],
        }
    ).to_parquet(terms_file, index=False)

    settings = VerbatimMappingSettings(
        terms_folder=str(tmp_path),
        verbatim_mapping_index_file=str(tmp_path / "index.pkl"),
        preferred_vocabulary_ids=["ICD10CM", "SNOMED"],
    )

    mapper = VocabVerbatimTermMapper(settings=settings)

    assert mapper.map_term("Pain") == [(2, "Pain ICD10CM")]


def test_map_term_keeps_existing_behavior_when_no_preferred_vocabularies(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "ariadne.verbatim_mapping.vocab_verbatim_term_mapper.TermNormalizer",
        _IdentityNormalizer,
    )

    terms_file = tmp_path / "terms.parquet"
    pd.DataFrame(
        {
            "term": ["Pain", "Pain"],
            "concept_id": [1, 2],
            "concept_name": ["Pain SNOMED", "Pain ICD10CM"],
            "vocabulary_id": ["SNOMED", "ICD10CM"],
        }
    ).to_parquet(terms_file, index=False)

    settings = VerbatimMappingSettings(
        terms_folder=str(tmp_path),
        verbatim_mapping_index_file=str(tmp_path / "index.pkl"),
        preferred_vocabulary_ids=[],
    )

    mapper = VocabVerbatimTermMapper(settings=settings)

    assert mapper.map_term("Pain") == [(1, "Pain SNOMED"), (2, "Pain ICD10CM")]


def test_map_terms_returns_single_row_with_preferred_concept(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "ariadne.verbatim_mapping.vocab_verbatim_term_mapper.TermNormalizer",
        _IdentityNormalizer,
    )

    terms_file = tmp_path / "terms.parquet"
    pd.DataFrame(
        {
            "term": ["Pain", "Pain"],
            "concept_id": [1, 2],
            "concept_name": ["Pain SNOMED", "Pain ICD10CM"],
            "vocabulary_id": ["SNOMED", "ICD10CM"],
        }
    ).to_parquet(terms_file, index=False)

    settings = VerbatimMappingSettings(
        terms_folder=str(tmp_path),
        verbatim_mapping_index_file=str(tmp_path / "index.pkl"),
        preferred_vocabulary_ids=["ICD10CM", "SNOMED"],
    )
    mapper = VocabVerbatimTermMapper(settings=settings)

    source_df = pd.DataFrame({"cleaned_term": ["Pain"]})
    mapped = mapper.map_terms(source_df.copy())

    assert len(mapped) == 1
    assert mapped.iloc[0]["mapped_concept_id"] == 2
    assert mapped.iloc[0]["mapped_concept_name"] == "Pain ICD10CM"


def test_store_in_parquet_writes_vocabulary_id_column(tmp_path):
    file_name = tmp_path / "terms.parquet"
    _store_in_parquet(
        concept_ids=[1],
        terms=["Pain"],
        concept_names=["Pain concept"],
        vocabulary_ids=["SNOMED"],
        file_name=str(file_name),
    )

    table = pq.read_table(file_name)
    assert table.column_names == ["concept_id", "term", "concept_name", "vocabulary_id"]
    assert table.column("vocabulary_id").to_pylist() == ["SNOMED"]

