import pandas as pd
import pytest

from ariadne.llm_mapping.source_context_retriever import SourceContextRetriever


def _context_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_vocabulary_id": ["ICD10CM", "ICD9CM"],
            "source_concept_code": ["N76.8", "474.02"],
            "source_parents": ["Inflammatory disease;Genital disorder", "Tonsil disorder"],
            "source_children": ["", "Chronic adenoiditis"],
            "source_description": ["Inflammation of vagina and vulva", "Tonsillitis and adenoiditis"],
            "source_synonyms": ["Vulvovaginal inflammation", ""],
        }
    )


def test_add_source_context_merges_on_composite_key():
    retriever = SourceContextRetriever(_context_frame())
    candidates = pd.DataFrame(
        {
            "source_vocabulary_id": ["ICD10CM"],
            "source_concept_code": ["N76.8"],
            "source_term": ["Other specified inflammation of vagina and vulva"],
        }
    )

    enriched = retriever.add_source_context(candidates)

    row = enriched.iloc[0]
    assert row["source_parents"] == "Inflammatory disease;Genital disorder"
    assert row["source_description"] == "Inflammation of vagina and vulva"
    assert row["source_synonyms"] == "Vulvovaginal inflammation"


def test_add_source_context_unmatched_rows_are_empty():
    retriever = SourceContextRetriever(_context_frame())
    candidates = pd.DataFrame(
        {
            "source_vocabulary_id": ["SNOMED"],
            "source_concept_code": ["12345"],
            "source_term": ["something else"],
        }
    )

    enriched = retriever.add_source_context(candidates)

    row = enriched.iloc[0]
    assert row["source_parents"] == ""
    assert row["source_children"] == ""
    assert row["source_description"] == ""
    assert row["source_synonyms"] == ""


def test_add_source_context_creates_missing_context_columns():
    context = _context_frame().drop(columns=["source_description"])
    retriever = SourceContextRetriever(context)
    candidates = pd.DataFrame(
        {
            "source_vocabulary_id": ["ICD10CM"],
            "source_concept_code": ["N76.8"],
        }
    )

    enriched = retriever.add_source_context(candidates)

    assert enriched.iloc[0]["source_description"] == ""
    assert enriched.iloc[0]["source_parents"] == "Inflammatory disease;Genital disorder"


def test_add_source_context_overwrites_preexisting_context_columns():
    retriever = SourceContextRetriever(_context_frame())
    # Candidate already carries empty source_* columns (e.g. carried through the pipeline).
    candidates = pd.DataFrame(
        {
            "source_vocabulary_id": ["ICD10CM"],
            "source_concept_code": ["N76.8"],
            "source_parents": [""],
            "source_children": [""],
        }
    )

    enriched = retriever.add_source_context(candidates)

    assert "source_parents_x" not in enriched.columns
    assert "source_parents_y" not in enriched.columns
    assert enriched.iloc[0]["source_parents"] == "Inflammatory disease;Genital disorder"


def test_add_source_context_raises_on_missing_candidate_keys():    retriever = SourceContextRetriever(_context_frame())
    candidates = pd.DataFrame({"source_vocabulary_id": ["ICD10CM"]})

    with pytest.raises(ValueError, match="source_concept_code"):
        retriever.add_source_context(candidates)


def test_constructor_raises_on_missing_key_columns():
    context = _context_frame().drop(columns=["source_concept_code"])

    with pytest.raises(ValueError, match="source_concept_code"):
        SourceContextRetriever(context)


def test_reads_from_csv(tmp_path):
    csv_path = tmp_path / "source_context.csv"
    _context_frame().to_csv(csv_path, index=False)
    retriever = SourceContextRetriever(csv_path)

    candidates = pd.DataFrame(
        {
            "source_vocabulary_id": ["ICD9CM"],
            "source_concept_code": ["474.02"],
        }
    )
    enriched = retriever.add_source_context(candidates)

    assert enriched.iloc[0]["source_children"] == "Chronic adenoiditis"
