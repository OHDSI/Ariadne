import pandas as pd
from sqlalchemy import create_engine

from ariadne.llm_mapping import concept_context_retriever


def test_add_concept_context_adds_clinical_drug_form_child_count(tmp_path, monkeypatch):
    db_path = tmp_path / "vocab.db"
    engine = create_engine(f"sqlite:///{db_path}")

    with engine.begin() as connection:
        connection.exec_driver_sql(
            """
            CREATE TABLE concept (
                concept_id INTEGER PRIMARY KEY,
                concept_name TEXT,
                concept_class_id TEXT,
                domain_id TEXT,
                vocabulary_id TEXT,
                standard_concept TEXT
            )
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TABLE concept_synonym (
                concept_id INTEGER,
                concept_synonym_name TEXT,
                language_concept_id INTEGER
            )
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TABLE concept_ancestor (
                ancestor_concept_id INTEGER,
                descendant_concept_id INTEGER,
                min_levels_of_separation INTEGER
            )
            """
        )

        connection.exec_driver_sql(
            """
            INSERT INTO concept (concept_id, concept_name, concept_class_id, domain_id, vocabulary_id, standard_concept)
            VALUES
              (1, 'Aspirin', 'Ingredient', 'Drug', 'RxNorm', 'S'),
              (2, 'Aspirin oral tablet', 'Clinical Drug Form', 'Drug', 'RxNorm', 'S'),
              (3, 'Aspirin oral capsule', 'Clinical Drug Form', 'Drug', 'RxNorm Extension', 'S'),
              (4, 'Aspirin non-standard form', 'Clinical Drug Form', 'Drug', 'RxNorm', 'C')
            """
        )
        connection.exec_driver_sql(
            """
            INSERT INTO concept_ancestor (ancestor_concept_id, descendant_concept_id, min_levels_of_separation)
            VALUES
              (1, 2, 1),
              (1, 3, 2),
              (1, 4, 1)
            """
        )

    def fake_get_environment_variable(name):
        if name == "VOCAB_CONNECTION_STRING":
            return f"sqlite:///{db_path}"
        if name == "VOCAB_SCHEMA":
            return None
        raise KeyError(name)

    monkeypatch.setattr(concept_context_retriever, "get_environment_variable", fake_get_environment_variable)

    concept_table = pd.DataFrame({"matched_concept_id": [1]})
    enriched = concept_context_retriever.add_concept_context(
        concept_table=concept_table,
        add_parents=False,
        add_children=False,
        add_synonyms=False,
        add_clinical_drug_form_child_count=True,
    )

    assert "matched_clinical_drug_form_child_count" in enriched.columns
    assert int(enriched.iloc[0]["matched_clinical_drug_form_child_count"]) == 2

