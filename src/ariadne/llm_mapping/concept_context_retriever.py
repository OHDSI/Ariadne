from typing import List

import pandas as pd
from ariadne.utils.utils import get_environment_variable
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData, Table, select, func, cast, literal, Text, and_, literal_column
from sqlalchemy.engine import Engine

load_dotenv()


def _create_query(
    concept_ids: List[int],
    domain_id_column: str,
    concept_class_id_column: str,
    vocabulary_id_column: str,
    add_parents: bool,
    parents_column: str,
    add_children: bool,
    children_column: str,
    add_synonyms: bool,
    synonyms_column: str,
    engine: Engine,
):
    vocabulary_schema = get_environment_variable("VOCAB_SCHEMA")

    metadata = MetaData()
    concept = Table("concept", metadata, schema=vocabulary_schema, autoload_with=engine)
    concept_synonym = Table("concept_synonym", metadata, schema=vocabulary_schema, autoload_with=engine)
    concept_ancestor = Table("concept_ancestor", metadata, schema=vocabulary_schema, autoload_with=engine)

    if add_parents:
        parent_concept = concept.alias("parent_concept")
        parents_sq = (
            select(
                concept_ancestor.c.descendant_concept_id.label("concept_id"),
                func.string_agg(parent_concept.c.concept_name, literal(";")).label("parent_names"),
            )
            .select_from(concept_ancestor)
            .where(concept_ancestor.c.min_levels_of_separation == 1)
            .join(parent_concept, concept_ancestor.c.ancestor_concept_id == parent_concept.c.concept_id)
            .group_by(concept_ancestor.c.descendant_concept_id)
            .alias("parent_names")
        )
        parent_names = parents_sq.columns

    if add_children:
        child_concept = concept.alias("child_concept")
        limited_children_sq = (
            select(
                concept_ancestor.c.ancestor_concept_id.label("concept_id"),
                child_concept.c.concept_name,
                func.row_number()
                .over(partition_by=concept_ancestor.c.ancestor_concept_id, order_by=func.random())
                .label("rn"),
            )
            .select_from(concept_ancestor)
            .where(concept_ancestor.c.min_levels_of_separation == 1)
            .join(child_concept, concept_ancestor.c.descendant_concept_id == child_concept.c.concept_id)
            .alias("limited_children")
        )
        children_Sq = (
            select(
                limited_children_sq.c.concept_id,
                func.string_agg(limited_children_sq.c.concept_name, literal(";")).label("child_names"),
            )
            .where(limited_children_sq.c.rn <= 10)
            .group_by(limited_children_sq.c.concept_id)
            .alias("child_names")
        )
        child_names = children_Sq.columns

    if add_synonyms:
        CONCEPT_SYNONYM_ENGLISH_ID = 4180186
        synonyms_sq = (
            select(
                concept_synonym.c.concept_id,
                func.string_agg(concept_synonym.c.concept_synonym_name, literal(";")).label("synonym_names"),
            )
            .select_from(concept_synonym)
            .where(concept_synonym.c.language_concept_id == CONCEPT_SYNONYM_ENGLISH_ID)
            .group_by(concept_synonym.c.concept_id)
            .alias("synonym_names")
        )
        synonym_names = synonyms_sq.columns

    select_columns = [
        concept.c.concept_id,
        concept.c.concept_class_id.label(concept_class_id_column),
        concept.c.domain_id.label(domain_id_column),
        concept.c.vocabulary_id.label(vocabulary_id_column)
    ]
    if add_parents:
        select_columns.append(func.coalesce(parent_names.parent_names, literal_column("")).label(parents_column))
    if add_children:
        select_columns.append(func.coalesce(child_names.child_names, literal_column("")).label(children_column))
    if add_synonyms:
        select_columns.append(func.coalesce(synonym_names.synonym_names, literal_column("")).label(synonyms_column))

    query = select(*select_columns).select_from(concept).where(concept.c.concept_id.in_(concept_ids))

    if add_parents:
        query = query.outerjoin(parents_sq, concept.c.concept_id == parent_names.concept_id)
    if add_children:
        query = query.outerjoin(children_Sq, concept.c.concept_id == child_names.concept_id)
    if add_synonyms:
        query = query.outerjoin(synonyms_sq, concept.c.concept_id == synonym_names.concept_id)

    return query


def add_concept_context(
    concept_table: pd.DataFrame,
    concept_id_column: str = "matched_concept_id",
    domain_id_column: str = "matched_domain_id",
    concept_class_id_column: str = "matched_concept_class_id",
    vocabulary_id_column: str = "matched_vocabulary_id",
    add_parents: bool = True,
    parents_column: str = "matched_parents",
    add_children: bool = True,
    children_column: str = "matched_children",
    add_synonyms: bool = True,
    synonyms_column: str = "matched_synonyms",
) -> pd.DataFrame:
    """
    Adds concept context (domain, concept class, vocabulary, parents, children, synonyms) to the given concept table.
    Multiple entries per concept will be concatenated with semicolons. Children are limited to 10 random entries per
    concept.

    Args:
        concept_table: DataFrame containing concept IDs.
        concept_id_column: Name of the column with concept IDs.
        domain_id_column: Name of the column for the domain ID.
        concept_class_id_column: Name of the column  for the domain concept class ID.
        vocabulary_id_column: Name of the column  for the domain vocabulary ID.
        add_parents:  Whether to add parent concepts.
        parents_column: Name of the column for parent concept names.
        add_children: Whether to add child concepts.
        children_column: Name of the column for child concept names.
        add_synonyms: Whether to add concept synonyms.
        synonyms_column: Name of the column for concept synonyms.

    Returns:
        DataFrame enriched with concept context columns.
    """

    engine = create_engine(get_environment_variable("VOCAB_CONNECTION_STRING"))

    concept_ids = concept_table[concept_id_column].unique().tolist()
    query = _create_query(
        concept_ids=concept_ids,
        concept_class_id_column=concept_class_id_column,
        domain_id_column=domain_id_column,
        vocabulary_id_column=vocabulary_id_column,
        add_parents=add_parents,
        parents_column=parents_column,
        add_children=add_children,
        children_column=children_column,
        add_synonyms=add_synonyms,
        synonyms_column=synonyms_column,
        engine=engine,
    )

    with engine.connect() as connection:
        result = connection.execute(query)
        context_df = pd.DataFrame(result.fetchall(), columns=result.keys())
    merged_df = concept_table.merge(context_df, left_on=concept_id_column, right_on="concept_id", how="left")
    merged_df.drop(columns=["concept_id"], inplace=True)
    return merged_df


if __name__ == "__main__":
    df = pd.DataFrame({"matched_concept_id": [312327, 198124, 4189939]})
    enriched_df = add_concept_context(df, add_parents=True, add_children=True, add_synonyms=True)
    print(enriched_df)
