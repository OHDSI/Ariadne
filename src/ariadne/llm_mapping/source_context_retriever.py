from pathlib import Path
from typing import List, Union

import pandas as pd


# Composite key identifying a source concept across vocabularies.
KEY_COLUMNS: List[str] = ["source_vocabulary_id", "source_concept_code"]

# Normalized source-context schema. Multi-values are semicolon-joined, matching
# the target-side convention in concept_context_retriever.add_concept_context.
SOURCE_CONTEXT_COLUMNS: List[str] = [
    "source_parents",
    "source_children",
    "source_description",
    "source_synonyms",
]


class SourceContextRetriever:
    """
    Adds normalized source-concept context (parents, children, description, synonyms)
    onto a candidate table, keyed by source vocabulary + concept code.

    This is the source-side mirror of
    :func:`ariadne.llm_mapping.concept_context_retriever.add_concept_context`.
    Instead of querying the OMOP database, it reads context from a CSV (or an
    already-loaded DataFrame) that carries the normalized schema, so the same
    file can supply both source concepts and their context.
    """

    def __init__(
        self,
        context_source: Union[str, Path, pd.DataFrame],
        context_columns: List[str] = SOURCE_CONTEXT_COLUMNS,
    ) -> None:
        """
        Args:
            context_source: Path to a CSV, or a DataFrame, carrying the key columns
                (source_vocabulary_id, source_concept_code) and any context columns.
            context_columns: Normalized context columns to expose. Columns absent from
                the source are treated as empty.
        """
        self.context_columns = list(context_columns)

        if isinstance(context_source, pd.DataFrame):
            context_df = context_source.copy()
        else:
            context_df = pd.read_csv(context_source)

        missing_keys = [c for c in KEY_COLUMNS if c not in context_df.columns]
        if missing_keys:
            raise ValueError(f"context_source is missing required key columns: {missing_keys}")

        available_context = [c for c in self.context_columns if c in context_df.columns]
        context_df = context_df[KEY_COLUMNS + available_context].copy()
        for column in KEY_COLUMNS:
            context_df[column] = context_df[column].astype(str)
        # One context row per source concept.
        self._context = context_df.drop_duplicates(subset=KEY_COLUMNS)

    def add_source_context(
        self,
        candidate_table: pd.DataFrame,
        vocabulary_id_column: str = "source_vocabulary_id",
        code_column: str = "source_concept_code",
    ) -> pd.DataFrame:
        """
        Merges the normalized source context onto *candidate_table* by
        (vocabulary_id, concept_code).

        Args:
            candidate_table: DataFrame containing the source key columns.
            vocabulary_id_column: Column in *candidate_table* holding the source vocabulary id.
            code_column: Column in *candidate_table* holding the source concept code.

        Returns:
            *candidate_table* enriched with the normalized source_* context columns.
            Unmatched rows and columns absent from the context source are filled with "".
        """
        missing_keys = [c for c in (vocabulary_id_column, code_column) if c not in candidate_table.columns]
        if missing_keys:
            raise ValueError(f"candidate_table is missing required key columns: {missing_keys}")

        context = self._context.rename(
            columns={
                "source_vocabulary_id": vocabulary_id_column,
                "source_concept_code": code_column,
            }
        )

        merge_keys = [vocabulary_id_column, code_column]
        merged = candidate_table.copy()
        merged[vocabulary_id_column] = merged[vocabulary_id_column].astype(str)
        merged[code_column] = merged[code_column].astype(str)
        # Drop any pre-existing context columns so retrieved context is authoritative and no _x/_y suffixes appear.
        overlap = [c for c in self.context_columns if c in merged.columns]
        if overlap:
            merged = merged.drop(columns=overlap)
        merged = merged.merge(context, on=merge_keys, how="left")

        for column in self.context_columns:
            if column not in merged.columns:
                merged[column] = ""
            else:
                merged[column] = merged[column].fillna("")

        return merged


if __name__ == "__main__":
    context = pd.DataFrame(
        {
            "source_vocabulary_id": ["ICD10CM"],
            "source_concept_code": ["N76.8"],
            "source_parents": ["Inflammatory disease of female pelvic organs"],
            "source_children": [""],
            "source_description": ["Other specified inflammation of vagina and vulva"],
            "source_synonyms": ["Vulvovaginal inflammation NOS"],
        }
    )
    candidates = pd.DataFrame(
        {
            "source_vocabulary_id": ["ICD10CM"],
            "source_concept_code": ["N76.8"],
            "source_term": ["Other specified inflammation of vagina and vulva"],
        }
    )
    retriever = SourceContextRetriever(context)
    enriched = retriever.add_source_context(candidates)
    print(enriched)
