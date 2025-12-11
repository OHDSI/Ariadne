# Copyright 2025 Observational Health Data Sciences and Informatics
#
# This file is part of Ariadne
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import os
from typing import List

from dotenv import load_dotenv
from sqlalchemy import (
    create_engine,
    select,
    Select,
    cast,
    String,
    union_all,
    MetaData,
    Table,
    Column,
    Integer,
    or_,
    func,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import aliased
import pyarrow as pa
import pyarrow.parquet as pq

from ariadne.utils.logger import open_log
from ariadne.utils.config import Config
from ariadne.utils.utils import get_environment_variable

load_dotenv()


def _create_query(engine: Engine, config: Config) -> Select:
    vocabulary_schema = get_environment_variable("VOCAB_SCHEMA")
    filter_config = config.verbatim_mapping.standard_concept_filter

    metadata = MetaData()
    concept = Table("concept", metadata, schema=vocabulary_schema, autoload_with=engine)

    standard_concepts = ["S"]
    if filter_config.include_classification_concepts:
        standard_concepts.append("C")

    # Get concept names
    query1 = select(
        concept.c.concept_id,
        concept.c.concept_name.label("term"),
        concept.c.concept_name,
        # concept.c.vocabulary_id,
        # concept.c.domain_id,
        # concept.c.standard_concept,
        # cast("name", String).label("source"),
    ).where(concept.c.standard_concept.in_(standard_concepts))
    if filter_config.domain_ids:
        query1 = query1.where(concept.c.domain_id.in_(filter_config.domain_ids))
    if filter_config.vocabularies:
        query1 = query1.where(concept.c.vocabulary_id.in_(filter_config.vocabularies))

    if filter_config.include_synonyms:
        # Get concept synonyms.
        concept_synonym = Table(
            "concept_synonym",
            metadata,
            schema=vocabulary_schema,
            autoload_with=engine,
        )

        concept_names = query1.subquery()

        cs_alias = aliased(concept_synonym)
        query2 = select(
            cs_alias.c.concept_id,
            cs_alias.c.concept_synonym_name.label("term"),
            concept_names.c.concept_name,
            # concept_names.c.vocabulary_id,
            # concept_names.c.domain_id,
            # concept_names.c.standard_concept,
            # cast("synonym", String).label("source"),
        ).join(concept_names, cs_alias.c.concept_id == concept_names.c.concept_id)

        # Combine queries
        final_query = union_all(query1, query2)
    else:
        final_query = query1

    return final_query


def _store_in_parquet(
    concept_ids: List[int],
    terms: List[str],
    concept_names: List[str],
    # vocabulary_ids: List[str],
    # domain_ids: List[str],
    # standard_concepts: List[str],
    # sources: List[str],
    file_name: str,
) -> None:
    concept_id_array = pa.array(concept_ids)
    term_array = pa.array(terms)
    concept_name_array = pa.array(concept_names)
    # vocabulary_id_array = pa.array(vocabulary_ids)
    # domain_id_array = pa.array(domain_ids)
    # standard_concept_array = pa.array(standard_concepts)
    # source_array = pa.array(sources)
    table = pa.Table.from_arrays(
        arrays=[
            concept_id_array,
            term_array,
            concept_name_array,
            # vocabulary_id_array,
            # domain_id_array,
            # standard_concept_array,
            # source_array,
        ],
        names=[
            "concept_id",
            "term",
            "concept_name",
            # "vocabulary_id",
            # "domain_id",
            # "standard_concept",
            # "source",
        ],
    )
    pq.write_table(table, file_name)


def download_terms(config: Config = Config()) -> None:
    """
    Download terms from vocabulary database and store them in parquet files for use in verbatim mapping.

    Args:
        config: A Config object containing configuration parameters. This function uses the verbatim_mapping section of
            the config, which specifies the vocabularies, domains, etc. to filter the terms to be downloaded.

    Returns:
        None
    """
    # Check if Parquet files already exist. Skip download if they do.
    if os.path.exists(config.system.terms_folder) and os.listdir(config.system.terms_folder):
        print(f"Parquet files already exist in folder {config.system.terms_folder}. Skipping download.")
        return

    os.makedirs(config.system.log_folder, exist_ok=True)
    os.makedirs(config.system.terms_folder, exist_ok=True)
    open_log(os.path.join(config.system.log_folder, "logDownloadTerms.txt"))

    logging.info("Starting downloading terms")

    engine = create_engine(get_environment_variable("VOCAB_CONNECTION_STRING"))
    query = _create_query(engine=engine, config=config)

    with engine.connect() as connection:
        terms_result_set = connection.execution_options(stream_results=True).execute(query)
        total_inserted = 0
        while True:
            chunk = terms_result_set.fetchmany(config.system.download_batch_size)
            if not chunk:
                break
            _store_in_parquet(
                concept_ids=[row.concept_id for row in chunk],
                terms=[row.term for row in chunk],
                concept_names=[row.concept_name for row in chunk],
                # vocabulary_ids=[row.vocabulary_id for row in chunk],
                # domain_ids=[row.domain_id for row in chunk],
                # standard_concepts=[row.standard_concept for row in chunk],
                # sources=[row.source for row in chunk],
                file_name=os.path.join(
                    config.system.terms_folder,
                    f"Terms_{total_inserted + 1}_{total_inserted + len(chunk)}.parquet",
                ),
            )
            total_inserted += len(chunk)
            logging.info(f"Downloaded {len(chunk)} rows, total downloaded: {total_inserted}")
    logging.info("Finished downloading terms")


if __name__ == "__main__":
    download_terms()
