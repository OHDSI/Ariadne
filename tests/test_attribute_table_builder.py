import numpy as np
import pandas as pd

import ariadne.hierarchy.attribute_table_builder as builder
from ariadne.utils.settings import HierarchySettings, IndexBuildConfig


def test_prepare_attribute_embedding_batches_creates_files_and_reuses_concept_embeddings(tmp_path, monkeypatch):
    calls = []

    def _fake_get_embedding_vectors(texts):
        calls.append(list(texts))
        embeddings = np.array([[idx + 1.0, (idx + 1.0) * 10.0] for idx in range(len(texts))], dtype=np.float32)
        return {
            "embeddings": embeddings,
            "usage": {"total_cost_usd": 0.01},
        }

    monkeypatch.setattr(builder, "get_embedding_vectors", _fake_get_embedding_vectors)

    df = pd.DataFrame(
        [
            {
                "concept_id": 2,
                "concept_code": "C2",
                "concept_name": "Beta",
                "attribute_category": "Has finding site",
            },
            {
                "concept_id": 1,
                "concept_code": "C1",
                "concept_name": "Alpha",
                "attribute_category": "Has asso morph",
            },
            {
                "concept_id": 1,
                "concept_code": "C1",
                "concept_name": "Alpha",
                "attribute_category": "Has severity",
            },
        ]
    )

    batch_files, dim = builder._prepare_attribute_embedding_batches(
        df,
        embedding_batch_size=1,
        embedding_cache_dir=tmp_path,
    )

    assert len(batch_files) == 2
    assert dim == 2
    assert calls == [["Alpha"], ["Beta"]]

    first = pd.read_parquet(batch_files[0])
    second = pd.read_parquet(batch_files[1])
    assert sorted(first["concept_id"].tolist()) == [1, 1]
    assert second["concept_id"].tolist() == [2]


def test_prepare_attribute_embedding_batches_infers_dim_from_cached_files(tmp_path, monkeypatch):
    monkeypatch.setattr(
        builder,
        "get_embedding_vectors",
        lambda _texts: (_ for _ in ()).throw(AssertionError("should not request embeddings for cached batch")),
    )

    batch_file = tmp_path / "snomed_attribute_batch_000000.parquet"
    pd.DataFrame(
        [
            {
                "concept_id": 10,
                "concept_code": "C10",
                "concept_name": "Gamma",
                "attribute_category": "Has interpretation",
                "embedding": [0.1, 0.2, 0.3],
            }
        ]
    ).to_parquet(batch_file, index=False)

    df = pd.DataFrame(
        [
            {
                "concept_id": 10,
                "concept_code": "C10",
                "concept_name": "Gamma",
                "attribute_category": "Has interpretation",
            }
        ]
    )

    batch_files, dim = builder._prepare_attribute_embedding_batches(
        df,
        embedding_batch_size=1,
        embedding_cache_dir=tmp_path,
    )

    assert batch_files == [batch_file]
    assert dim == 3


class _FakeConnection:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_build_attribute_table_skip_mode_returns_early_for_populated_table(tmp_path, monkeypatch):
    fake_conn = _FakeConnection()
    settings = HierarchySettings(
        index_build=IndexBuildConfig(
            embedding_cache_folder=str(tmp_path / "cache"),
            embedding_batch_size=2,
        ),
    )

    monkeypatch.setattr(builder, "load_dotenv", lambda: None)
    monkeypatch.setattr(builder, "_pg_connect", lambda: fake_conn)
    monkeypatch.setattr(builder, "check_populated", lambda _conn, _table: True)
    monkeypatch.setattr(
        builder,
        "load_attributes_from_db",
        lambda: (_ for _ in ()).throw(AssertionError("should skip before loading attributes")),
    )

    builder.build_attribute_table(settings, if_exists="skip")

    assert fake_conn.closed is True


def test_build_attribute_table_uses_probe_embedding_dim_when_batch_dim_missing(tmp_path, monkeypatch):
    fake_conn = _FakeConnection()
    settings = HierarchySettings(
        index_build=IndexBuildConfig(
            embedding_cache_folder=str(tmp_path / "cache"),
            embedding_batch_size=2,
        ),
    )

    calls = {"create_table_dim": None, "insert_called": False, "index_called": False}

    monkeypatch.setattr(builder, "load_dotenv", lambda: None)
    monkeypatch.setattr(builder, "load_attributes_from_db", lambda: pd.DataFrame())
    monkeypatch.setattr(builder, "_prepare_attribute_embedding_batches", lambda *_args, **_kwargs: ([], None))
    monkeypatch.setattr(
        builder,
        "get_embedding_vectors",
        lambda _texts: {"embeddings": np.zeros((1, 7), dtype=np.float32), "usage": {"total_cost_usd": 0.0}},
    )
    monkeypatch.setattr(builder, "_pg_connect", lambda: fake_conn)

    def _fake_create_table(_conn, dim):
        calls["create_table_dim"] = dim

    def _fake_insert_attributes(conn, batch_files, upload_batch_size, rebuild=False):
        assert conn is fake_conn
        assert batch_files == []
        assert upload_batch_size == 2
        assert rebuild is False
        calls["insert_called"] = True
        return conn

    def _fake_create_embedding_index(_conn):
        calls["index_called"] = True

    monkeypatch.setattr(builder, "create_table", _fake_create_table)
    monkeypatch.setattr(builder, "insert_attributes", _fake_insert_attributes)
    monkeypatch.setattr(builder, "create_embedding_index", _fake_create_embedding_index)

    builder.build_attribute_table(settings, if_exists="append")

    assert calls["create_table_dim"] == 7
    assert calls["insert_called"] is True
    assert calls["index_called"] is True
    assert fake_conn.closed is True


