import pytest

from ariadne.utils.config import Config, load_hierarchy_settings


def test_config_uses_config_condition_mapping_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("ariadne.utils.config.get_project_root", lambda: tmp_path)

    (tmp_path / "config_condition_mapping.yaml").write_text("{}\n", encoding="utf-8")

    config = Config()

    assert config is not None


def test_config_does_not_fallback_to_config_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("ariadne.utils.config.get_project_root", lambda: tmp_path)

    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        Config()


def test_load_hierarchy_settings_uses_new_default_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("ariadne.utils.config.get_project_root", lambda: tmp_path)

    (tmp_path / "config_condition_mapping.yaml").write_text(
        "hierarchy:\n"
        "  prompts:\n"
        "    extraction: extraction prompt\n"
        "    selection: selection prompt\n",
        encoding="utf-8",
    )

    hierarchy = load_hierarchy_settings()

    assert hierarchy.prompts.extraction == "extraction prompt"
    assert hierarchy.prompts.selection == "selection prompt"

