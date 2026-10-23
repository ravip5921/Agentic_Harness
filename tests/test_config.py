"""Environment loading and application startup behavior."""

import os

import pytest

from harness import config, main
from harness.cli import harness_cli


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.setattr(os, "environ", {})


def test_environment_values_are_available_to_python(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GROQ_API_KEY=test-key\nGROQ_MODEL=test-model\nCUSTOM_SETTING=enabled\n",
        encoding="utf-8",
    )

    assert config.load_environment(env_file)
    assert os.environ["GROQ_API_KEY"] == "test-key"
    assert os.getenv("GROQ_MODEL") == "test-model"
    assert os.getenv("CUSTOM_SETTING") == "enabled"


def test_existing_environment_takes_precedence(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("GROQ_API_KEY=file-key\n", encoding="utf-8")
    monkeypatch.setenv("GROQ_API_KEY", "exported-key")

    config.load_environment(env_file)
    config.load_environment(env_file)

    assert os.environ["GROQ_API_KEY"] == "exported-key"


def test_default_environment_file_is_independent_of_working_directory(
    monkeypatch, tmp_path,
):
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / ".env").write_text("CUSTOM_SETTING=project\n", encoding="utf-8")
    other_directory = tmp_path / "other"
    other_directory.mkdir()
    (other_directory / ".env").write_text("CUSTOM_SETTING=other\n", encoding="utf-8")
    monkeypatch.setattr(config, "PROJECT_ROOT", project_root)
    monkeypatch.chdir(other_directory)

    assert config.load_environment()
    assert os.getenv("CUSTOM_SETTING") == "project"


def test_missing_environment_file_allows_exported_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "exported-key")

    assert not config.load_environment(tmp_path / "missing.env")
    assert os.environ["GROQ_API_KEY"] == "exported-key"


def test_main_loads_environment_before_running_cli(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("GROQ_API_KEY=startup-key\n", encoding="utf-8")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)

    def cli():
        assert os.environ["GROQ_API_KEY"] == "startup-key"
        return 2

    monkeypatch.setattr(main, "harness_cli", cli)

    assert main.main() == 2


def test_direct_cli_call_loads_environment(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("GROQ_API_KEY=cli-key\n", encoding="utf-8")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)

    with pytest.raises(SystemExit) as exc:
        harness_cli(["--help"])

    assert exc.value.code == 0
    assert os.environ["GROQ_API_KEY"] == "cli-key"
