from pathlib import Path

import pytest
from pydantic import ValidationError

from mechagent.core.config import (
    CONFIG_ENV_VAR,
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    Settings,
    load_settings,
)


def test_default_config_file_is_valid():
    settings = load_settings(DEFAULT_CONFIG_PATH)

    assert settings.units.length == "mm"
    assert settings.units.stress == "MPa"
    assert settings.solvers.calculix_executable == "ccx"
    assert settings.llm.provider == "lmstudio"


def test_partial_config_falls_back_on_defaults(write_config):
    path = write_config({"llm": {"model": "google/gemma-4-e4b"}})

    settings = load_settings(path)

    assert settings.llm.model == "google/gemma-4-e4b"
    assert settings.solvers.timeout_s == 600


def test_env_variable_selects_config(write_config, monkeypatch):
    path = write_config({"logging": {"level": "DEBUG"}})
    monkeypatch.setenv(CONFIG_ENV_VAR, str(path))

    assert load_settings().logging.level == "DEBUG"


def test_unknown_keys_are_rejected(write_config):
    path = write_config({"solvers": {"calculx_executable": "ccx"}})  # typo on purpose

    with pytest.raises(ValidationError):
        load_settings(path)


def test_other_unit_systems_are_rejected(write_config):
    path = write_config({"units": {"length": "m"}})

    with pytest.raises(ValidationError):
        load_settings(path)


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "nope.yaml")


def test_resolve_relative_and_absolute_paths(tmp_path):
    settings = Settings()

    assert settings.resolve(Path("runs")) == PROJECT_ROOT / "runs"
    assert settings.resolve(tmp_path) == tmp_path
