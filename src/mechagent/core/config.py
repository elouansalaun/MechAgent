"""Application configuration loaded from a YAML file and validated with Pydantic."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
CONFIG_ENV_VAR = "MECHAGENT_CONFIG"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectConfig(_Section):
    name: str = "MechAgent"
    work_dir: Path = Path("runs")
    data_dir: Path = Path("data")


class UnitsConfig(_Section):
    # Only one unit system is supported for now, we keep it explicit so it shows up in reports.
    length: Literal["mm"] = "mm"
    force: Literal["N"] = "N"
    stress: Literal["MPa"] = "MPa"
    mass: Literal["tonne"] = "tonne"
    time: Literal["s"] = "s"


class SolversConfig(_Section):
    calculix_executable: str = "ccx"
    gmsh_executable: str = "gmsh"
    timeout_s: float = Field(600, gt=0)


class LLMConfig(_Section):
    provider: Literal["lmstudio"] = "lmstudio"
    base_url: str = "http://localhost:1234/v1"
    model: str = "qwen/qwen3.5-9b"
    temperature: float = Field(0.0, ge=0.0, le=2.0)


class LoggingConfig(_Section):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file: Path | None = Path("logs/mechagent.log")


class Settings(_Section):
    project: ProjectConfig = ProjectConfig()
    units: UnitsConfig = UnitsConfig()
    solvers: SolversConfig = SolversConfig()
    llm: LLMConfig = LLMConfig()
    logging: LoggingConfig = LoggingConfig()

    def resolve(self, path: Path) -> Path:
        """Relative paths in the config are relative to the project root."""
        return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings(path: str | Path | None = None) -> Settings:
    """Load settings from `path`, then $MECHAGENT_CONFIG, then the default file."""
    if path is None:
        path = os.environ.get(CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    data = yaml.safe_load(path.read_text()) or {}
    return Settings.model_validate(data)
