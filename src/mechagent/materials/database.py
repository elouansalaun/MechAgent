"""Loading the material database from data/materials.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

from mechagent.core.config import PROJECT_ROOT
from mechagent.materials.material import Material

DEFAULT_DATABASE = PROJECT_ROOT / "data" / "materials.yaml"


def load_materials(path: Path = DEFAULT_DATABASE) -> dict[str, Material]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    return {name: Material(name=name, **properties) for name, properties in data.items()}


def get_material(name: str, path: Path = DEFAULT_DATABASE) -> Material:
    materials = load_materials(path)
    if name not in materials:
        raise KeyError(f"unknown material '{name}', available: {sorted(materials)}")
    return materials[name]
