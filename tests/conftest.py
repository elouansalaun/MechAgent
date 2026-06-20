from pathlib import Path

import pytest
import yaml


@pytest.fixture
def write_config(tmp_path: Path):
    """Write a YAML config in a temp folder and return its path."""

    def _write(data: dict) -> Path:
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(data))
        return path

    return _write


@pytest.fixture
def require_ccx():
    import shutil

    if shutil.which("ccx") is None:
        pytest.skip("CalculiX (ccx) is not on PATH")
