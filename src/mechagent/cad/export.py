"""STEP export of a bracket, placed in the user's global frame."""

from __future__ import annotations

from pathlib import Path

import cadquery as cq
import numpy as np
from OCP.gp import gp_Trsf

from mechagent.cad.bracket import BracketParameters, build_bracket_solid


def placed_bracket(params: BracketParameters, rotation: np.ndarray, origin: np.ndarray) -> cq.Shape:
    """The bracket moved from its own frame: global = origin + rotation @ local."""
    rotation = np.asarray(rotation, dtype=float)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9) or np.linalg.det(rotation) < 0:
        raise ValueError("the placement must be a proper rotation")
    transform = gp_Trsf()
    transform.SetValues(*(float(v) for i in range(3) for v in (*rotation[i], origin[i])))
    return build_bracket_solid(params).moved(cq.Location(transform))


def export_step(
    params: BracketParameters, rotation: np.ndarray, origin: np.ndarray, path: Path
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(placed_bracket(params, rotation, origin), str(path))
    return path
