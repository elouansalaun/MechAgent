"""Geometry interface shared by the CAD and simulation packages.

The CAD side exports a solid to a BREP file and describes the faces we care about (holes,
supports, loaded faces) by a fingerprint: centroid and area. The mesher finds the same faces
again after importing the BREP by comparing fingerprints, so both sides never need to share
OpenCascade objects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

Vec3 = tuple[float, float, float]


class FaceFingerprint(BaseModel):
    model_config = ConfigDict(frozen=True)

    center: Vec3  # mm
    area: float  # mm^2


class HoleFeature(BaseModel):
    """A cylindrical hole: a mounting point or a load introduction point."""

    name: str
    center: Vec3  # a point on the hole axis, mm
    axis: Vec3  # unit vector
    diameter: float  # mm


class GeometryModel(BaseModel):
    name: str
    brep_file: Path
    volume: float  # mm^3
    bounding_box: tuple[Vec3, Vec3]  # (min corner, max corner), mm
    face_groups: dict[str, list[FaceFingerprint]]
    holes: list[HoleFeature] = []

    @property
    def size(self) -> float:
        """Diagonal of the bounding box, used as a characteristic length."""
        low, high = (np.array(corner) for corner in self.bounding_box)
        return float(np.linalg.norm(high - low))

    def hole(self, name: str) -> HoleFeature:
        for hole in self.holes:
            if hole.name == name:
                return hole
        raise KeyError(f"No hole named '{name}' in geometry '{self.name}'")
