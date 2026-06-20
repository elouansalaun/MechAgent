"""Simple reference geometries used to validate the solver chain against analytical results."""

from __future__ import annotations

from pathlib import Path

import cadquery as cq

from mechagent.cad.common import export_geometry, hole_faces, planar_faces_at, single_solid
from mechagent.core.geometry import GeometryModel, HoleFeature


def box(length: float, width: float, height: float, out_dir: Path, name: str = "box"):
    """Box [0, length] x [0, width] x [0, height] with its six faces as groups x_min ... z_max."""
    solid = single_solid(cq.Workplane("XY").box(length, width, height, centered=False))
    groups = {}
    for axis, size in enumerate((length, width, height)):
        groups[f"{'xyz'[axis]}_min"] = planar_faces_at(solid, axis, 0.0)
        groups[f"{'xyz'[axis]}_max"] = planar_faces_at(solid, axis, size)
    return export_geometry(solid, name, groups, out_dir)


def quarter_plate_with_hole(
    half_length: float,
    half_width: float,
    thickness: float,
    radius: float,
    out_dir: Path,
    name: str = "quarter_plate",
) -> GeometryModel:
    """Quarter of a plate with a central hole, cut along its two symmetry planes x=0 and y=0.

    The full plate is 2 * half_length long (x), 2 * half_width wide (y) and `thickness` thick.
    """
    plate = cq.Workplane("XY").box(half_length, half_width, thickness, centered=False)
    hole = HoleFeature(name="hole", center=(0, 0, 0), axis=(0, 0, 1), diameter=2 * radius)
    cylinder = cq.Solid.makeCylinder(radius, thickness + 2, cq.Vector(0, 0, -1), cq.Vector(0, 0, 1))
    solid = single_solid(plate.cut(cylinder))

    groups = {
        "x_min": planar_faces_at(solid, 0, 0.0),
        "x_max": planar_faces_at(solid, 0, half_length),
        "y_min": planar_faces_at(solid, 1, 0.0),
        "z_min": planar_faces_at(solid, 2, 0.0),
        "hole": hole_faces(solid, hole),
    }
    return export_geometry(solid, name, groups, out_dir, holes=[hole])
