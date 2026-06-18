"""Helpers shared by the CadQuery geometry builders."""

from __future__ import annotations

from pathlib import Path

import cadquery as cq
import numpy as np

from mechagent.core.geometry import FaceFingerprint, GeometryModel, HoleFeature, Vec3


class GeometryError(ValueError):
    """Raised when CadQuery cannot build a valid solid from the parameters."""


def single_solid(workplane: cq.Workplane) -> cq.Solid:
    solids = workplane.solids().vals()
    if len(solids) != 1:
        raise GeometryError(f"expected one solid, CadQuery produced {len(solids)}")
    solid = solids[0]
    if not solid.isValid():
        raise GeometryError("CadQuery produced an invalid solid")
    return solid


def fingerprint(face: cq.Face) -> FaceFingerprint:
    center = face.Center()
    return FaceFingerprint(center=(center.x, center.y, center.z), area=face.Area())


def _distance_to_axis(point: cq.Vector, center: Vec3, axis: Vec3) -> float:
    offset = np.array([point.x, point.y, point.z]) - np.array(center)
    return float(np.linalg.norm(np.cross(offset, np.array(axis))))


def hole_faces(solid: cq.Solid, hole: HoleFeature, tol: float = 1e-4) -> list[cq.Face]:
    """Cylindrical faces of `solid` lying on the surface of `hole`."""
    radius = hole.diameter / 2
    faces = []
    for face in solid.Faces():
        if face.geomType() != "CYLINDER":
            continue
        vertices = face.Vertices()
        if vertices and all(
            abs(_distance_to_axis(v.Center(), hole.center, hole.axis) - radius) < tol
            for v in vertices
        ):
            faces.append(face)
    if not faces:
        raise GeometryError(f"no cylindrical face found for hole '{hole.name}'")
    return faces


def planar_faces_at(solid: cq.Solid, axis: int, value: float, tol: float = 1e-6) -> list[cq.Face]:
    """Planar faces lying entirely in the plane x[axis] = value."""
    faces = []
    for face in solid.Faces():
        if face.geomType() != "PLANE":
            continue
        box = face.BoundingBox()
        low = (box.xmin, box.ymin, box.zmin)[axis]
        high = (box.xmax, box.ymax, box.zmax)[axis]
        if abs(low - value) < tol and abs(high - value) < tol:
            faces.append(face)
    if not faces:
        raise GeometryError(f"no planar face found at {'xyz'[axis]} = {value}")
    return faces


def export_geometry(
    solid: cq.Solid,
    name: str,
    face_groups: dict[str, list[cq.Face]],
    out_dir: Path,
    holes: list[HoleFeature] | None = None,
) -> GeometryModel:
    out_dir.mkdir(parents=True, exist_ok=True)
    brep_file = out_dir / f"{name}.brep"
    solid.exportBrep(str(brep_file))

    box = solid.BoundingBox()
    fingerprints = {group: [fingerprint(f) for f in faces] for group, faces in face_groups.items()}
    return GeometryModel(
        name=name,
        brep_file=brep_file,
        volume=solid.Volume(),
        bounding_box=((box.xmin, box.ymin, box.zmin), (box.xmax, box.ymax, box.zmax)),
        face_groups=fingerprints,
        holes=holes or [],
    )
