"""Tetrahedral meshing of a BREP geometry with Gmsh.

We only produce quadratic tetrahedra (CalculiX C3D10). Linear tetrahedra are far too stiff in
bending to be trusted on a bracket.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gmsh
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from mechagent.core.geometry import FaceFingerprint, GeometryModel
from mechagent.core.logging import get_logger

log = get_logger(__name__)

GMSH_TET10 = 11
GMSH_TRI6 = 9
# Gmsh numbers the last two edge nodes of a tet10 the other way around compared to CalculiX.
GMSH_TO_CALCULIX_TET10 = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]


class MeshError(RuntimeError):
    pass


class MeshSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    element_size: float = Field(5.0, gt=0, description="maximum element size, mm")
    curvature_points: int = Field(16, ge=0, description="elements per 2*pi on curved faces")
    refinements: dict[str, float] = Field(
        default_factory=dict, description="element size on named face groups, mm"
    )
    refinement_distance: float = Field(
        3.0, gt=0, description="distance over which refinements fade out, in element sizes"
    )
    optimize: bool = True

    def scaled(self, factor: float) -> MeshSettings:
        """Same settings with every element size multiplied by `factor`."""
        return self.model_copy(
            update={
                "element_size": self.element_size * factor,
                "refinements": {k: v * factor for k, v in self.refinements.items()},
            }
        )


@dataclass
class Mesh:
    node_ids: np.ndarray  # (N,)
    coords: np.ndarray  # (N, 3), mm
    element_ids: np.ndarray  # (M,)
    elements: np.ndarray  # (M, 10) node ids in CalculiX C3D10 order
    node_sets: dict[str, np.ndarray] = field(default_factory=dict)  # nodes on each face group
    facets: dict[str, np.ndarray] = field(default_factory=dict)  # (K, 6) tri6 per face group
    min_quality: float = 0.0  # Gmsh minSICN, negative means inverted elements
    mean_quality: float = 0.0

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_elements(self) -> int:
        return len(self.element_ids)

    def node_coords(self, ids: np.ndarray) -> np.ndarray:
        index = np.searchsorted(self.node_ids, ids)
        return self.coords[index]


def _match_surface(fp: FaceFingerprint, surfaces: list[int], tol: float) -> int:
    matches = []
    for tag in surfaces:
        center = np.array(gmsh.model.occ.getCenterOfMass(2, tag))
        area = gmsh.model.occ.getMass(2, tag)
        if np.linalg.norm(center - fp.center) < tol and abs(area - fp.area) < 1e-4 * fp.area:
            matches.append(tag)
    if len(matches) != 1:
        raise MeshError(f"face {fp} matched {len(matches)} surfaces after BREP import")
    return matches[0]


def _add_refinements(groups: dict[str, list[int]], settings: MeshSettings) -> None:
    fields = []
    for name, size in settings.refinements.items():
        if name not in groups:
            raise MeshError(f"refinement on unknown face group '{name}'")
        distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance, "SurfacesList", groups[name])
        gmsh.model.mesh.field.setNumber(distance, "Sampling", 30)
        threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMin", size)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMax", settings.element_size)
        gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(
            threshold, "DistMax", settings.refinement_distance * settings.element_size
        )
        fields.append(threshold)

    if fields:
        minimum = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(minimum)


def _surface_mesh(tags: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Nodes (including edges) and 6-node triangles of a list of surfaces."""
    nodes, facets = [], []
    for tag in tags:
        node_tags, _, _ = gmsh.model.mesh.getNodes(2, tag, includeBoundary=True)
        nodes.append(node_tags)
        types, _, connectivity = gmsh.model.mesh.getElements(2, tag)
        for etype, conn in zip(types, connectivity, strict=True):
            if etype != GMSH_TRI6:
                raise MeshError(f"unexpected surface element type {etype}")
            facets.append(conn.reshape(-1, 6))
    return np.unique(np.concatenate(nodes)).astype(np.int64), np.vstack(facets).astype(np.int64)


def generate_mesh(geometry: GeometryModel, settings: MeshSettings) -> Mesh:
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(geometry.name)
        gmsh.model.occ.importShapes(str(geometry.brep_file))
        gmsh.model.occ.synchronize()

        volumes = gmsh.model.getEntities(3)
        if len(volumes) != 1:
            raise MeshError(f"expected one volume in {geometry.brep_file}, found {len(volumes)}")

        surfaces = [tag for _, tag in gmsh.model.getEntities(2)]
        tol = 1e-5 * geometry.size
        groups = {
            name: [_match_surface(fp, surfaces, tol) for fp in fingerprints]
            for name, fingerprints in geometry.face_groups.items()
        }

        gmsh.option.setNumber("Mesh.MeshSizeMax", settings.element_size)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", settings.curvature_points)
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.option.setNumber("Mesh.Optimize", int(settings.optimize))
        gmsh.option.setNumber("Mesh.OptimizeNetgen", int(settings.optimize))
        _add_refinements(groups, settings)

        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.renumberNodes()
        gmsh.model.mesh.renumberElements()

        node_ids, coords, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(node_ids)
        node_ids = node_ids[order].astype(np.int64)
        coords = coords.reshape(-1, 3)[order]

        types, element_ids, connectivity = gmsh.model.mesh.getElements(3)
        if list(types) != [GMSH_TET10]:
            raise MeshError(f"expected only tet10 elements, got element types {list(types)}")
        elements = connectivity[0].reshape(-1, 10)[:, GMSH_TO_CALCULIX_TET10].astype(np.int64)
        quality = gmsh.model.mesh.getElementQualities(element_ids[0], "minSICN")

        mesh = Mesh(
            node_ids=node_ids,
            coords=coords,
            element_ids=element_ids[0].astype(np.int64),
            elements=elements,
            min_quality=float(quality.min()),
            mean_quality=float(quality.mean()),
        )
        for name, tags in groups.items():
            mesh.node_sets[name], mesh.facets[name] = _surface_mesh(tags)
    except MeshError:
        raise
    except Exception as exc:
        raise MeshError(f"Gmsh failed on {geometry.brep_file}: {exc}") from exc
    finally:
        gmsh.finalize()

    log.info(
        "meshed %s: %d nodes, %d C3D10 elements, min quality %.3f",
        geometry.name,
        mesh.n_nodes,
        mesh.n_elements,
        mesh.min_quality,
    )
    return mesh
