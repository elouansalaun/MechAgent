"""Turn surface forces into consistent nodal forces on the quadratic surface mesh."""

from __future__ import annotations

import numpy as np

from mechagent.simulation.mesh import Mesh
from mechagent.simulation.problem import SurfaceForce

# For a uniform traction on a 6-node triangle the consistent nodal loads are zero on the
# corner nodes and one third of the triangle force on each edge node.
TRI6_WEIGHTS = np.array([0, 0, 0, 1, 1, 1]) / 3


def _facet_geometry(mesh: Mesh, facets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corners = mesh.node_coords(facets[:, :3].ravel()).reshape(-1, 3, 3)
    areas = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    )
    return areas, corners.mean(axis=1)


def bearing_facets(mesh: Mesh, load: SurfaceForce) -> np.ndarray:
    """Facets of the hole the pin pushes on: radial direction pointing along the force."""
    facets = mesh.facets[load.surface]
    if load.bearing_axis is None or load.bearing_center is None:
        return facets

    axis = np.array(load.bearing_axis, dtype=float)
    axis /= np.linalg.norm(axis)
    force = np.array(load.force, dtype=float)
    radial_force = force - force.dot(axis) * axis
    if np.linalg.norm(radial_force) < 1e-9 * max(np.linalg.norm(force), 1.0):
        return facets  # purely axial force, the whole hole carries it

    _, centroids = _facet_geometry(mesh, facets)
    radial = centroids - np.array(load.bearing_center)
    radial -= np.outer(radial @ axis, axis)
    return facets[radial @ radial_force > 0]


def nodal_forces(mesh: Mesh, load: SurfaceForce) -> dict[int, np.ndarray]:
    """Nodal forces whose sum is exactly the requested total force."""
    if load.surface not in mesh.facets:
        raise KeyError(f"load on unknown face group '{load.surface}'")

    facets = bearing_facets(mesh, load)
    if len(facets) == 0:
        raise ValueError(f"no facet selected to carry the load on '{load.surface}'")

    areas, _ = _facet_geometry(mesh, facets)
    weights = np.outer(areas, TRI6_WEIGHTS)

    totals: dict[int, float] = {}
    for node, weight in zip(facets.ravel(), weights.ravel(), strict=True):
        if weight > 0:
            totals[int(node)] = totals.get(int(node), 0.0) + weight

    total_weight = sum(totals.values())
    force = np.array(load.force, dtype=float)
    return {node: force * weight / total_weight for node, weight in totals.items()}


def combine_nodal_forces(mesh: Mesh, loads: list[SurfaceForce]) -> dict[int, np.ndarray]:
    combined: dict[int, np.ndarray] = {}
    for load in loads:
        for node, force in nodal_forces(mesh, load).items():
            combined[node] = combined.get(node, np.zeros(3)) + force
    return combined
