import numpy as np
import pytest

from mechagent.cad.bracket import BracketParameters, build_bracket
from mechagent.cad.reference import box
from mechagent.simulation.loads import combine_nodal_forces, nodal_forces
from mechagent.simulation.mesh import MeshSettings, generate_mesh
from mechagent.simulation.problem import SurfaceForce


@pytest.fixture(scope="module")
def box_mesh(tmp_path_factory):
    geometry = box(100, 20, 10, tmp_path_factory.mktemp("box"))
    return generate_mesh(geometry, MeshSettings(element_size=4.0))


@pytest.fixture(scope="module")
def bracket(tmp_path_factory):
    geometry = build_bracket(BracketParameters(), tmp_path_factory.mktemp("bracket"))
    return geometry, generate_mesh(geometry, MeshSettings(element_size=5.0))


def resultant(mesh, forces):
    nodes = np.array(list(forces))
    values = np.array(list(forces.values()))
    return values.sum(axis=0), mesh.node_coords(nodes), values


def test_uniform_traction_sums_to_the_total_force(box_mesh):
    load = SurfaceForce(surface="x_max", force=(0, 0, -100))

    total, points, values = resultant(box_mesh, nodal_forces(box_mesh, load))

    assert total == pytest.approx([0, 0, -100], abs=1e-9)
    # A uniform traction acts at the centroid of the face: no moment about it.
    moment = np.cross(points - np.array([100, 10, 5]), values).sum(axis=0)
    assert np.linalg.norm(moment) < 1e-6 * 100 * 10


def test_corner_nodes_carry_no_load(box_mesh):
    forces = nodal_forces(box_mesh, SurfaceForce(surface="x_max", force=(10, 0, 0)))
    corner_nodes = set(box_mesh.facets["x_max"][:, :3].ravel())

    assert corner_nodes.isdisjoint(forces)


def test_bearing_load_only_on_the_pushed_half(bracket):
    geometry, mesh = bracket
    hole = geometry.hole("load_hole")
    load = SurfaceForce(
        surface="load_hole",
        force=(0, 0, -2000),
        bearing_axis=hole.axis,
        bearing_center=hole.center,
    )

    forces = nodal_forces(mesh, load)
    total, points, values = resultant(mesh, forces)

    assert total == pytest.approx([0, 0, -2000], abs=1e-6)
    # Pin pushed downward: it bears on the lower half of the hole. A uniform traction on half
    # a cylinder acts at 2r/pi below the axis.
    radius = hole.diameter / 2
    mean_z = np.average(points[:, 2], weights=values[:, 2])
    assert mean_z == pytest.approx(hole.center[2] - 2 * radius / np.pi, abs=0.05 * radius)
    assert np.all(points[:, 2] < hole.center[2] + 0.5 * radius)


def test_purely_axial_force_uses_the_whole_hole(bracket):
    geometry, mesh = bracket
    hole = geometry.hole("load_hole")
    load = SurfaceForce(
        surface="load_hole", force=(500, 0, 0), bearing_axis=hole.axis, bearing_center=hole.center
    )

    _, points, _ = resultant(mesh, nodal_forces(mesh, load))

    assert points[:, 2].min() < hole.center[2] < points[:, 2].max()


def test_loads_on_the_same_face_are_added(box_mesh):
    loads = [
        SurfaceForce(surface="x_max", force=(0, 0, -100)),
        SurfaceForce(surface="x_max", force=(0, 50, 0)),
    ]

    total, _, _ = resultant(box_mesh, combine_nodal_forces(box_mesh, loads))

    assert total == pytest.approx([0, 50, -100], abs=1e-9)


def test_unknown_surface_is_an_error(box_mesh):
    with pytest.raises(KeyError):
        nodal_forces(box_mesh, SurfaceForce(surface="top", force=(0, 0, 1)))
