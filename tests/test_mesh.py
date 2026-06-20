import numpy as np
import pytest

from mechagent.cad.bracket import BracketParameters, build_bracket
from mechagent.simulation.mesh import MeshError, MeshSettings, generate_mesh

# Corner pairs of each edge node in the CalculiX C3D10 numbering (0-based).
C3D10_EDGES = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]


@pytest.fixture(scope="module")
def bracket_geometry(tmp_path_factory):
    return build_bracket(BracketParameters(), tmp_path_factory.mktemp("bracket"))


@pytest.fixture(scope="module")
def bracket_mesh(bracket_geometry):
    return generate_mesh(bracket_geometry, MeshSettings(element_size=6.0))


def element_points(mesh):
    index = np.searchsorted(mesh.node_ids, mesh.elements)
    return mesh.coords[index]  # (M, 10, 3)


def test_edge_nodes_follow_calculix_numbering(bracket_mesh):
    points = element_points(bracket_mesh)

    for k, (i, j) in enumerate(C3D10_EDGES, start=4):
        midpoint = (points[:, i] + points[:, j]) / 2
        edge = np.linalg.norm(points[:, i] - points[:, j], axis=1)
        offset = np.linalg.norm(points[:, k] - midpoint, axis=1)
        # Edge nodes on curved faces sit slightly off the chord, never by a large fraction.
        assert np.all(offset < 0.1 * edge), f"edge node {k + 1} is not between nodes {i}, {j}"


def test_elements_have_positive_volume(bracket_mesh, bracket_geometry):
    p = element_points(bracket_mesh)
    normals = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    volumes = np.einsum("ij,ij->i", normals, p[:, 3] - p[:, 0]) / 6

    assert np.all(volumes > 0)
    # Straight-sided tets cut a little off the curved faces, the total stays very close.
    assert volumes.sum() == pytest.approx(bracket_geometry.volume, rel=5e-3)
    assert bracket_mesh.min_quality > 0


def test_hole_groups_lie_on_the_holes(bracket_mesh, bracket_geometry):
    hole = bracket_geometry.hole("load_hole")
    points = bracket_mesh.node_coords(bracket_mesh.node_sets["load_hole"])
    offset = points - np.array(hole.center)
    radius = np.linalg.norm(np.cross(offset, np.array(hole.axis)), axis=1)

    assert np.allclose(radius, hole.diameter / 2, atol=1e-6)


def test_facets_cover_the_face_groups(bracket_mesh, bracket_geometry):
    for name, fingerprints in bracket_geometry.face_groups.items():
        p = bracket_mesh.node_coords(bracket_mesh.facets[name][:, :3].ravel()).reshape(-1, 3, 3)
        area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1).sum()

        assert area == pytest.approx(sum(f.area for f in fingerprints), rel=1e-2)
        assert set(bracket_mesh.facets[name].ravel()) == set(bracket_mesh.node_sets[name])


def test_refinement_adds_nodes_near_holes(bracket_geometry, bracket_mesh):
    settings = MeshSettings(element_size=6.0, refinements={"load_hole": 1.5})

    refined = generate_mesh(bracket_geometry, settings)

    assert len(refined.node_sets["load_hole"]) > 2 * len(bracket_mesh.node_sets["load_hole"])


def test_scaled_settings():
    settings = MeshSettings(element_size=4.0, refinements={"load_hole": 2.0}).scaled(0.5)

    assert settings.element_size == 2.0
    assert settings.refinements == {"load_hole": 1.0}


def test_unknown_refinement_group_is_an_error(bracket_geometry):
    with pytest.raises(MeshError, match="unknown face group"):
        generate_mesh(bracket_geometry, MeshSettings(refinements={"nope": 1.0}))
