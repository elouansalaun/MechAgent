import numpy as np
import pytest

from mechagent.reporting.figures import field_grid, render_design
from mechagent.simulation.static import FieldResults


def single_tet() -> FieldResults:
    corners = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=float)
    edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]  # C3D10 edge node order
    coords = np.vstack([corners] + [(corners[i] + corners[j]) / 2 for i, j in edges])
    ids = np.arange(1, 11)
    displacement = np.zeros((10, 3))
    displacement[:, 2] = coords[:, 0] / 100
    return FieldResults(
        node_ids=ids,
        coords=coords,
        displacement=displacement,
        stress=np.zeros((10, 6)),
        von_mises=coords[:, 0] * 2,
        evaluated=np.ones(10, dtype=bool),
        elements=ids.reshape(1, 10),
    )


def test_grid_is_one_valid_quadratic_tetrahedron():
    grid = field_grid(single_tet())

    assert grid.n_cells == 1 and grid.n_points == 10
    assert grid.volume == pytest.approx(1000 / 6)
    assert grid.point_data["von Mises (MPa)"].max() == pytest.approx(20.0)


def test_results_without_connectivity_cannot_be_drawn():
    fields = single_tet()
    fields.elements = None

    with pytest.raises(ValueError, match="connectivity"):
        field_grid(fields)


def test_figures_are_written(tmp_path):
    figures = render_design(single_tet(), tmp_path, design_stress=15.0)

    assert set(figures) == {"mesh", "von_mises", "displacement"}
    for path in figures.values():
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert path.stat().st_size > 5000
