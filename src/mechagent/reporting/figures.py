"""Pictures of a static result with PyVista, rendered off screen to PNG files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv

from mechagent.simulation.static import FieldResults

pv.OFF_SCREEN = True
WINDOW = (1400, 1000)
DEFORMED_SIZE = 0.05  # largest displacement drawn at 5 % of the part size


def field_grid(fields: FieldResults) -> pv.UnstructuredGrid:
    """Quadratic tetrahedra with the nodal results as point data.

    CalculiX C3D10 and VTK quadratic tetrahedra number their edge nodes the same way.
    """
    if fields.elements is None:
        raise ValueError("these results were saved without the mesh connectivity")
    index = np.searchsorted(fields.node_ids, fields.elements)
    cells = np.hstack([np.full((len(index), 1), 10), index]).ravel()
    types = np.full(len(index), pv.CellType.QUADRATIC_TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, types, fields.coords)
    grid.point_data["von Mises (MPa)"] = fields.von_mises
    grid.point_data["displacement (mm)"] = np.linalg.norm(fields.displacement, axis=1)
    grid.point_data["displacement vector"] = fields.displacement
    return grid


def _deformed(grid: pv.UnstructuredGrid) -> tuple[pv.DataSet, float]:
    size = np.linalg.norm(np.ptp(grid.points, axis=0))
    largest = grid.point_data["displacement (mm)"].max()
    factor = DEFORMED_SIZE * size / largest if largest > 0 else 0.0
    return grid.warp_by_vector("displacement vector", factor=factor), factor


def _plotter() -> pv.Plotter:
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW)
    plotter.set_background("white")
    return plotter


def render_mesh(fields: FieldResults, path: Path) -> Path:
    plotter = _plotter()
    plotter.add_mesh(
        field_grid(fields).extract_surface(algorithm="dataset_surface"),
        color="lightsteelblue",
        show_edges=True,
        edge_color="dimgray",
        line_width=0.3,
    )
    plotter.add_text(f"Mesh: {len(fields.node_ids)} nodes, C3D10", font_size=12, color="black")
    plotter.view_isometric()
    plotter.screenshot(str(path))
    plotter.close()
    return path


def render_field(
    fields: FieldResults, path: Path, field: str, upper: float | None = None, note: str = ""
) -> Path:
    grid = field_grid(fields)
    deformed, factor = _deformed(grid)
    plotter = _plotter()
    plotter.add_mesh(grid.extract_feature_edges(), color="gray", opacity=0.4)
    plotter.add_mesh(
        deformed.extract_surface(algorithm="dataset_surface"),
        scalars=field,
        cmap="turbo",
        clim=(0.0, upper) if upper else None,
        scalar_bar_args={"title": field, "color": "black"},
    )
    title = f"{field}, deformed shape x {factor:.0f}"
    if note:
        title += f"\n{note}"
    plotter.add_text(title, font_size=12, color="black")
    plotter.view_isometric()
    plotter.screenshot(str(path))
    plotter.close()
    return path


def render_design(fields: FieldResults, out_dir: Path, design_stress: float) -> dict[str, Path]:
    """The three figures of the report: mesh, von Mises stress and displacement."""
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "mesh": render_mesh(fields, out_dir / "mesh.png"),
        "von_mises": render_field(
            fields,
            out_dir / "von_mises.png",
            "von Mises (MPa)",
            upper=design_stress,
            note=f"colour scale capped at the design stress {design_stress:.0f} MPa, higher "
            "values sit at the rigid bolt supports",
        ),
        "displacement": render_field(fields, out_dir / "displacement.png", "displacement (mm)"),
    }
