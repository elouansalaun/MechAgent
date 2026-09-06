"""Files handed over with a verified design: STEP model in the global frame and figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mechagent.cad.export import export_step
from mechagent.reporting.figures import render_design
from mechagent.simulation.static import load_fields


def write_deliverables(state, work_dir: Path) -> tuple[dict[str, str], list[str]]:
    """Paths relative to `work_dir`, and notes about what could not be produced."""
    best = state.get("best")
    if state.get("status") != "verified" or best is None or best.final is None:
        return {}, []

    files: dict[str, str] = {}
    notes: list[str] = []
    final = best.final
    interface = state["review"].interface
    try:
        step = export_step(
            final.params, interface.rotation, np.array(interface.origin), work_dir / "design.step"
        )
        files["step"] = step.name
    except Exception as exc:  # a missing deliverable must not hide the verified result
        notes.append(f"STEP export failed: {exc}")

    static = final.static
    if static is not None and (static.run_dir / "fields.npz").exists():
        try:
            figures = render_design(
                load_fields(static.run_dir), work_dir / "figures", static.design_von_mises
            )
            files.update({name: str(path.relative_to(work_dir)) for name, path in figures.items()})
        except Exception as exc:
            notes.append(f"figures failed: {exc}")
    return files, notes
