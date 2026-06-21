"""Mesh convergence study: refine the mesh until the design quantities stop changing."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from mechagent.core.checks import Check, check_below
from mechagent.core.config import SolversConfig
from mechagent.core.logging import get_logger
from mechagent.simulation.problem import StaticProblem
from mechagent.simulation.static import StaticResult, run_static_analysis

log = get_logger(__name__)


class ConvergenceLevel(BaseModel):
    element_size: float
    n_nodes: int
    max_displacement: float
    design_von_mises: float
    run_dir: Path


class ConvergenceStudy(BaseModel):
    levels: list[ConvergenceLevel]
    displacement_change: float  # relative change between the last two levels
    stress_change: float
    checks: list[Check]

    @property
    def converged(self) -> bool:
        return all(check.passed for check in self.checks)


def _relative_change(previous: float, current: float) -> float:
    return abs(current - previous) / max(abs(current), 1e-30)


def run_convergence_study(
    problem: StaticProblem,
    work_dir: Path,
    refinement_factor: float = 0.7,
    n_levels: int = 3,
    displacement_tolerance: float = 0.02,
    stress_tolerance: float = 0.05,
    solvers: SolversConfig | None = None,
) -> tuple[ConvergenceStudy, StaticResult]:
    """Solve the problem on `n_levels` meshes, each `refinement_factor` times finer.

    Returns the study and the result on the finest mesh.
    """
    if not 0 < refinement_factor < 1:
        raise ValueError("refinement factor must be between 0 and 1")
    if n_levels < 2:
        raise ValueError("a convergence study needs at least two mesh levels")

    levels, result = [], None
    for level in range(n_levels):
        mesh = problem.mesh.scaled(refinement_factor**level)
        refined = problem.model_copy(update={"mesh": mesh})
        result = run_static_analysis(refined, work_dir / f"level_{level}", solvers)
        levels.append(
            ConvergenceLevel(
                element_size=mesh.element_size,
                n_nodes=result.n_nodes,
                max_displacement=result.max_displacement,
                design_von_mises=result.design_von_mises,
                run_dir=result.run_dir,
            )
        )

    previous, last = levels[-2], levels[-1]
    displacement_change = _relative_change(previous.max_displacement, last.max_displacement)
    stress_change = _relative_change(previous.design_von_mises, last.design_von_mises)
    study = ConvergenceStudy(
        levels=levels,
        displacement_change=displacement_change,
        stress_change=stress_change,
        checks=[
            check_below(
                "mesh_convergence_displacement",
                displacement_change,
                displacement_tolerance,
                message="relative change of max displacement between the two finest meshes",
            ),
            check_below(
                "mesh_convergence_stress",
                stress_change,
                stress_tolerance,
                message="relative change of design von Mises between the two finest meshes",
            ),
        ],
    )
    log.info(
        "convergence of %s: displacement %.2f %%, stress %.2f %%, converged=%s",
        problem.name,
        100 * displacement_change,
        100 * stress_change,
        study.converged,
    )
    return study, result
