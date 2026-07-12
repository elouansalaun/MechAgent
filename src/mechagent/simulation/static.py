"""Linear static analysis: mesh, write the deck, run CalculiX, read and verify the results."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from scipy.spatial import cKDTree

from mechagent.core.checks import Check, all_passed, check_below
from mechagent.core.config import SolversConfig, UnitsConfig
from mechagent.core.geometry import Vec3
from mechagent.core.logging import get_logger
from mechagent.simulation.calculix import (
    SolverError,
    SolverRun,
    read_node_forces,
    run_calculix,
    write_input,
)
from mechagent.simulation.frd import read_frd, von_mises
from mechagent.simulation.loads import combine_nodal_forces
from mechagent.simulation.mesh import Mesh, generate_mesh
from mechagent.simulation.problem import StaticProblem

log = get_logger(__name__)

JOB_NAME = "job"
MIN_ELEMENT_QUALITY = 0.1  # Gmsh minSICN
EQUILIBRIUM_TOLERANCE = 1e-3  # |reactions + applied| / |applied|
LOST_LOAD_TOLERANCE = 1e-3  # share of the applied load sitting on constrained dofs
SMALL_DISPLACEMENT_RATIO = 0.01  # max displacement / part size

GENERAL_ASSUMPTIONS = [
    "Linear elastic isotropic material.",
    "Small displacements and strains (geometrically linear), static loading.",
    "Quadratic tetrahedral elements (CalculiX C3D10) meshed with Gmsh.",
    "Nodal stresses are extrapolated from integration points and averaged by CalculiX.",
    "Design stresses ignore support nodes and nodes within the stress exclusion distance.",
    "Units: mm, N, MPa, tonne, s.",
]


class StaticResult(BaseModel):
    problem: StaticProblem
    run_dir: Path
    units: UnitsConfig = UnitsConfig()
    n_nodes: int
    n_elements: int
    mesh_min_quality: float
    mesh_mean_quality: float
    mass_kg: float
    max_displacement: float  # mm
    max_displacement_location: Vec3
    max_von_mises: float  # MPa, raw peak over all nodes, singular supports included
    max_von_mises_location: Vec3
    design_von_mises: float  # MPa, peak outside the stress exclusion zone around supports
    design_von_mises_location: Vec3
    yield_safety_factor: float  # yield strength / design_von_mises
    applied_force: Vec3  # N, total of the nodal forces written to the deck
    reaction_force: Vec3  # N, sum of the reactions on the constrained dofs only
    solver: SolverRun
    checks: list[Check]
    assumptions: list[str]

    @property
    def verified(self) -> bool:
        return all_passed(self.checks)


@dataclass
class FieldResults:
    node_ids: np.ndarray
    coords: np.ndarray
    displacement: np.ndarray  # (N, 3) mm
    stress: np.ndarray  # (N, 6) MPa, SXX SYY SZZ SXY SYZ SZX
    von_mises: np.ndarray  # (N,) MPa
    evaluated: np.ndarray  # (N,) bool, nodes outside the stress exclusion zone


def make_run_dir(base: Path, name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = base / f"{stamp}_{name}"
    run_dir.mkdir(parents=True)
    return run_dir


def load_fields(run_dir: Path) -> FieldResults:
    data = np.load(run_dir / "fields.npz")
    return FieldResults(**{key: data[key] for key in data.files})


def evaluated_nodes(mesh: Mesh, problem: StaticProblem) -> np.ndarray:
    """Mask of the nodes where stresses are used as design values (away from supports)."""
    supports = [mesh.node_sets[c.surface] for c in problem.constraints if c.kind == "support"]
    if not supports:
        return np.ones(mesh.n_nodes, dtype=bool)

    support_nodes = np.unique(np.concatenate(supports))
    excluded = np.isin(mesh.node_ids, support_nodes)
    if problem.stress_exclusion_distance > 0:
        tree = cKDTree(mesh.node_coords(support_nodes))
        distance, _ = tree.query(
            mesh.coords, distance_upper_bound=problem.stress_exclusion_distance
        )
        excluded |= distance <= problem.stress_exclusion_distance
    if excluded.all():
        raise ValueError(f"stress exclusion zone of {problem.name} covers the whole part")
    return ~excluded


def constrained_dofs(mesh: Mesh, problem: StaticProblem) -> dict[int, set[int]]:
    dofs: dict[int, set[int]] = {}
    for constraint in problem.constraints:
        for node in mesh.node_sets[constraint.surface]:
            dofs.setdefault(int(node), set()).update(constraint.dofs)
    return dofs


def split_forces(
    forces: dict[int, np.ndarray], dofs: dict[int, set[int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Sum nodal forces separately on free and on constrained degrees of freedom."""
    free, constrained = np.zeros(3), np.zeros(3)
    for node, force in forces.items():
        mask = np.array([d in dofs.get(node, ()) for d in (1, 2, 3)])
        constrained += np.where(mask, force, 0.0)
        free += np.where(mask, 0.0, force)
    return free, constrained


def _read_fields(mesh: Mesh, frd_file: Path, evaluated: np.ndarray) -> FieldResults:
    blocks = read_frd(frd_file)
    for name in ("DISP", "STRESS"):
        if name not in blocks:
            raise SolverError(f"{frd_file} has no {name} results")
        if not np.array_equal(blocks[name].node_ids, mesh.node_ids):
            raise SolverError(f"{name} results in {frd_file} do not cover every mesh node")

    stress = blocks["STRESS"].values
    return FieldResults(
        node_ids=mesh.node_ids,
        coords=mesh.coords,
        displacement=blocks["DISP"].values[:, :3],
        stress=stress,
        von_mises=von_mises(stress),
        evaluated=evaluated,
    )


def run_static_analysis(
    problem: StaticProblem, run_dir: Path, solvers: SolversConfig | None = None
) -> StaticResult:
    """Run the full static chain in `run_dir`. Raises if meshing or the solver fails."""
    solvers = solvers or SolversConfig()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "problem.json").write_text(problem.model_dump_json(indent=2))

    mesh = generate_mesh(problem.geometry, problem.mesh)
    for item in [*problem.constraints, *problem.loads]:
        if item.surface not in mesh.node_sets:
            raise KeyError(f"unknown face group '{item.surface}' in problem {problem.name}")

    forces = combine_nodal_forces(mesh, problem.loads)
    applied = np.sum(list(forces.values()), axis=0)
    write_input(
        run_dir / f"{JOB_NAME}.inp",
        mesh,
        problem.material,
        problem.constraints,
        forces,
        title=problem.name,
    )

    solver = run_calculix(run_dir, JOB_NAME, solvers.calculix_executable, solvers.timeout_s)
    if not solver.success:
        raise SolverError(
            f"CalculiX failed for {problem.name}: {'; '.join(solver.errors)} "
            f"(see {solver.log_file})"
        )

    fields = _read_fields(mesh, run_dir / f"{JOB_NAME}.frd", evaluated_nodes(mesh, problem))
    np.savez_compressed(run_dir / "fields.npz", **vars(fields))
    # Loads on a constrained dof go straight into the support, the structure never sees them.
    dofs = constrained_dofs(mesh, problem)
    effective_load, lost_load = split_forces(forces, dofs)
    _, reactions = split_forces(read_node_forces(run_dir / f"{JOB_NAME}.dat"), dofs)

    magnitude = np.linalg.norm(fields.displacement, axis=1)
    i_disp = int(np.argmax(magnitude))
    i_stress = int(np.argmax(fields.von_mises))
    evaluated = np.flatnonzero(fields.evaluated)
    i_design = evaluated[np.argmax(fields.von_mises[evaluated])]
    design_vm = float(fields.von_mises[i_design])

    applied_norm = np.linalg.norm(applied)
    equilibrium_error = float(np.linalg.norm(reactions + effective_load) / applied_norm)
    checks = [
        Check(name="solver_success", passed=True, message="ccx finished with result files"),
        Check(
            name="mesh_quality",
            passed=mesh.min_quality >= MIN_ELEMENT_QUALITY,
            value=mesh.min_quality,
            limit=MIN_ELEMENT_QUALITY,
            message="minimum Gmsh SICN element quality",
        ),
        check_below(
            "load_on_free_dofs",
            float(np.linalg.norm(lost_load) / applied_norm),
            LOST_LOAD_TOLERANCE,
            message="share of the applied load put on constrained dofs and lost to supports",
        ),
        check_below(
            "equilibrium",
            equilibrium_error,
            EQUILIBRIUM_TOLERANCE,
            message="relative difference between reactions and applied loads",
        ),
        check_below(
            "small_displacement",
            float(magnitude[i_disp] / problem.geometry.size),
            SMALL_DISPLACEMENT_RATIO,
            message="max displacement / part size, linear geometry assumption",
        ),
        check_below(
            "linear_elastic",
            design_vm,
            problem.material.yield_strength,
            unit="MPa",
            message="design von Mises stress must stay below yield for a linear elastic model",
        ),
    ]

    result = StaticResult(
        problem=problem,
        run_dir=run_dir,
        n_nodes=mesh.n_nodes,
        n_elements=mesh.n_elements,
        mesh_min_quality=mesh.min_quality,
        mesh_mean_quality=mesh.mean_quality,
        mass_kg=problem.geometry.volume * problem.material.density * 1000.0,
        max_displacement=float(magnitude[i_disp]),
        max_displacement_location=tuple(fields.coords[i_disp]),
        max_von_mises=float(fields.von_mises[i_stress]),
        max_von_mises_location=tuple(fields.coords[i_stress]),
        design_von_mises=design_vm,
        design_von_mises_location=tuple(fields.coords[i_design]),
        yield_safety_factor=problem.material.yield_strength / design_vm,
        applied_force=tuple(applied),
        reaction_force=tuple(reactions),
        solver=solver,
        checks=checks,
        assumptions=GENERAL_ASSUMPTIONS + problem.assumptions,
    )
    (run_dir / "result.json").write_text(result.model_dump_json(indent=2))
    log.info(
        "%s: max displacement %.4g mm, design von Mises %.4g MPa, verified=%s",
        problem.name,
        result.max_displacement,
        result.design_von_mises,
        result.verified,
    )
    return result
