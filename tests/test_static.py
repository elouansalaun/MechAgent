import json
import stat

import numpy as np
import pytest

from mechagent.cad.bracket import BracketParameters, build_bracket
from mechagent.cad.reference import box
from mechagent.core.config import SolversConfig
from mechagent.materials.material import Material
from mechagent.simulation.bracket import bracket_static_problem
from mechagent.simulation.calculix import SolverError
from mechagent.simulation.mesh import MeshSettings, generate_mesh
from mechagent.simulation.problem import Constraint, StaticProblem, SurfaceForce
from mechagent.simulation.static import (
    evaluated_nodes,
    load_fields,
    run_static_analysis,
    split_forces,
)

ALUMINIUM = Material(
    name="Al6061-T6",
    youngs_modulus=68900,
    poisson_ratio=0.33,
    density=2.70e-9,
    yield_strength=276,
    ultimate_strength=310,
)
COARSE = MeshSettings(element_size=6.0, refinements={"bolt_holes": 3.0, "load_hole": 3.0})


@pytest.fixture(scope="module")
def bracket(tmp_path_factory):
    params = BracketParameters()
    return params, build_bracket(params, tmp_path_factory.mktemp("bracket"))


@pytest.fixture(scope="module")
def box_problem(tmp_path_factory):
    geometry = box(60, 10, 10, tmp_path_factory.mktemp("box"))
    return StaticProblem(
        name="box",
        geometry=geometry,
        material=ALUMINIUM,
        constraints=[Constraint(surface="x_min")],
        loads=[SurfaceForce(surface="x_max", force=(0, 0, -10))],
        mesh=MeshSettings(element_size=5.0),
        stress_exclusion_distance=5.0,
    )


def test_split_forces_keeps_only_constrained_directions():
    forces = {1: np.array([1.0, 2.0, 3.0]), 2: np.array([10.0, 0.0, 0.0])}
    dofs = {1: {2}, 3: {1, 2, 3}}

    free, constrained = split_forces(forces, dofs)

    assert free == pytest.approx([11.0, 0.0, 3.0])
    assert constrained == pytest.approx([0.0, 2.0, 0.0])


def test_stress_exclusion_zone_only_around_supports(box_problem):
    mesh = generate_mesh(box_problem.geometry, box_problem.mesh)
    x = mesh.coords[:, 0]

    mask = evaluated_nodes(mesh, box_problem)
    assert np.all(x[mask] > 5.0 - 1e-9)
    assert np.any(x[mask] < 6.0)

    symmetric = box_problem.model_copy(
        update={"constraints": [Constraint(surface="x_min", dofs=(1,), kind="symmetry")]}
    )
    assert evaluated_nodes(mesh, symmetric).all()


@pytest.mark.external
def test_bracket_static_analysis(bracket, tmp_path, require_ccx):
    params, geometry = bracket
    problem = bracket_static_problem(geometry, params, ALUMINIUM, (0, 0, -2000), COARSE)

    result = run_static_analysis(problem, tmp_path)

    assert result.verified, [c for c in result.checks if not c.passed]
    assert {c.name for c in result.checks} == {
        "solver_success",
        "mesh_quality",
        "load_on_free_dofs",
        "equilibrium",
        "small_displacement",
        "linear_elastic",
    }
    assert np.array(result.reaction_force) == pytest.approx([0, 0, 2000], abs=1e-2)
    assert result.mass_kg == pytest.approx(geometry.volume * 2.70e-9 * 1000)
    assert result.design_von_mises <= result.max_von_mises
    assert result.yield_safety_factor == pytest.approx(276 / result.design_von_mises)
    for name in ("job.inp", "job.frd", "job.dat", "job.log", "fields.npz", "result.json"):
        assert (tmp_path / name).exists()
    assert json.loads((tmp_path / "result.json").read_text())["assumptions"]

    # The design stress must lie outside the exclusion zone around the bolt holes.
    location = np.array(result.design_von_mises_location)
    for hole in geometry.holes[:2]:
        distance = np.linalg.norm(location[:2] - np.array(hole.center[:2])) - hole.diameter / 2
        assert distance >= problem.stress_exclusion_distance - 1e-6

    fields = load_fields(tmp_path)
    assert fields.displacement.shape == (result.n_nodes, 3)
    assert fields.von_mises.max() == pytest.approx(result.max_von_mises)


@pytest.mark.external
def test_overloaded_bracket_is_not_verified(bracket, tmp_path, require_ccx):
    params, geometry = bracket
    problem = bracket_static_problem(geometry, params, ALUMINIUM, (20000, 0, 0), COARSE)

    result = run_static_analysis(problem, tmp_path)

    failed = {c.name for c in result.checks if not c.passed}
    assert not result.verified
    assert {"linear_elastic", "small_displacement"} <= failed


def test_solver_failure_raises_and_writes_no_result(box_problem, tmp_path):
    fake = tmp_path / "broken_ccx"
    fake.write_text('#!/bin/sh\necho " *ERROR reading the input deck"\nexit 201\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)

    with pytest.raises(SolverError, match="ERROR reading"):
        run_static_analysis(box_problem, tmp_path, SolversConfig(calculix_executable=str(fake)))
    assert not (tmp_path / "result.json").exists()
