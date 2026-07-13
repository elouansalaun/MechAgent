import stat

import numpy as np
import pytest

from mechagent.cad.reference import box
from mechagent.materials.material import Material
from mechagent.simulation.calculix import (
    SolverError,
    read_node_forces,
    run_calculix,
    write_input,
)
from mechagent.simulation.frd import FrdError, read_frd, von_mises
from mechagent.simulation.loads import nodal_forces
from mechagent.simulation.mesh import MeshSettings, generate_mesh
from mechagent.simulation.problem import Constraint, SurfaceForce

STEEL = Material(
    name="steel",
    youngs_modulus=210000,
    poisson_ratio=0.3,
    density=7.85e-9,
    yield_strength=355,
    ultimate_strength=490,
)

FRD_SAMPLE = """\
    1C
    2C                             2                                     1
 -1         1 0.00000E+00 0.00000E+00 0.00000E+00
 -1         2 1.00000E+00 0.00000E+00 0.00000E+00
 -3
  100CL  101 1.000000000           2                     0    1           1
 -4  DISP        4    1
 -5  D1          1    2    1    0
 -5  D2          1    2    2    0
 -5  D3          1    2    3    0
 -5  ALL         1    2    0    0    1ALL
 -1         1 0.00000E+00 0.00000E+00 0.00000E+00
 -1         2 4.76190E-04-1.42857E-04-1.04761E-20
 -3
  100CL  101 1.000000000           2                     0    1           1
 -4  STRESS      6    1
 -5  SXX         1    4    1    1
 -1         1 1.00000E+02 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00
 -1         2-5.00000E+01 0.00000E+00 0.00000E+00 1.00000E+01 0.00000E+00 0.00000E+00
 -3
 9999
"""

DAT_SAMPLE = """
                        S T E P       1


                                INCREMENT     1


 forces (fx,fy,fz) for set CONSTRAINED and time  0.1000000E+01

         1 -5.000000E+01  4.336809E-16  1.000000E+03
         4 -5.000000E+01 -2.000000E-16  1.000000E+03
"""


def make_script(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_frd_values_glued_together_are_split_by_position(tmp_path):
    path = tmp_path / "job.frd"
    path.write_text(FRD_SAMPLE)

    blocks = read_frd(path)

    assert list(blocks) == ["DISP", "STRESS"]
    assert blocks["DISP"].node_ids.tolist() == [1, 2]
    assert blocks["DISP"].values[1] == pytest.approx([4.7619e-4, -1.42857e-4, -1.04761e-20])
    assert blocks["STRESS"].values[1, 0] == -50.0


def test_frd_without_results_is_an_error(tmp_path):
    path = tmp_path / "job.frd"
    path.write_text("    1C\n 9999\n")

    with pytest.raises(FrdError):
        read_frd(path)


def test_von_mises_reference_states():
    uniaxial = np.array([[120.0, 0, 0, 0, 0, 0]])
    shear = np.array([[0, 0, 0, 50.0, 0, 0]])
    hydrostatic = np.array([[-80.0, -80.0, -80.0, 0, 0, 0]])

    assert von_mises(uniaxial) == pytest.approx([120.0])
    assert von_mises(shear) == pytest.approx([np.sqrt(3) * 50.0])
    assert von_mises(hydrostatic) == pytest.approx([0.0])


def test_node_forces_are_read(tmp_path):
    path = tmp_path / "job.dat"
    path.write_text(DAT_SAMPLE)

    forces = read_node_forces(path)

    assert sorted(forces) == [1, 4]
    assert forces[4] == pytest.approx([-50.0, -2e-16, 1000.0])


def test_written_deck_contains_sets_constraints_and_loads(tmp_path):
    mesh = generate_mesh(box(50, 10, 10, tmp_path), MeshSettings(element_size=5.0))
    constraints = [Constraint(surface="x_min"), Constraint(surface="y_min", dofs=(2,))]
    forces = nodal_forces(mesh, SurfaceForce(surface="x_max", force=(0, 0, -100)))

    deck = write_input(tmp_path / "job.inp", mesh, STEEL, constraints, forces).read_text()

    assert "*ELEMENT, TYPE=C3D10, ELSET=EALL" in deck
    assert "*NSET, NSET=X_MIN" in deck and "*NSET, NSET=CONSTRAINED" in deck
    assert "X_MIN, 1, 1\nX_MIN, 2, 2\nX_MIN, 3, 3\nY_MIN, 2, 2" in deck
    assert "210000, 0.3" in deck

    cload = deck.split("*CLOAD\n")[1].split("*NODE PRINT")[0].strip().splitlines()
    total_fz = sum(float(line.split(",")[2]) for line in cload if line.split(",")[1].strip() == "3")
    assert total_fz == pytest.approx(-100.0)


def test_missing_executable_raises(tmp_path):
    with pytest.raises(SolverError, match="not found"):
        run_calculix(tmp_path, "job", executable="surely_not_ccx")


def test_solver_errors_are_detected(tmp_path):
    fake = make_script(tmp_path, "fake_ccx", 'echo " *ERROR in calinput: unknown keyword"')

    run = run_calculix(tmp_path, "job", executable=str(fake))

    assert not run.success
    assert any("unknown keyword" in e for e in run.errors)
    assert run.log_file.exists()


def test_job_finished_without_results_is_not_a_success(tmp_path):
    # Exit code 0 and the usual message, but no result files: we must not trust it.
    fake = make_script(tmp_path, "fake_ccx", 'echo " Job finished"')

    run = run_calculix(tmp_path, "job", executable=str(fake))

    assert not run.success
    assert any("job.frd" in e for e in run.errors)


def test_solver_thread_count_is_passed_to_ccx(tmp_path):
    fake = make_script(tmp_path, "fake_ccx", 'echo "threads $OMP_NUM_THREADS"')

    run = run_calculix(tmp_path, "job", executable=str(fake), threads=3)

    assert "threads 3" in run.log_file.read_text()
