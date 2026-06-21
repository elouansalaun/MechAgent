"""Analytical validation of the whole CAD -> Gmsh -> CalculiX chain."""

import pytest

from mechagent.simulation.validation import run_validation_suite

pytestmark = pytest.mark.external


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    import shutil

    if shutil.which("ccx") is None:
        pytest.skip("CalculiX (ccx) is not on PATH")
    return run_validation_suite(tmp_path_factory.mktemp("validation"))


def test_every_reference_case_runs(cases):
    names = {(c.name, c.quantity) for c in cases}

    assert len(names) == 5


@pytest.mark.parametrize("index", range(5))
def test_reference_case_matches_analytical_solution(cases, index):
    case = cases[index]

    assert case.analysis_verified, f"{case.name}: a verification check failed"
    assert case.relative_error <= case.tolerance, (
        f"{case.name} / {case.quantity}: FEM {case.fem:.5g} vs {case.reference:.5g} {case.unit}"
    )
