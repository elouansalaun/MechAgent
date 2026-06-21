import pytest

from mechagent.cad.bracket import BracketParameters, build_bracket
from mechagent.materials.material import Material
from mechagent.simulation.bracket import bracket_static_problem
from mechagent.simulation.convergence import run_convergence_study
from mechagent.simulation.mesh import MeshSettings

ALUMINIUM = Material(
    name="Al6061-T6",
    youngs_modulus=68900,
    poisson_ratio=0.33,
    density=2.70e-9,
    yield_strength=276,
    ultimate_strength=310,
)


@pytest.fixture(scope="module")
def problem(tmp_path_factory):
    params = BracketParameters()
    geometry = build_bracket(params, tmp_path_factory.mktemp("bracket"))
    mesh = MeshSettings(element_size=6.0, refinements={"bolt_holes": 3.0, "load_hole": 3.0})
    return bracket_static_problem(geometry, params, ALUMINIUM, (0, 0, -2000), mesh)


@pytest.mark.external
def test_bracket_displacement_converges(problem, tmp_path, require_ccx):
    study, finest = run_convergence_study(problem, tmp_path, refinement_factor=0.7, n_levels=2)

    sizes = [level.element_size for level in study.levels]
    nodes = [level.n_nodes for level in study.levels]
    assert sizes == pytest.approx([6.0, 4.2])
    assert nodes[1] > nodes[0]
    assert study.displacement_change < 0.02
    assert finest.n_nodes == nodes[-1]
    assert {c.name for c in study.checks} == {
        "mesh_convergence_displacement",
        "mesh_convergence_stress",
    }


@pytest.mark.parametrize("kwargs", [{"refinement_factor": 1.2}, {"n_levels": 1}])
def test_invalid_study_settings(problem, tmp_path, kwargs):
    with pytest.raises(ValueError):
        run_convergence_study(problem, tmp_path, **kwargs)
