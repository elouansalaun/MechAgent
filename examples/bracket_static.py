"""Phase 1 example: static analysis of the default aluminium bracket.

Run from the project root:  python examples/bracket_static.py

It first validates the solver chain on the analytical reference cases, then runs a mesh
convergence study on the bracket and prints the results with every check and assumption.
Everything is written to a new folder under runs/.
"""

from mechagent.cad.bracket import BracketParameters, build_bracket
from mechagent.core.config import load_settings
from mechagent.core.logging import setup_logging
from mechagent.materials.material import Material
from mechagent.simulation.bracket import bracket_static_problem
from mechagent.simulation.convergence import run_convergence_study
from mechagent.simulation.static import make_run_dir
from mechagent.simulation.validation import run_validation_suite

# Typical handbook values for 6061-T6 (MPa, tonne/mm^3), the material database comes in Phase 2.
ALUMINIUM_6061_T6 = Material(
    name="Al6061-T6",
    youngs_modulus=68900,
    poisson_ratio=0.33,
    density=2.70e-9,
    yield_strength=276,
    ultimate_strength=310,
    source="typical handbook values, not a certified datasheet",
)
FORCE = (0.0, 0.0, -2000.0)  # N, pin load pushing the flange hole down


def main() -> None:
    settings = load_settings()
    setup_logging(settings.logging.level, settings.resolve(settings.logging.file))
    run_dir = make_run_dir(settings.resolve(settings.project.work_dir), "bracket_static")

    print("Validating the solver chain against analytical solutions")
    for case in run_validation_suite(run_dir / "validation", settings.solvers):
        print(
            f"  {'PASS' if case.passed else 'FAIL'}  {case.name:16} {case.quantity:38} "
            f"FEM {case.fem:9.5g}  reference {case.reference:9.5g} {case.unit:4} "
            f"error {100 * case.relative_error:5.2f} % (limit {100 * case.tolerance:.1f} %)"
        )

    params = BracketParameters()
    geometry = build_bracket(params, run_dir / "geometry")
    problem = bracket_static_problem(geometry, params, ALUMINIUM_6061_T6, FORCE)

    print("\nMesh convergence study on the bracket")
    study, result = run_convergence_study(
        problem, run_dir / "convergence", solvers=settings.solvers
    )
    for level in study.levels:
        print(
            f"  h = {level.element_size:4.2f} mm  {level.n_nodes:7d} nodes  "
            f"max displacement {level.max_displacement:.4f} mm  "
            f"design von Mises {level.design_von_mises:7.2f} MPa"
        )

    print(f"\nResults on the finest mesh ({result.run_dir})")
    print(f"  mass                    {result.mass_kg * 1000:.1f} g")
    print(f"  max displacement        {result.max_displacement:.4f} mm")
    print(f"  design von Mises        {result.design_von_mises:.1f} MPa")
    print(
        f"  raw peak von Mises      {result.max_von_mises:.1f} MPa (includes support singularity)"
    )
    print(f"  safety factor on yield  {result.yield_safety_factor:.2f}")

    print("\nChecks")
    for check in result.checks + study.checks:
        value = "" if check.value is None else f"{check.value:.3g} (limit {check.limit:.3g})"
        print(f"  {'PASS' if check.passed else 'FAIL'}  {check.name:32} {value}")
    print("\nAssumptions")
    for assumption in result.assumptions:
        print(f"  - {assumption}")

    verified = result.verified and study.converged
    print(f"\nStatic analysis verified: {verified}")
    print("This is a static check only. Fatigue and design criteria come in Phase 2.")


if __name__ == "__main__":
    main()
