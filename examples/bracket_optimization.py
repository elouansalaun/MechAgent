"""Phase 2 example: minimum mass bracket under a cyclic load.

Run from the project root:  python examples/bracket_optimization.py [n_trials]

Requirements (from the project description): 2000 N vertical pin load applied cyclically,
1 million cycles, max displacement 0.5 mm, static safety factor 2. We assume a pulsating
load (R = 0) and a machined surface. It takes about 20 minutes with 40 trials per material.
"""

import sys

from mechagent.core.config import load_settings
from mechagent.core.logging import setup_logging
from mechagent.core.requirements import DesignRequirements, FatigueRequirement
from mechagent.optimization.optimizer import optimize_bracket_design
from mechagent.simulation.static import make_run_dir

REQUIREMENTS = DesignRequirements(
    force=(0.0, 0.0, -2000.0),
    min_safety_factor=2.0,
    max_displacement=0.5,
    fatigue=FatigueRequirement(cycles=1e6, load_ratio=0.0, min_safety_factor=1.0),
    objective="mass",
)


def main() -> None:
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    settings = load_settings()
    setup_logging(settings.logging.level, settings.resolve(settings.logging.file))
    run_dir = make_run_dir(settings.resolve(settings.project.work_dir), "bracket_optimization")

    result = optimize_bracket_design(
        REQUIREMENTS, run_dir, n_trials=n_trials, max_materials=3, solvers=settings.solvers
    )

    print("Decisions")
    for decision in result.decisions:
        print(f"  - {decision}")

    final = result.final
    if final is None:
        print("\nNo design to report.")
        return

    print(f"\nFinal design: {final.material} ({final.run_dir})")
    for name, value in final.params.model_dump().items():
        if name.startswith("rib_") and final.params.rib_thickness == 0:
            continue  # no ribs, their size and fillet do not apply
        print(f"  {name:20} {value:8.2f} mm")
    print(f"  mass                 {final.mass_kg * 1000:8.1f} g")
    print(f"  cost                 {final.cost.total:8.2f} EUR (indicative)")
    print(f"  max displacement     {final.static.max_displacement:8.3f} mm")
    print(f"  design von Mises     {final.static.design_von_mises:8.1f} MPa")
    print(f"  fatigue damage       {final.fatigue.damage:8.3g} over 1e6 cycles")
    print(f"  fatigue safety       {final.fatigue.safety_factor:8.2f} (on the load)")

    print("\nChecks")
    for check in final.verification + final.criteria + result.convergence.checks:
        value = "" if check.value is None else f"{check.value:.4g} (limit {check.limit:.4g})"
        print(f"  {'PASS' if check.passed else 'FAIL'}  {check.name:32} {value}")

    print("\nAssumptions")
    for assumption in final.static.assumptions + final.fatigue.assumptions:
        print(f"  - {assumption}")
    print(f"\nDesign valid: {result.valid}")


if __name__ == "__main__":
    main()
