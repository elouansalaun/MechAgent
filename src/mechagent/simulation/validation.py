"""Validation of the CAD -> mesh -> CalculiX chain against analytical reference solutions.

Each case runs the same code path as a real analysis and compares one quantity with a
textbook formula. A case only passes if the error is within tolerance and every
verification check of the underlying analysis passed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from mechagent.cad.reference import box, quarter_plate_with_hole
from mechagent.core.config import SolversConfig
from mechagent.core.logging import get_logger
from mechagent.materials.fatigue import FatigueSettings, SNData
from mechagent.materials.material import Material
from mechagent.simulation.fatigue import LoadBlock, ReferenceStress, evaluate_fatigue
from mechagent.simulation.mesh import MeshSettings
from mechagent.simulation.problem import Constraint, StaticProblem, SurfaceForce
from mechagent.simulation.static import load_fields, run_static_analysis

log = get_logger(__name__)

# Generic structural steel. Its yield strength is set high so that the fatigue case can use
# stresses above the endurance limit without failing the linear elastic check.
VALIDATION_STEEL = Material(
    name="validation_steel",
    youngs_modulus=210000.0,
    poisson_ratio=0.3,
    density=7.85e-9,
    yield_strength=450.0,
    ultimate_strength=490.0,
    fatigue=SNData(
        strength_at_1e3=441.0, knee_strength=245.0, knee_cycles=1e6, endurance_limit=True
    ),
)
# Specimen curve without any Marin correction, so the hand calculation stays simple.
SPECIMEN = FatigueSettings(surface_finish="polished", reliability=0.5, size_factor=1.0)


class ValidationCase(BaseModel):
    name: str
    quantity: str
    unit: str
    reference_formula: str
    reference: float
    fem: float
    tolerance: float  # relative
    analysis_verified: bool
    run_dir: Path

    @property
    def relative_error(self) -> float:
        return abs(self.fem - self.reference) / abs(self.reference)

    @property
    def passed(self) -> bool:
        return self.analysis_verified and self.relative_error <= self.tolerance


def axial_bar(work_dir: Path, solvers: SolversConfig | None = None) -> list[ValidationCase]:
    """Bar in uniform tension, held by three symmetry planes so it can contract freely."""
    length, side, force = 100.0, 10.0, 10000.0
    material = VALIDATION_STEEL
    geometry = box(length, side, side, work_dir / "geometry", name="axial_bar")
    problem = StaticProblem(
        name="axial_bar",
        geometry=geometry,
        material=material,
        constraints=[
            Constraint(surface="x_min", dofs=(1,), kind="symmetry"),
            Constraint(surface="y_min", dofs=(2,), kind="symmetry"),
            Constraint(surface="z_min", dofs=(3,), kind="symmetry"),
        ],
        loads=[SurfaceForce(surface="x_max", force=(force, 0, 0))],
        mesh=MeshSettings(element_size=5.0),
    )
    result = run_static_analysis(problem, work_dir / "run", solvers)
    fields = load_fields(result.run_dir)

    end = np.isclose(fields.coords[:, 0], length)
    area = side * side
    common = dict(analysis_verified=result.verified, run_dir=result.run_dir)

    # Fatigue: cycles between -1 and 3 times the static load, so 200 MPa amplitude and 100 MPa
    # mean. Hand calculation: Goodman, then the Basquin line through the S-N data points.
    cycles, low, high = 1e5, -1.0, 3.0
    sigma = force / area
    amplitude, mean = (high - low) / 2 * sigma, (high + low) / 2 * sigma
    sn = material.fatigue
    equivalent = amplitude / (1 - mean / material.ultimate_strength)
    exponent = np.log(sn.knee_strength / sn.strength_at_1e3) / np.log(sn.knee_cycles / 1e3)
    cycles_to_failure = 1e3 * (equivalent / sn.strength_at_1e3) ** (1 / exponent)
    fatigue = evaluate_fatigue(
        ReferenceStress(positive=fields.stress),
        fields.coords,
        fields.evaluated,
        [LoadBlock(low=low, high=high, cycles=cycles)],
        material,
        SPECIMEN,
    )

    return [
        ValidationCase(
            name="axial_bar",
            quantity="end displacement",
            unit="mm",
            reference_formula="u = F L / (E A)",
            reference=force * length / (material.youngs_modulus * area),
            fem=float(fields.displacement[end, 0].mean()),
            tolerance=1e-3,
            **common,
        ),
        ValidationCase(
            name="axial_bar",
            quantity="axial stress",
            unit="MPa",
            reference_formula="sigma = F / A",
            reference=force / area,
            fem=float(fields.stress[:, 0].mean()),
            tolerance=1e-3,
            **common,
        ),
        ValidationCase(
            name="axial_bar",
            quantity="fatigue damage, 1e5 cycles at 200 MPa amplitude, 100 MPa mean",
            unit="-",
            reference_formula="Goodman Sa / (1 - Sm / Su), Basquin N, D = n / N",
            reference=float(cycles / cycles_to_failure),
            fem=fatigue.damage,
            tolerance=1e-3,
            **common,
        ),
    ]


def cantilever(work_dir: Path, solvers: SolversConfig | None = None) -> list[ValidationCase]:
    """Clamped beam with a tip load, compared with Timoshenko beam theory."""
    length, width, height, load = 200.0, 10.0, 20.0, 500.0
    material = VALIDATION_STEEL
    geometry = box(length, width, height, work_dir / "geometry", name="cantilever")
    problem = StaticProblem(
        name="cantilever",
        geometry=geometry,
        material=material,
        constraints=[Constraint(surface="x_min")],
        loads=[SurfaceForce(surface="x_max", force=(0, 0, -load))],
        mesh=MeshSettings(element_size=4.0),
        stress_exclusion_distance=height,
    )
    result = run_static_analysis(problem, work_dir / "run", solvers)
    fields = load_fields(result.run_dir)

    inertia = width * height**3 / 12
    area = width * height
    shear_factor = 5 / 6
    bending = load * length**3 / (3 * material.youngs_modulus * inertia)
    shear = load * length / (shear_factor * material.shear_modulus * area)

    tip = np.isclose(fields.coords[:, 0], length)
    # Bending stress varies linearly along the top face, we fit it around mid span.
    top = np.isclose(fields.coords[:, 2], height)
    band = top & (np.abs(fields.coords[:, 0] - length / 2) < 0.1 * length)
    slope, intercept = np.polyfit(fields.coords[band, 0], fields.stress[band, 0], 1)

    common = dict(analysis_verified=result.verified, run_dir=result.run_dir)
    return [
        ValidationCase(
            name="cantilever",
            quantity="tip deflection",
            unit="mm",
            reference_formula="w = P L^3 / (3 E I) + P L / (k G A), k = 5/6",
            reference=bending + shear,
            fem=float(-fields.displacement[tip, 2].mean()),
            tolerance=0.02,
            **common,
        ),
        ValidationCase(
            name="cantilever",
            quantity="bending stress at mid span, top fibre",
            unit="MPa",
            reference_formula="sigma = M c / I, M = P L / 2, c = h / 2",
            reference=(load * length / 2) * (height / 2) / inertia,
            fem=float(slope * length / 2 + intercept),
            tolerance=0.02,
            **common,
        ),
    ]


def plate_with_hole(work_dir: Path, solvers: SolversConfig | None = None) -> list[ValidationCase]:
    """Finite width plate with a central hole in tension, quarter model.

    The z_min face is a mid-thickness symmetry plane, so the full plate is 2 * t thick.
    Reference: Pilkey, Peterson's Stress Concentration Factors, finite width plate with a
    central circular hole in tension, Kt based on the net section.
    """
    half_length, half_width, half_thickness, radius = 150.0, 50.0, 2.0, 10.0
    gross_stress = 100.0
    geometry = quarter_plate_with_hole(
        half_length, half_width, half_thickness, radius, work_dir / "geometry"
    )
    problem = StaticProblem(
        name="plate_with_hole",
        geometry=geometry,
        material=VALIDATION_STEEL,
        constraints=[
            Constraint(surface="x_min", dofs=(1,), kind="symmetry"),
            Constraint(surface="y_min", dofs=(2,), kind="symmetry"),
            Constraint(surface="z_min", dofs=(3,), kind="symmetry"),
        ],
        loads=[
            SurfaceForce(surface="x_max", force=(gross_stress * half_width * half_thickness, 0, 0))
        ],
        mesh=MeshSettings(element_size=5.0, refinements={"hole": 0.5}),
    )
    result = run_static_analysis(problem, work_dir / "run", solvers)
    fields = load_fields(result.run_dir)

    ratio = 2 * radius / (2 * half_width)
    kt_net = 3.00 - 3.13 * ratio + 3.66 * ratio**2 - 1.53 * ratio**3
    net_stress = gross_stress / (1 - ratio)

    return [
        ValidationCase(
            name="plate_with_hole",
            quantity="peak stress at the hole",
            unit="MPa",
            reference_formula=(
                "sigma_max = Kt sigma_net, Kt = 3.00 - 3.13 r + 3.66 r^2 - 1.53 r^3, r = d / W"
            ),
            reference=kt_net * net_stress,
            fem=float(fields.stress[:, 0].max()),
            tolerance=0.03,
            analysis_verified=result.verified,
            run_dir=result.run_dir,
        )
    ]


VALIDATION_CASES: dict[str, Callable[..., list[ValidationCase]]] = {
    "axial_bar": axial_bar,
    "cantilever": cantilever,
    "plate_with_hole": plate_with_hole,
}


def run_validation_suite(
    work_dir: Path, solvers: SolversConfig | None = None
) -> list[ValidationCase]:
    cases = []
    for name, case in VALIDATION_CASES.items():
        cases += case(work_dir / name, solvers)
    for c in cases:
        log.info(
            "validation %s / %s: FEM %.5g, reference %.5g %s, error %.3f %% -> %s",
            c.name,
            c.quantity,
            c.fem,
            c.reference,
            c.unit,
            100 * c.relative_error,
            "PASS" if c.passed else "FAIL",
        )
    return cases
