import math

import pytest

from mechagent.cad.bracket import BracketParameters
from mechagent.core.requirements import DesignRequirements, FatigueRequirement
from mechagent.materials.database import get_material
from mechagent.optimization.evaluation import _buckling_criterion
from mechagent.simulation.buckling import flange_compression, flange_critical_load


def test_euler_fixed_free_column_by_hand():
    params = BracketParameters(width=54.0, flange_thickness=3.5, load_hole_height=60.0)
    aluminium = get_material("Al6061-T6")

    critical = flange_critical_load(params, aluminium)

    inertia = 54.0 * 3.5**3 / 12
    assert critical == pytest.approx(math.pi**2 * 68900 * inertia / (4 * 60.0**2))


@pytest.mark.parametrize(
    "force_z, min_factor, expected",
    [(-2000, 1.0, 2000), (2000, 1.0, 0.0), (2000, -0.5, 1000), (-2000, 0.0, 2000)],
)
def test_flange_compression_over_the_load_cycle(force_z, min_factor, expected):
    assert flange_compression(force_z, min_factor) == pytest.approx(expected)


def test_buckling_criterion_without_compression_passes():
    requirements = DesignRequirements(force=(-2000, 0, 0), min_buckling_factor=3.0)

    check = _buckling_criterion(BracketParameters(), get_material("Al6061-T6"), requirements)

    assert check.passed and math.isinf(check.value)


def test_thin_flange_fails_the_screening():
    requirements = DesignRequirements(
        force=(0, 0, 2000),
        min_buckling_factor=3.0,
        fatigue=FatigueRequirement(cycles=1e6, load_ratio=-1.0),
    )
    thin = BracketParameters(flange_thickness=2.0)

    check = _buckling_criterion(thin, get_material("Al6061-T6"), requirements)

    assert not check.passed  # fully reversed: the flange sees 2000 N in compression
