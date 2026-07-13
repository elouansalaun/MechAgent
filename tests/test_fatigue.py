import math

import numpy as np
import pytest

from mechagent.materials.database import get_material
from mechagent.materials.fatigue import FatigueSettings, SNCurve
from mechagent.simulation.fatigue import (
    LoadBlock,
    ReferenceStress,
    blocks_from_history,
    constant_amplitude,
    cycle_amplitude_mean,
    equivalent_amplitude,
    evaluate_fatigue,
    signed_von_mises,
)

NO_CORRECTION = FatigueSettings(surface_finish="polished", reliability=0.5, size_factor=1.0)


@pytest.fixture(scope="module")
def al6061():
    return get_material("Al6061-T6")


@pytest.fixture(scope="module")
def steel():
    return get_material("S355J2")


def uniaxial_field(stress: float, n: int = 5) -> tuple[ReferenceStress, np.ndarray, np.ndarray]:
    field = np.zeros((n, 6))
    field[:, 0] = stress
    return ReferenceStress(positive=field), np.zeros((n, 3)), np.ones(n, dtype=bool)


@pytest.mark.parametrize(
    "method, expected",
    [
        ("goodman", 100 / (1 - 155 / 310)),
        ("gerber", 100 / (1 - (155 / 310) ** 2)),
        ("soderberg", 100 / (1 - 155 / 276)),
        ("swt", math.sqrt(255 * 100)),
    ],
)
def test_mean_stress_corrections_by_hand(al6061, method, expected):
    assert equivalent_amplitude(100.0, 155.0, al6061, method) == pytest.approx(expected)


@pytest.mark.parametrize("method", ["goodman", "gerber", "soderberg"])
def test_compressive_mean_gives_no_benefit(al6061, method):
    assert equivalent_amplitude(100.0, -150.0, al6061, method) == pytest.approx(100.0)


def test_mean_above_ultimate_is_a_static_failure(al6061):
    assert np.isinf(equivalent_amplitude(10.0, 320.0, al6061, "goodman"))


def test_signed_von_mises_keeps_the_sign_of_the_dominant_stress():
    tension = np.array([[100.0, 0, 0, 0, 0, 0]])
    compression = np.array([[-100.0, 20.0, 0, 0, 0, 0]])

    assert signed_von_mises(tension) == pytest.approx([100.0])
    assert signed_von_mises(compression)[0] < 0


def test_pulsating_cycle_amplitude_and_mean():
    high = np.array([[200.0, 0, 0, 0, 0, 0]])

    amplitude, mean = cycle_amplitude_mean(np.zeros_like(high), high)

    assert amplitude == pytest.approx([100.0])
    assert mean == pytest.approx([100.0])


def test_reversed_pin_load_uses_its_own_solution():
    positive = np.array([[100.0, 0, 0, 0, 0, 0]])
    negative = np.array([[-60.0, 0, 0, 0, 0, 0]])
    reference = ReferenceStress(positive=positive, negative=negative)

    assert reference.at(0.5)[0, 0] == 50.0
    assert reference.at(-0.5)[0, 0] == -30.0
    assert ReferenceStress(positive=positive).at(-0.5)[0, 0] == -50.0


def test_fully_reversed_damage_matches_hand_calculation(al6061):
    reference, coords, evaluated = uniaxial_field(150.0)
    blocks = constant_amplitude(-1.0, 1e6)

    result = evaluate_fatigue(reference, coords, evaluated, blocks, al6061, NO_CORRECTION)

    b = math.log(96.5 / 279) / math.log(5e8 / 1e3)
    cycles_to_failure = 1e3 * (150 / 279) ** (1 / b)
    assert result.damage == pytest.approx(1e6 / cycles_to_failure)
    assert result.life_factor == pytest.approx(cycles_to_failure / 1e6)
    assert result.critical_amplitude == pytest.approx(150.0)
    assert result.critical_mean == pytest.approx(0.0, abs=1e-9)


def test_miner_adds_the_damage_of_each_block(al6061):
    reference, coords, evaluated = uniaxial_field(100.0)
    curve = SNCurve.for_component(al6061.fatigue, al6061.ultimate_strength, NO_CORRECTION)
    blocks = [LoadBlock(low=-1, high=1, cycles=1e5), LoadBlock(low=-2, high=2, cycles=1e4)]

    result = evaluate_fatigue(reference, coords, evaluated, blocks, al6061, NO_CORRECTION)

    expected = 1e5 / curve.cycles_to_failure(100.0) + 1e4 / curve.cycles_to_failure(200.0)
    assert result.damage == pytest.approx(float(expected))
    assert result.total_cycles == 1.1e5


def test_safety_factor_matches_closed_form_goodman(al6061):
    # Pulsating load, R = 0: amplitude = mean = sigma / 2, both scale with the load.
    reference, coords, evaluated = uniaxial_field(120.0)
    required = 1e7
    result = evaluate_fatigue(
        reference, coords, evaluated, constant_amplitude(0.0, required), al6061, FatigueSettings()
    )

    allowable = result.sn_curve.strength_at(required)
    closed_form = allowable / (60.0 + allowable * 60.0 / al6061.ultimate_strength)
    assert result.safety_factor == pytest.approx(closed_form, rel=1e-5)


def test_below_endurance_limit_means_no_damage(steel):
    reference, coords, evaluated = uniaxial_field(100.0)

    result = evaluate_fatigue(
        reference, coords, evaluated, constant_amplitude(-1.0, 1e9), steel, FatigueSettings()
    )

    assert result.damage == 0.0
    assert math.isinf(result.life_factor)
    assert result.safety_factor > 1.0
    assert all(c.passed for c in result.checks)


def test_overloaded_part_fails_the_validity_checks(al6061):
    reference, coords, evaluated = uniaxial_field(300.0)

    result = evaluate_fatigue(
        reference, coords, evaluated, constant_amplitude(-1.0, 1e6), al6061, FatigueSettings()
    )

    failed = {c.name for c in result.checks if not c.passed}
    assert failed == {"fatigue_cyclic_yield", "fatigue_high_cycle_range", "fatigue_damage"}


def test_excluded_nodes_are_not_evaluated(al6061):
    field = np.zeros((3, 6))
    field[:, 0] = [400.0, 100.0, 80.0]  # the first node is a support singularity
    evaluated = np.array([False, True, True])
    coords = np.arange(9.0).reshape(3, 3)

    result = evaluate_fatigue(
        ReferenceStress(positive=field),
        coords,
        evaluated,
        constant_amplitude(-1, 1e6),
        al6061,
        FatigueSettings(),
    )

    assert result.critical_location == (3.0, 4.0, 5.0)
    assert result.max_cycle_stress == pytest.approx(100.0)


def test_history_is_rainflow_counted_and_repeated():
    blocks = blocks_from_history([0, 1, -0.5, 1, 0], repeats=1000)

    assert sum(b.cycles for b in blocks) == pytest.approx(2000)
    with pytest.raises(ValueError):
        blocks_from_history([0, 1.5, 0], repeats=1)


def test_material_without_sn_data_is_refused(al6061):
    reference, coords, evaluated = uniaxial_field(100.0)
    no_data = al6061.model_copy(update={"fatigue": None})

    with pytest.raises(ValueError, match="S-N"):
        evaluate_fatigue(
            reference, coords, evaluated, constant_amplitude(-1, 1e6), no_data, FatigueSettings()
        )


def test_critical_node_with_infinite_life_is_the_most_loaded(steel):
    field = np.zeros((3, 6))
    field[:, 0] = [50.0, 120.0, 80.0]
    coords = np.arange(9.0).reshape(3, 3)

    result = evaluate_fatigue(
        ReferenceStress(positive=field),
        coords,
        np.ones(3, dtype=bool),
        constant_amplitude(-1, 1e6),
        steel,
        FatigueSettings(),
    )

    assert result.damage == 0.0
    assert result.critical_location == (3.0, 4.0, 5.0)
    assert result.critical_amplitude == pytest.approx(120.0)
