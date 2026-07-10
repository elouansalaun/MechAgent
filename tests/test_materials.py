import math

import numpy as np
import pytest
from pydantic import ValidationError

from mechagent.materials.database import get_material, load_materials
from mechagent.materials.fatigue import FatigueSettings, SNCurve, SNData, surface_factor
from mechagent.materials.selection import rank_materials

# Settings that leave the specimen curve unchanged.
NO_CORRECTION = FatigueSettings(surface_finish="polished", reliability=0.5, size_factor=1.0)


@pytest.fixture(scope="module")
def materials():
    return load_materials()


def test_database_loads_all_candidates(materials):
    assert set(materials) == {
        "Al6061-T6",
        "Al7075-T6",
        "S355J2",
        "42CrMo4-QT",
        "X5CrNi18-10",
        "Ti-6Al-4V",
    }
    for material in materials.values():
        assert material.fatigue is not None
        assert material.cost_per_kg and material.removal_rate
        assert material.fatigue.strength_at_1e3 < material.ultimate_strength


def test_unknown_material_lists_the_available_ones():
    with pytest.raises(KeyError, match="Al6061-T6"):
        get_material("unobtainium")


def test_sn_curve_passes_through_its_data_points(materials):
    steel = materials["S355J2"]
    curve = SNCurve.for_component(steel.fatigue, steel.ultimate_strength, NO_CORRECTION)

    assert curve.cycles_to_failure(459.0) == pytest.approx(1e3)
    assert curve.cycles_to_failure(255.0) == pytest.approx(1e6)
    assert curve.strength_at(1e4) == pytest.approx(459 * (255 / 459) ** (1 / 3))
    assert np.isinf(curve.cycles_to_failure(254.0))  # below the endurance limit
    assert curve.strength_at(1e9) == pytest.approx(255.0)


def test_aluminium_has_no_endurance_limit(materials):
    al = materials["Al6061-T6"]
    curve = SNCurve.for_component(al.fatigue, al.ultimate_strength, NO_CORRECTION)

    assert curve.cycles_to_failure(96.5) == pytest.approx(5e8)
    assert 5e8 < curve.cycles_to_failure(80.0) < np.inf
    assert curve.strength_at(1e9) < 96.5


def test_strength_and_cycles_are_inverse(materials):
    al = materials["Al7075-T6"]
    curve = SNCurve.for_component(al.fatigue, al.ultimate_strength, FatigueSettings())

    for cycles in (2e3, 1e5, 1e7, 1e9):
        assert curve.cycles_to_failure(curve.strength_at(cycles)) == pytest.approx(cycles)


def test_marin_factors_match_hand_calculation(materials):
    steel = materials["S355J2"]
    settings = FatigueSettings(surface_finish="machined", reliability=0.99, size_factor=0.9)

    curve = SNCurve.for_component(steel.fatigue, steel.ultimate_strength, settings)

    ka = 4.51 * 510**-0.265
    assert surface_factor("machined", 510) == pytest.approx(ka)
    assert curve.marin_factor == pytest.approx(ka * 0.9 * 0.814)
    assert curve.knee_strength == pytest.approx(255 * ka * 0.9 * 0.814)
    assert curve.strength_at_1e3 == 459  # the low cycle point is not corrected


def test_surface_factor_never_exceeds_one():
    assert surface_factor("ground", 100.0) == 1.0


def test_invalid_fatigue_inputs():
    with pytest.raises(ValidationError):
        SNData(strength_at_1e3=100, knee_strength=200, knee_cycles=1e6, endurance_limit=True)
    with pytest.raises(ValidationError, match="reliability"):
        FatigueSettings(reliability=0.97)


def test_light_alloys_win_on_mass_steel_wins_on_cost(materials):
    by_mass = rank_materials(materials, objective="mass")
    by_cost = rank_materials(materials, objective="cost")

    assert by_mass.ranking[0].name == "Al7075-T6"
    assert by_mass.ranking[-1].name in {"S355J2", "X5CrNi18-10"}
    assert by_cost.ranking[0].name == "S355J2"
    assert by_cost.ranking[-1].name == "Ti-6Al-4V"


def test_stiffness_index_by_hand(materials):
    result = rank_materials(materials, objective="mass", candidates=["Al6061-T6", "S355J2"])
    al = next(r for r in result.ranking if r.name == "Al6061-T6")

    assert al.indices["stiffness"] == pytest.approx(68900 ** (1 / 3) / 2.70)
    assert al.indices["strength"] == pytest.approx(math.sqrt(276) / 2.70)


def test_fatigue_criterion_rejects_materials_without_sn_data(materials):
    no_data = materials["S355J2"].model_copy(update={"name": "mystery", "fatigue": None})
    pool = {"S355J2": materials["S355J2"], "mystery": no_data}

    result = rank_materials(pool, required_cycles=1e6)

    assert [r.name for r in result.ranking] == ["S355J2"]
    assert result.rejected[0].name == "mystery"
    assert "fatigue" in result.criteria


def test_candidates_filter_by_name_or_category(materials):
    result = rank_materials(materials, candidates=["aluminium", "Ti-6Al-4V"])

    assert {r.name for r in result.ranking} == {"Al6061-T6", "Al7075-T6", "Ti-6Al-4V"}
