import pytest
from pydantic import ValidationError

from mechagent.core.geometry import GeometryModel
from mechagent.core.requirements import DesignRequirements, FatigueRequirement
from mechagent.materials.cost import CostModel, estimate_cost
from mechagent.materials.database import get_material


def test_constant_amplitude_requirement():
    requirement = FatigueRequirement(cycles=1e6, load_ratio=-0.5)

    assert requirement.min_load_factor == -0.5


def test_history_requirement():
    requirement = FatigueRequirement(history=[0, 1, -0.2, 0.8, 0], history_repeats=1e5)

    assert requirement.min_load_factor == -0.2


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"cycles": 1e6, "history": [0, 1], "history_repeats": 10},
        {"history": [0, 1]},
        {"cycles": 1e6, "load_ratio": 1.0},
    ],
)
def test_ambiguous_or_incomplete_fatigue_requirements(kwargs):
    with pytest.raises(ValidationError):
        FatigueRequirement(**kwargs)


def test_design_requirements_defaults():
    requirements = DesignRequirements(force=(0, 0, -2000))

    assert requirements.objective == "mass"
    assert requirements.fatigue is None
    assert requirements.min_safety_factor == 1.5


def test_cost_estimate_by_hand(tmp_path):
    geometry = GeometryModel(
        name="block",
        brep_file=tmp_path / "block.brep",
        volume=50_000.0,
        bounding_box=((0, 0, 0), (94, 44, 24)),
        face_groups={},
    )
    steel = get_material("S355J2")
    model = CostModel(machine_rate=60, setup_time=30, batch_size=5, stock_allowance=3)

    cost = estimate_cost(geometry, steel, model)

    stock = 100 * 50 * 30  # mm^3
    assert cost.stock_mass_kg == pytest.approx(stock * 7.85e-6)
    assert cost.removed_volume_cm3 == pytest.approx(100.0)
    assert cost.machining_time_min == pytest.approx(100 / 12)
    assert cost.setup_cost == pytest.approx(6.0)
    assert cost.total == pytest.approx(cost.stock_mass_kg * 1.5 + 100 / 12 / 60 * 60 + 6.0)


def test_cost_needs_price_data(tmp_path):
    geometry = GeometryModel(
        name="b",
        brep_file=tmp_path / "b.brep",
        volume=1.0,
        bounding_box=((0, 0, 0), (1, 1, 1)),
        face_groups={},
    )
    no_price = get_material("S355J2").model_copy(update={"cost_per_kg": None})

    with pytest.raises(ValueError, match="price"):
        estimate_cost(geometry, no_price)
