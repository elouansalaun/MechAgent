import pytest
from pydantic import ValidationError

from mechagent.core.checks import Check, all_passed, check_above, check_below, failed
from mechagent.core.geometry import GeometryModel, HoleFeature
from mechagent.materials.material import Material

STEEL = dict(
    name="S355",
    youngs_modulus=210000.0,
    poisson_ratio=0.3,
    density=7.85e-9,
    yield_strength=355.0,
    ultimate_strength=490.0,
)


def test_checks_compare_against_limits():
    assert check_below("disp", 0.4, 0.5).passed
    assert not check_below("disp", 0.6, 0.5).passed
    assert check_above("sf", 2.1, 2.0).passed
    assert not check_above("sf", 1.9, 2.0).passed


def test_empty_check_list_is_not_a_success():
    assert not all_passed([])


def test_one_failed_check_fails_everything():
    checks = [Check(name="a", passed=True), Check(name="b", passed=False)]

    assert not all_passed(checks)
    assert [c.name for c in failed(checks)] == ["b"]


def test_material_shear_modulus():
    steel = Material(**STEEL)

    assert steel.shear_modulus == pytest.approx(210000.0 / 2.6)


def test_material_rejects_density_in_si_units():
    with pytest.raises(ValidationError, match="tonne/mm"):
        Material(**{**STEEL, "density": 7850.0})


def test_material_rejects_ultimate_below_yield():
    with pytest.raises(ValidationError):
        Material(**{**STEEL, "ultimate_strength": 300.0})


def test_geometry_size_and_hole_lookup(tmp_path):
    geometry = GeometryModel(
        name="block",
        brep_file=tmp_path / "block.brep",
        volume=1000.0,
        bounding_box=((0, 0, 0), (3, 4, 0)),
        face_groups={},
        holes=[HoleFeature(name="load_hole", center=(0, 0, 0), axis=(1, 0, 0), diameter=10)],
    )

    assert geometry.size == pytest.approx(5.0)
    assert geometry.hole("load_hole").diameter == 10
    with pytest.raises(KeyError):
        geometry.hole("missing")
