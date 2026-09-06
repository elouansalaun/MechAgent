import cadquery as cq
import numpy as np
import pytest

from mechagent.agent.interface import map_bracket_interface
from mechagent.cad.bracket import BracketParameters, build_bracket_solid
from mechagent.cad.export import export_step, placed_bracket
from tests.test_agent_review import wall_request


@pytest.fixture(scope="module")
def wall():
    interface, _ = map_bracket_interface(wall_request())
    params = BracketParameters(**{**interface.fixed, "width": 60.0})
    return interface, params


def test_placed_bracket_lands_on_the_wall(wall):
    interface, params = wall

    placed = placed_bracket(params, interface.rotation, np.array(interface.origin))

    box = placed.BoundingBox()
    # Local z (out of the wall) is global x, local x (towards the bolts) is global z.
    assert box.xmin == pytest.approx(0.0, abs=1e-6)
    assert box.xmax == pytest.approx(params.flange_height)
    assert (box.ymin, box.ymax) == pytest.approx((-30.0, 30.0))
    assert box.zmin == pytest.approx(150.0)
    assert box.zmax == pytest.approx(150.0 + params.base_length)
    assert placed.Volume() == pytest.approx(build_bracket_solid(params).Volume())


def test_step_file_round_trip(wall, tmp_path):
    interface, params = wall

    path = export_step(params, interface.rotation, np.array(interface.origin), tmp_path / "b.step")
    imported = cq.importers.importStep(str(path)).val()

    assert imported.Volume() == pytest.approx(build_bracket_solid(params).Volume(), rel=1e-6)
    assert imported.BoundingBox().zmin == pytest.approx(150.0, abs=1e-6)


def test_mirroring_is_refused():
    mirror = np.diag([1.0, 1.0, -1.0])

    with pytest.raises(ValueError, match="rotation"):
        placed_bracket(BracketParameters(), mirror, np.zeros(3))
