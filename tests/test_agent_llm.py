import pytest

from mechagent.agent.llm import LMStudioLLM, strict_schema
from mechagent.agent.review import review_request
from mechagent.core.config import load_settings
from mechagent.core.environment import check_llm_server
from mechagent.core.request import EngineeringRequest
from mechagent.materials.database import load_materials
from tests.test_agent_review import EXPLICIT_TEXT


def test_strict_schema_requires_every_property():
    schema = strict_schema(EngineeringRequest)

    assert set(schema["required"]) == set(schema["properties"])
    point = schema["$defs"]["InterfacePoint"]
    assert set(point["required"]) == {"position", "axis", "hole_diameter"}
    assert point["additionalProperties"] is False


@pytest.fixture(scope="module")
def llm():
    config = load_settings().llm
    status = check_llm_server(config.base_url, config.model)
    if not status.available:
        pytest.skip(f"LM Studio is not ready: {status.detail}")
    return LMStudioLLM(config)


@pytest.mark.external
def test_live_extraction_passes_the_review(llm):
    request = llm.extract_request(EXPLICIT_TEXT)

    review = review_request(request, load_materials(), text=EXPLICIT_TEXT)

    assert review.blocking == [], [i.message for i in review.blocking]
    assert review.requirements.force == pytest.approx((-2000, 0, 0))
    assert review.requirements.fatigue.cycles == 1e6


@pytest.mark.external
def test_live_choice_is_one_of_the_options(llm):
    decision = llm.choose_action(
        "Every requirement is met. Displacement 0.40 mm against a 0.50 mm limit.",
        ["accept", "refine"],
    )

    assert decision.action in {"accept", "refine"}
    assert decision.reason
