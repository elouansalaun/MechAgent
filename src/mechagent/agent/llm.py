"""The language model side of the agent: reading a specification and choosing between options.

We use the local LM Studio server through its OpenAI compatible API. The LLM never computes
anything: it extracts a request (checked afterwards against the text) and picks one action
among the ones the rules allow (checked afterwards against that list).

With Qwen 3.5 9B, reasoning mode spent more than 10 000 tokens without answering on a
request extraction, so we switch reasoning off and constrain the answer with a JSON schema.
"""

from __future__ import annotations

import copy
import json
from typing import Literal, Protocol

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, create_model

from mechagent.core.config import LLMConfig
from mechagent.core.request import EngineeringRequest

EXTRACTION_PROMPT = """You turn an engineering specification into JSON. Rules:
- Copy only values the text states. When a value is not stated, write null. Never guess.
- Units: positions and lengths in mm, forces in N, mass in kg, frequency in Hz.
- Directions are vectors in the global x, y, z frame of the text.
- mounting_points: the bolt holes. position = where the bolt axis meets the support surface.
  axis = normal of the support surface, pointing from the support into the part.
- load_point: the hole through which the load enters. position = where the hole axis meets the
  face of the part on the support side. axis = direction of the hole axis.
- force: the maximum load vector. load_type: static or cyclic. load_ratio: minimum load divided
  by maximum load (0 when the load goes from zero to its maximum).
- candidate_materials: names or families exactly as written (aluminium, steel, titanium, ...)."""

DECISION_PROMPT = """You are the engineering lead of an automated design loop. The numbers come
from verified simulation tools, do not question them. Choose the next action among the allowed
ones and give a one sentence reason based on the numbers. Prefer accepting a verified design
when every requirement is close to its limit, and spending more search only when it is likely
to improve the objective."""


class Decision(BaseModel):
    action: str
    reason: str


class AgentLLM(Protocol):
    def extract_request(self, text: str) -> EngineeringRequest: ...

    def choose_action(self, situation: str, options: list[str]) -> Decision: ...


def strict_schema(model: type[BaseModel]) -> dict:
    """JSON schema where every property is required (nullable ones may still be null).

    With optional properties the grammar accepts {} and the model happily returns it.
    """
    schema = copy.deepcopy(model.model_json_schema())

    def visit(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(schema)
    return schema


class LMStudioLLM:
    def __init__(self, config: LLMConfig, timeout: float = 300.0):
        self.chat = ChatOpenAI(
            base_url=config.base_url,
            model=config.model,
            temperature=config.temperature,
            api_key="lm-studio",  # LM Studio ignores it, the client needs one
            timeout=timeout,
            extra_body={"reasoning_effort": "none"},
        )

    def _structured(self, system: str, user: str, model: type[BaseModel]) -> BaseModel:
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": model.__name__, "schema": strict_schema(model), "strict": True},
        }
        answer = self.chat.invoke(
            [("system", system), ("human", user)], response_format=response_format
        )
        return model.model_validate_json(answer.content)

    def extract_request(self, text: str) -> EngineeringRequest:
        return self._structured(EXTRACTION_PROMPT, text, EngineeringRequest)

    def choose_action(self, situation: str, options: list[str]) -> Decision:
        choice = create_model(
            "ActionChoice", action=(Literal[tuple(options)], ...), reason=(str, ...)
        )
        user = f"{situation}\n\nAllowed actions: {json.dumps(options)}"
        answer = self._structured(DECISION_PROMPT, user, choice)
        return Decision(action=answer.action, reason=answer.reason)
