"""Review of an engineering request: missing data, inconsistencies and default assumptions.

Blocking issues stop the agent until the user answers. Non blocking ones are defaults the
agent takes, each written down as an assumption. When the request was parsed from text by
the LLM, every number it produced must also appear in the text, and every position, axis
and force direction must appear as an explicit (x, y, z) triple. Otherwise the user has to
confirm it: an LLM must not be able to slip an invented value into a design. Directions
matter most: a 9B model reading "vertical" without thinking often picks the wrong axis.
"""

from __future__ import annotations

import re

import numpy as np
from pydantic import BaseModel

from mechagent.agent.interface import BracketInterface, map_bracket_interface
from mechagent.agent.issues import RequirementIssue, assumption, inconsistent, missing
from mechagent.core.request import EngineeringRequest, Vector
from mechagent.core.requirements import DesignRequirements, FatigueRequirement
from mechagent.materials.material import Material
from mechagent.optimization.optimizer import DEFAULT_BRACKET_SPACE, DesignSpace

DEFAULT_SAFETY_FACTOR = 1.5
DEFAULT_LOAD_RATIO = 0.0
MULTIPLIERS = {"thousand": 1e3, "k": 1e3, "kn": 1e3, "million": 1e6, "millions": 1e6}

_NUMBER = re.compile(
    r"(?<![\w.])(-?\d{1,3}(?:[\s,]\d{3})+(?:\.\d+)?|-?\d+(?:[.,]\d+)?(?:e[+-]?\d+)?)"
    r"\s*(thousand|millions?|kn|k)?\b",
    re.IGNORECASE,
)


class ReviewResult(BaseModel):
    issues: list[RequirementIssue]
    interface: BracketInterface | None = None
    requirements: DesignRequirements | None = None
    space: DesignSpace | None = None

    @property
    def blocking(self) -> list[RequirementIssue]:
        return [issue for issue in self.issues if issue.blocking]

    @property
    def assumptions(self) -> list[str]:
        return [issue.message for issue in self.issues if issue.kind == "assumption"]


def numbers_in_text(text: str) -> set[float]:
    """Every number written in the text, also with thousand separators, decimal commas,
    'kN' and words like 'million' applied."""
    values = set()
    for match in _NUMBER.finditer(text):
        raw, word = match.group(1), (match.group(2) or "").lower()
        compact = re.sub(r"\s", "", raw)
        candidates = set()
        if re.fullmatch(r"-?\d{1,3}(?:[\s,]\d{3})+(?:\.\d+)?", raw):
            candidates.add(compact.replace(",", ""))  # thousand separators: 2,000 or 2 000
        if re.fullmatch(r"-?\d+,\d+", compact):
            candidates.add(compact.replace(",", "."))  # decimal comma: 0,5
        if "," not in compact:
            candidates.add(compact)
        for candidate in candidates:
            try:
                value = abs(float(candidate))
            except ValueError:
                continue
            values.add(value)
            if word:
                values.add(value * MULTIPLIERS[word])
    return values


def _request_numbers(request: EngineeringRequest) -> dict[str, float]:
    """Scalar values of the request. Vectors are checked as whole triples."""
    numbers: dict[str, float] = {}
    points = [(f"mounting_points[{i}]", p) for i, p in enumerate(request.mounting_points)]
    if request.load_point is not None:
        points.append(("load_point", request.load_point))
    for name, point in points:
        if point.hole_diameter is not None:
            numbers[f"{name}.hole_diameter"] = point.hole_diameter
    for name in (
        "cycles",
        "load_ratio",
        "min_safety_factor",
        "max_displacement",
        "max_mass",
        "excitation_frequency",
        "max_width",
    ):
        value = getattr(request, name)
        if value is not None:
            numbers[name] = value
    return numbers


_TRIPLE = re.compile(
    r"\(\s*(-?\d+(?:\.\d+)?)\s*[,;]\s*(-?\d+(?:\.\d+)?)\s*[,;]\s*(-?\d+(?:\.\d+)?)\s*\)"
)


def triples_in_text(text: str) -> list[np.ndarray]:
    return [np.array([float(v) for v in m.groups()]) for m in _TRIPLE.finditer(text)]


def _request_vectors(request: EngineeringRequest) -> dict[str, tuple[Vector, bool]]:
    """Vectors of the request, with True when only their direction matters."""
    vectors: dict[str, tuple[Vector, bool]] = {}
    points = [(f"mounting_points[{i}]", p) for i, p in enumerate(request.mounting_points)]
    if request.load_point is not None:
        points.append(("load_point", request.load_point))
    for name, point in points:
        vectors[f"{name}.position"] = (point.position, False)
        vectors[f"{name}.axis"] = (point.axis, True)
    if request.force is not None:
        vectors["force"] = (request.force, True)
    return vectors


def unverified_vectors(request: EngineeringRequest, text: str) -> list[RequirementIssue]:
    triples = triples_in_text(text)
    issues = []
    for name, (vector, direction_only) in _request_vectors(request).items():
        value = vector.array()
        if direction_only:
            norm = np.linalg.norm(value)
            found = norm > 0 and any(
                np.linalg.norm(t) > 0 and value @ t / (norm * np.linalg.norm(t)) > 1 - 1e-6
                for t in triples
            )
        else:
            found = any(np.allclose(value, t, atol=1e-6) for t in triples)
        if not found:
            shown = ", ".join(f"{v:g}" for v in value)
            issues.append(
                RequirementIssue(
                    field=name,
                    kind="unverified",
                    message=f"{name} was read as ({shown}) but the text gives no such vector, "
                    "please confirm it or write it as (x, y, z)",
                    blocking=True,
                )
            )
    return issues


def unverified_values(request: EngineeringRequest, text: str) -> list[RequirementIssue]:
    found = numbers_in_text(text)
    issues = []
    for name, value in _request_numbers(request).items():
        if not any(abs(abs(value) - f) <= 1e-6 * max(1.0, f) for f in found):
            issues.append(
                RequirementIssue(
                    field=name,
                    kind="unverified",
                    message=f"{name} = {value:g} does not appear in the text, please confirm it",
                    blocking=True,
                )
            )
    return issues


def _known_material(name: str, materials: dict[str, Material]) -> bool:
    lowered = name.lower()
    return any(
        lowered in (key.lower(), material.category.lower()) for key, material in materials.items()
    )


def _design_space(interface: BracketInterface, max_width: float | None) -> DesignSpace:
    fixed = interface.fixed
    # Narrowest bracket that keeps a ligament of one bolt diameter beside each bolt hole.
    min_width = fixed["bolt_hole_spacing"] + 2 * fixed["bolt_hole_diameter"]
    max_width = max_width or max(DEFAULT_BRACKET_SPACE.variables[2].high, 1.5 * min_width)
    variables = [
        v.model_copy(update={"low": min_width, "high": max_width}) if v.name == "width" else v
        for v in DEFAULT_BRACKET_SPACE.variables
    ]
    return DesignSpace(variables=variables, fixed=fixed)


def review_request(
    request: EngineeringRequest, materials: dict[str, Material], text: str | None = None
) -> ReviewResult:
    issues: list[RequirementIssue] = []
    if text is not None:
        issues += unverified_values(request, text) + unverified_vectors(request, text)

    interface, interface_issues = map_bracket_interface(request)
    issues += interface_issues

    load_type = request.load_type
    if load_type is None and request.cycles is not None:
        load_type = "cyclic"
        issues.append(assumption("load_type", "the load is cyclic, a cycle count was given"))
    if load_type is None:
        issues.append(missing("load_type", "is the load static or cyclic?"))
    fatigue = None
    if load_type == "cyclic":
        if request.cycles is None:
            issues.append(missing("cycles", "a cyclic load needs the required number of cycles"))
        load_ratio = request.load_ratio
        if load_ratio is None:
            load_ratio = DEFAULT_LOAD_RATIO
            issues.append(
                assumption(
                    "load_ratio",
                    "load ratio R = 0 (the load goes from zero to its maximum), not stated",
                )
            )
        if not -1.0 <= load_ratio < 1.0:
            issues.append(inconsistent("load_ratio", "the load ratio must be in [-1, 1)"))
        elif request.cycles is not None:
            fatigue = FatigueRequirement(cycles=request.cycles, load_ratio=load_ratio)
    elif load_type == "static" and request.cycles is not None:
        issues.append(inconsistent("load_type", "a static load was given with a cycle count"))

    safety_factor = request.min_safety_factor
    if safety_factor is None:
        safety_factor = DEFAULT_SAFETY_FACTOR
        issues.append(
            assumption("min_safety_factor", f"static safety factor {safety_factor} on yield")
        )
    if request.objective is None:
        issues.append(assumption("objective", "minimum mass, no objective was stated"))
    if request.candidate_materials:
        for name in request.candidate_materials:
            if not _known_material(name, materials):
                issues.append(
                    inconsistent(
                        "candidate_materials",
                        f"unknown material '{name}', known: {', '.join(sorted(materials))}",
                    )
                )
    else:
        issues.append(assumption("candidate_materials", "every material of the database"))
    if request.manufacturing and "cnc" not in request.manufacturing.lower():
        issues.append(
            assumption("manufacturing", f"'{request.manufacturing}' treated as CNC machining")
        )
    if fatigue is not None:
        issues.append(
            assumption("fatigue", "machined surface, 99 % reliability, Goodman mean stress")
        )

    if interface is not None and request.max_width is not None:
        min_width = interface.fixed["bolt_hole_spacing"] + 2 * interface.fixed["bolt_hole_diameter"]
        if request.max_width < min_width:
            issues.append(
                inconsistent(
                    "max_width", f"the bolt holes need a width of at least {min_width:.1f} mm"
                )
            )

    result = ReviewResult(issues=issues, interface=interface)
    if result.blocking:
        return result

    candidates = None
    if request.candidate_materials:
        candidates = [_canonical(name, materials) for name in request.candidate_materials]
    result.requirements = DesignRequirements(
        force=interface.force_local,
        min_safety_factor=safety_factor,
        max_displacement=request.max_displacement,
        max_mass=request.max_mass,
        fatigue=fatigue,
        objective=request.objective or "mass",
        candidate_materials=candidates,
    )
    result.space = _design_space(interface, request.max_width)
    return result


def _canonical(name: str, materials: dict[str, Material]) -> str:
    """Database spelling of a material name or category."""
    lowered = name.lower()
    for key, material in materials.items():
        if lowered == key.lower():
            return key
        if lowered == material.category.lower():
            return material.category
    return name
