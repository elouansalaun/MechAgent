"""Ranking candidate materials with Ashby performance indices.

A bracket wall works mostly as a plate in bending, so for a minimum mass design the indices
are E^(1/3)/rho (stiffness), Sy^(1/2)/rho (strength) and Sf^(1/2)/rho (fatigue, Sf being the
component fatigue strength at the required life). For a minimum cost design rho is replaced
by rho * price.

Each index is normalised by the best candidate. A material scores its worst normalised
index, because whichever criterion is active will size the part. The ranking only picks
which materials are worth optimising, the final choice comes from the optimised designs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from mechagent.materials.fatigue import FatigueSettings, SNCurve
from mechagent.materials.material import Material

Objective = Literal["mass", "cost"]


class MaterialRanking(BaseModel):
    name: str
    indices: dict[str, float]
    normalised: dict[str, float]
    score: float
    governing: str  # index giving the score


class RejectedMaterial(BaseModel):
    name: str
    reason: str


class SelectionResult(BaseModel):
    objective: Objective
    criteria: list[str]
    ranking: list[MaterialRanking]
    rejected: list[RejectedMaterial]


def performance_indices(
    material: Material,
    objective: Objective,
    criteria: list[str],
    required_cycles: float | None = None,
    fatigue_settings: FatigueSettings | None = None,
) -> dict[str, float]:
    weight = material.density * 1e9  # tonne/mm^3 -> kg/dm^3, only the ratio matters
    if objective == "cost":
        weight *= material.cost_per_kg

    indices = {}
    if "stiffness" in criteria:
        indices["stiffness"] = material.youngs_modulus ** (1 / 3) / weight
    if "strength" in criteria:
        indices["strength"] = material.yield_strength**0.5 / weight
    if "fatigue" in criteria:
        curve = SNCurve.for_component(
            material.fatigue, material.ultimate_strength, fatigue_settings or FatigueSettings()
        )
        indices["fatigue"] = curve.strength_at(required_cycles) ** 0.5 / weight
    return indices


def rank_materials(
    materials: dict[str, Material],
    objective: Objective = "mass",
    stiffness: bool = True,
    required_cycles: float | None = None,
    fatigue_settings: FatigueSettings | None = None,
    candidates: list[str] | None = None,
) -> SelectionResult:
    criteria = ["strength"] + (["stiffness"] if stiffness else [])
    if required_cycles is not None:
        criteria.append("fatigue")

    rejected, kept = [], {}
    for name, material in materials.items():
        if (
            candidates is not None
            and name not in candidates
            and material.category not in candidates
        ):
            continue
        if objective == "cost" and material.cost_per_kg is None:
            rejected.append(
                RejectedMaterial(name=name, reason="no price data for a cost objective")
            )
        elif "fatigue" in criteria and material.fatigue is None:
            rejected.append(RejectedMaterial(name=name, reason="no S-N data for a cyclic load"))
        else:
            kept[name] = performance_indices(
                material, objective, criteria, required_cycles, fatigue_settings
            )

    best = {c: max(indices[c] for indices in kept.values()) for c in criteria} if kept else {}
    ranking = []
    for name, indices in kept.items():
        normalised = {c: indices[c] / best[c] for c in criteria}
        governing = min(normalised, key=normalised.get)
        ranking.append(
            MaterialRanking(
                name=name,
                indices=indices,
                normalised=normalised,
                score=normalised[governing],
                governing=governing,
            )
        )
    ranking.sort(key=lambda r: r.score, reverse=True)
    return SelectionResult(
        objective=objective, criteria=criteria, ranking=ranking, rejected=rejected
    )
