"""Rough cost of a CNC machined part: stock material, machining time and setup.

It is meant to compare designs and materials, not to quote a price. The stock is the
bounding box plus an allowance on every side, all of it is bought and the removed volume is
machined at the material's typical removal rate.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from mechagent.core.geometry import GeometryModel
from mechagent.materials.material import Material


class CostModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    machine_rate: float = Field(80.0, gt=0, description="EUR/h")
    setup_time: float = Field(60.0, ge=0, description="min per batch")
    batch_size: int = Field(10, gt=0)
    stock_allowance: float = Field(3.0, ge=0, description="mm on each side")


class CostEstimate(BaseModel):
    stock_mass_kg: float
    material_cost: float  # EUR
    removed_volume_cm3: float
    machining_time_min: float
    machining_cost: float
    setup_cost: float
    total: float


def estimate_cost(
    geometry: GeometryModel, material: Material, model: CostModel | None = None
) -> CostEstimate:
    model = model or CostModel()
    if material.cost_per_kg is None or material.removal_rate is None:
        raise ValueError(f"material {material.name} has no price or removal rate data")

    low, high = (np.array(corner) for corner in geometry.bounding_box)
    stock_volume = float(np.prod(high - low + 2 * model.stock_allowance))  # mm^3
    stock_mass = stock_volume * material.density * 1000.0  # kg
    removed = (stock_volume - geometry.volume) / 1000.0  # cm^3

    machining_time = removed / material.removal_rate
    machining_cost = machining_time / 60 * model.machine_rate
    setup_cost = model.setup_time / 60 * model.machine_rate / model.batch_size
    material_cost = stock_mass * material.cost_per_kg
    return CostEstimate(
        stock_mass_kg=stock_mass,
        material_cost=material_cost,
        removed_volume_cm3=removed,
        machining_time_min=machining_time,
        machining_cost=machining_cost,
        setup_cost=setup_cost,
        total=material_cost + machining_cost + setup_cost,
    )
