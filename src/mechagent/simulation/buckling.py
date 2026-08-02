"""Buckling screening of the bracket flange.

We have no buckling FEM yet, so a compressed flange is screened with a conservative hand
calculation: an Euler column clamped at the base and free at the load hole,

    P_cr = pi^2 E I / (4 L^2),   I = width * t_flange^3 / 12,   L = load hole height

The ribs, the plate effect of the width and the stiffening of the base are ignored, all of
them make the real flange more stable.
"""

from __future__ import annotations

import math

from mechagent.cad.bracket import BracketParameters
from mechagent.materials.material import Material


def flange_critical_load(params: BracketParameters, material: Material) -> float:
    """Critical compressive load of the flange, N."""
    inertia = params.width * params.flange_thickness**3 / 12
    length = params.load_hole_height
    return math.pi**2 * material.youngs_modulus * inertia / (4 * length**2)


def flange_compression(force_z: float, min_load_factor: float = 1.0) -> float:
    """Largest compressive force along the flange (local z), N, over the load factors [min, 1]."""
    return max(0.0, -force_z, -force_z * min_load_factor)
