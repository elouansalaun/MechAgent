"""High cycle fatigue from linear static results: S-N curve, mean stress correction, Miner.

Loads are given as load factors of one reference force, so the stress tensor of any load
state comes from scaling a static solution. A pin cannot pull on a hole, so with a bearing
load the negative load factors use a second static solution computed with the reversed force.

For each cycle (low, high) and each node:
- amplitude: von Mises stress of the half range tensor (sigma_high - sigma_low) / 2
- mean: signed von Mises stress of the mean tensor (sign of its largest absolute principal)
- equivalent fully reversed amplitude from the mean stress correction
- cycles to failure from the component S-N curve, damage summed with Palmgren-Miner
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel
from scipy.optimize import brentq

from mechagent.core.checks import Check, check_above, check_below
from mechagent.core.geometry import Vec3
from mechagent.materials.fatigue import LOW_CYCLE_LIMIT, FatigueSettings, SNCurve
from mechagent.materials.material import Material
from mechagent.simulation.frd import von_mises
from mechagent.simulation.rainflow import rainflow


class LoadBlock(BaseModel):
    """`cycles` cycles between two load factors of the reference force."""

    low: float
    high: float
    cycles: float


def constant_amplitude(load_ratio: float, cycles: float) -> list[LoadBlock]:
    """Cycles between R * F_ref and F_ref, R = F_min / F_max."""
    if not -1.0 <= load_ratio < 1.0:
        raise ValueError("load ratio must be in [-1, 1)")
    return [LoadBlock(low=load_ratio, high=1.0, cycles=cycles)]


def blocks_from_history(history: Sequence[float], repeats: float) -> list[LoadBlock]:
    """Rainflow count one repetition of a load factor history, then repeat it."""
    if max(abs(v) for v in history) > 1.0 + 1e-9:
        raise ValueError("load factors must stay within [-1, 1] of the reference force")
    return [LoadBlock(low=c.low, high=c.high, cycles=c.count * repeats) for c in rainflow(history)]


@dataclass
class ReferenceStress:
    positive: np.ndarray  # (N, 6) stress under +F_ref
    negative: np.ndarray | None = None  # (N, 6) stress under -F_ref, when not just -positive

    def at(self, load_factor: float) -> np.ndarray:
        if load_factor >= 0 or self.negative is None:
            return load_factor * self.positive
        return -load_factor * self.negative


def _principal_extreme(stress: np.ndarray) -> np.ndarray:
    sxx, syy, szz, sxy, syz, szx = stress.T
    tensors = np.stack(
        [
            np.stack([sxx, sxy, szx], axis=-1),
            np.stack([sxy, syy, syz], axis=-1),
            np.stack([szx, syz, szz], axis=-1),
        ],
        axis=-2,
    )
    principal = np.linalg.eigvalsh(tensors)
    return np.where(
        np.abs(principal[:, 0]) > np.abs(principal[:, 2]), principal[:, 0], principal[:, 2]
    )


def signed_von_mises(stress: np.ndarray) -> np.ndarray:
    return np.sign(_principal_extreme(stress)) * von_mises(stress)


def cycle_amplitude_mean(low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return von_mises((high - low) / 2), signed_von_mises((high + low) / 2)


def equivalent_amplitude(
    amplitude: np.ndarray, mean: np.ndarray, material: Material, method: str
) -> np.ndarray:
    """Fully reversed amplitude with the same life. Compressive means get no benefit."""
    amplitude = np.asarray(amplitude, dtype=float)
    mean = np.asarray(mean, dtype=float)
    tensile = np.maximum(mean, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        if method == "goodman":
            denominator = 1 - tensile / material.ultimate_strength
        elif method == "gerber":
            denominator = 1 - (tensile / material.ultimate_strength) ** 2
        elif method == "soderberg":
            denominator = 1 - tensile / material.yield_strength
        elif method == "swt":
            maximum = amplitude + mean
            return np.where(maximum > 0, np.sqrt(np.maximum(maximum, 0) * amplitude), 0.0)
        else:
            raise ValueError(f"unknown mean stress correction '{method}'")
        # A mean stress at or above the strength fails statically: infinite amplitude.
        return np.where(denominator > 0, amplitude / denominator, np.inf)


@dataclass
class _BlockStresses:
    amplitude: np.ndarray  # (blocks, nodes) at load scale 1
    mean: np.ndarray
    peak: np.ndarray  # max von Mises of the two cycle ends
    cycles: np.ndarray  # (blocks,)


class FatigueResult(BaseModel):
    material: str
    sn_curve: SNCurve
    settings: FatigueSettings
    allowable_damage: float
    damage: float  # max over the evaluated nodes
    life_factor: float  # allowable damage / damage, how many design lives the part lasts
    safety_factor: float  # load factor that brings the damage to the allowable value
    critical_location: Vec3
    critical_amplitude: float  # MPa, most damaging block at the critical node
    critical_mean: float
    critical_equivalent_amplitude: float
    min_cycles_to_failure: float  # over all blocks and evaluated nodes
    max_cycle_stress: float  # MPa, max von Mises at a cycle end over evaluated nodes
    total_cycles: float
    checks: list[Check]
    assumptions: list[str]


def _damage(
    stresses: _BlockStresses, scale: float, material: Material, curve: SNCurve, method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Per node damage and per block, per node cycles to failure at a load scale."""
    amplitude = equivalent_amplitude(
        scale * stresses.amplitude, scale * stresses.mean, material, method
    )
    cycles_to_failure = curve.cycles_to_failure(amplitude)
    with np.errstate(divide="ignore"):
        damage = (stresses.cycles[:, None] / cycles_to_failure).sum(axis=0)
    return damage, cycles_to_failure


def _safety_factor(target, stresses, material, curve, method) -> float:
    def excess(scale: float) -> float:
        return _damage(stresses, scale, material, curve, method)[0].max() - target

    low, high = 1e-3, 1.0
    if excess(low) >= 0:
        return low
    while excess(high) < 0:
        high *= 2
        if high > 1e4:
            return float("inf")
    return float(brentq(excess, low, high, xtol=1e-6, rtol=1e-6))


def evaluate_fatigue(
    reference: ReferenceStress,
    coords: np.ndarray,
    evaluated: np.ndarray,
    blocks: list[LoadBlock],
    material: Material,
    settings: FatigueSettings,
    allowable_damage: float = 1.0,
) -> FatigueResult:
    if material.fatigue is None:
        raise ValueError(f"material {material.name} has no S-N data")
    if not blocks:
        raise ValueError("no load cycles to evaluate")

    curve = SNCurve.for_component(material.fatigue, material.ultimate_strength, settings)
    method = settings.mean_stress_correction
    nodes = np.flatnonzero(evaluated)

    amplitudes, means, peaks = [], [], []
    for block in blocks:
        low = reference.at(block.low)[nodes]
        high = reference.at(block.high)[nodes]
        amplitude, mean = cycle_amplitude_mean(low, high)
        amplitudes.append(amplitude)
        means.append(mean)
        peaks.append(np.maximum(von_mises(low), von_mises(high)))
    stresses = _BlockStresses(
        amplitude=np.array(amplitudes),
        mean=np.array(means),
        peak=np.array(peaks),
        cycles=np.array([b.cycles for b in blocks]),
    )

    damage, cycles_to_failure = _damage(stresses, 1.0, material, curve, method)
    critical = int(np.argmax(damage))
    if damage[critical] == 0:
        # Infinite life everywhere: the critical node is the one closest to the S-N curve.
        equivalent = equivalent_amplitude(stresses.amplitude, stresses.mean, material, method)
        critical = int(np.argmax(equivalent.max(axis=0)))
    block_damage = stresses.cycles / cycles_to_failure[:, critical]
    if block_damage.max() > 0:
        worst_block = int(np.argmax(block_damage))
    else:
        worst_block = int(np.argmax(stresses.amplitude[:, critical]))
    max_damage = float(damage[critical])
    safety = _safety_factor(allowable_damage, stresses, material, curve, method)
    max_cycle_stress = float(stresses.peak.max())
    min_life = float(cycles_to_failure.min())

    checks = [
        check_below(
            "fatigue_cyclic_yield",
            max_cycle_stress,
            material.yield_strength,
            unit="MPa",
            message="stress-life is only valid without cyclic plasticity",
        ),
        check_above(
            "fatigue_high_cycle_range",
            min_life,
            LOW_CYCLE_LIMIT,
            unit="cycles",
            message="S-N curves are not valid below 1e3 cycles, use strain-life instead",
        ),
        check_below(
            "fatigue_damage",
            max_damage,
            allowable_damage,
            message="Palmgren-Miner damage over the required life",
        ),
    ]
    assumptions = [
        "Stress-life (S-N) high cycle fatigue, Palmgren-Miner linear damage accumulation.",
        "Stresses scale linearly with the load factor of the reference force. With a pin "
        "load, negative load factors use the static solution of the reversed force.",
        "Cycle amplitude: von Mises of the half range tensor. Mean stress: signed von Mises "
        "of the mean tensor (sign of the largest absolute principal stress).",
        f"Mean stress correction: {method}. Compressive mean stresses are not credited.",
        f"Component S-N curve: knee strength x Marin factor {curve.marin_factor:.3f} (surface "
        f"{settings.surface_finish}, size {settings.size_factor}, reliability "
        f"{settings.reliability}). The 1e3 cycle point is not corrected.",
        "FE stresses are used directly, no notch sensitivity reduction (Kf = Kt, conservative).",
        "Nodes in the stress exclusion zone around supports are not evaluated.",
    ]

    return FatigueResult(
        material=material.name,
        sn_curve=curve,
        settings=settings,
        allowable_damage=allowable_damage,
        damage=max_damage,
        life_factor=allowable_damage / max_damage if max_damage > 0 else float("inf"),
        safety_factor=safety,
        critical_location=tuple(coords[nodes[critical]]),
        critical_amplitude=float(stresses.amplitude[worst_block, critical]),
        critical_mean=float(stresses.mean[worst_block, critical]),
        critical_equivalent_amplitude=float(
            equivalent_amplitude(
                stresses.amplitude[worst_block, critical],
                stresses.mean[worst_block, critical],
                material,
                method,
            )
        ),
        min_cycles_to_failure=min_life,
        max_cycle_stress=max_cycle_stress,
        total_cycles=float(stresses.cycles.sum()),
        checks=checks,
        assumptions=assumptions,
    )
