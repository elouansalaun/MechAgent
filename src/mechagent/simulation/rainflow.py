"""Rainflow cycle counting (ASTM E1049-85, section 5.4.4).

We keep the two extreme values of every counted cycle instead of only its range and mean:
the fatigue evaluation needs them to rebuild the stress tensors at both ends of the cycle.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel


class Cycle(BaseModel):
    low: float
    high: float
    count: float  # 1.0 for a full cycle, 0.5 for a half cycle

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def mean(self) -> float:
        return (self.high + self.low) / 2


def reversals(history: Sequence[float]) -> list[float]:
    """Keep only the peaks and valleys of a load history."""
    points: list[float] = []
    for value in history:
        if points and value == points[-1]:
            continue
        if len(points) >= 2 and (points[-1] - points[-2]) * (value - points[-1]) > 0:
            points[-1] = value  # still going the same way, extend the current excursion
        else:
            points.append(float(value))
    return points


def _cycle(a: float, b: float, count: float) -> Cycle:
    return Cycle(low=min(a, b), high=max(a, b), count=count)


def rainflow(history: Sequence[float]) -> list[Cycle]:
    cycles: list[Cycle] = []
    stack: list[float] = []
    start = 0  # index in the stack of the starting point S

    for point in reversals(history):
        stack.append(point)
        while len(stack) >= 3:
            x = abs(stack[-1] - stack[-2])
            y = abs(stack[-2] - stack[-3])
            if x < y:
                break
            if len(stack) - 3 == start:
                # Range Y contains the starting point: half cycle, S moves on.
                cycles.append(_cycle(stack[-3], stack[-2], 0.5))
                del stack[-3]
            else:
                cycles.append(_cycle(stack[-3], stack[-2], 1.0))
                del stack[-3:-1]

    for a, b in zip(stack, stack[1:], strict=False):
        cycles.append(_cycle(a, b, 0.5))
    return cycles
