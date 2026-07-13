"""Verification checks recorded alongside every result.

A result is only trusted when all of its checks passed. Checks never get a default `passed`
value, so forgetting to evaluate one is an error instead of a silent success.
"""

from __future__ import annotations

from pydantic import BaseModel


class Check(BaseModel):
    name: str
    passed: bool
    value: float | None = None
    limit: float | None = None
    unit: str = ""
    message: str = ""
    margin: float | None = None  # relative distance to the limit, negative when failed


def _margin(slack: float, limit: float) -> float:
    return slack / abs(limit) if limit != 0 else slack


def check_below(name: str, value: float, limit: float, unit: str = "", message: str = "") -> Check:
    return Check(
        name=name,
        passed=bool(value <= limit),
        value=value,
        limit=limit,
        unit=unit,
        message=message,
        margin=_margin(limit - value, limit),
    )


def check_above(name: str, value: float, limit: float, unit: str = "", message: str = "") -> Check:
    return Check(
        name=name,
        passed=bool(value >= limit),
        value=value,
        limit=limit,
        unit=unit,
        message=message,
        margin=_margin(value - limit, limit),
    )


def all_passed(checks: list[Check]) -> bool:
    return bool(checks) and all(check.passed for check in checks)


def failed(checks: list[Check]) -> list[Check]:
    return [check for check in checks if not check.passed]
