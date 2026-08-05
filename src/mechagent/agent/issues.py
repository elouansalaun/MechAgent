"""Problems found while reviewing a request."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

IssueKind = Literal["missing", "inconsistent", "unverified", "unsupported", "assumption"]


class RequirementIssue(BaseModel):
    field: str
    kind: IssueKind
    message: str
    blocking: bool  # the agent cannot go on without an answer from the user


def missing(field: str, message: str) -> RequirementIssue:
    return RequirementIssue(field=field, kind="missing", message=message, blocking=True)


def inconsistent(field: str, message: str) -> RequirementIssue:
    return RequirementIssue(field=field, kind="inconsistent", message=message, blocking=True)


def assumption(field: str, message: str) -> RequirementIssue:
    return RequirementIssue(field=field, kind="assumption", message=message, blocking=False)
