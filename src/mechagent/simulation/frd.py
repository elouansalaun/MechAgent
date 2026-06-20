"""Reader for the ASCII result file (.frd) written by CalculiX.

Nodal result lines are fixed width: " -1", a 10 character node number, then 12 character
values. Values are not always separated by spaces (e.g. "1.0E+00-2.0E+00"), so we cut by
position instead of splitting on whitespace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

NODE_WIDTH = 10
VALUE_WIDTH = 12


class FrdError(ValueError):
    pass


@dataclass
class FrdBlock:
    name: str
    node_ids: np.ndarray
    values: np.ndarray  # (N, n_components)


def _parse_values(line: str) -> tuple[int, list[float]]:
    node = int(line[3 : 3 + NODE_WIDTH])
    rest = line[3 + NODE_WIDTH :].rstrip()
    values = [float(rest[i : i + VALUE_WIDTH]) for i in range(0, len(rest), VALUE_WIDTH)]
    return node, values


def read_frd(path: Path) -> dict[str, FrdBlock]:
    """Return the nodal result blocks (DISP, STRESS, ...) of the last step in the file."""
    blocks: dict[str, FrdBlock] = {}
    lines = Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith(" -4"):
            i += 1
            continue

        name = line.split()[1]
        nodes, values = [], []
        i += 1
        while i < len(lines) and not lines[i].startswith(" -3"):
            if lines[i].startswith(" -1"):
                node, row = _parse_values(lines[i])
                nodes.append(node)
                values.append(row)
            i += 1

        if not nodes:
            raise FrdError(f"result block {name} in {path} has no values")
        if len({len(row) for row in values}) != 1:
            raise FrdError(f"result block {name} in {path} has rows of different lengths")
        blocks[name] = FrdBlock(name, np.array(nodes), np.array(values))

    if not blocks:
        raise FrdError(f"no result block found in {path}")
    return blocks


def von_mises(stress: np.ndarray) -> np.ndarray:
    """Von Mises stress from components ordered SXX, SYY, SZZ, SXY, SYZ, SZX."""
    sxx, syy, szz, sxy, syz, szx = stress.T
    return np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy**2 + syz**2 + szx**2)
    )
