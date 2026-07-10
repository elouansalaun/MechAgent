from collections import defaultdict

import pytest

from mechagent.simulation.rainflow import rainflow, reversals

# Load history of the ASTM E1049-85 rainflow counting example (fig. 6).
ASTM_HISTORY = [-2, 1, -3, 5, -1, 3, -4, 4, -2]


def counts_by_range(cycles):
    counts = defaultdict(float)
    for cycle in cycles:
        counts[cycle.range] += cycle.count
    return dict(counts)


def test_astm_e1049_example():
    cycles = rainflow(ASTM_HISTORY)

    assert counts_by_range(cycles) == {3: 0.5, 4: 1.5, 6: 0.5, 8: 1.0, 9: 0.5}


def test_astm_example_keeps_cycle_extremes():
    full_cycles = [c for c in rainflow(ASTM_HISTORY) if c.count == 1.0]

    assert [(c.low, c.high) for c in full_cycles] == [(-1, 3)]


def test_reversals_drop_intermediate_points():
    assert reversals([0, 1, 2, 3, 1, 1, -1, 0, 0, 4]) == [0, 3, -1, 4]


def test_constant_amplitude_history():
    history = [0, 1] * 10 + [0]

    cycles = rainflow(history)

    assert sum(c.count for c in cycles) == pytest.approx(10)
    assert {(c.low, c.high) for c in cycles} == {(0, 1)}


def test_counting_conserves_every_excursion():
    history = [0.2, -0.7, 0.9, -0.1, 0.4, -1.0, 0.6, 0.1, 0.8, -0.3]

    cycles = rainflow(history)

    # Each half cycle is one excursion of the reversal sequence, nothing is lost.
    assert sum(2 * c.count for c in cycles) == len(reversals(history)) - 1


def test_short_histories():
    assert rainflow([]) == []
    assert rainflow([1.0]) == []
    assert counts_by_range(rainflow([0, 2])) == {2: 0.5}
