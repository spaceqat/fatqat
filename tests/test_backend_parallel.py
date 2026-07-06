import pytest

import fatqcat as fqc
from fatqcat import operations as ops
from fatqcat.backends import StateVectorBackend


def _random_dynamic_program():
    p = fqc.Program(2, 2)
    p.add(ops.H, 0)
    p.add(ops.CX, (0, 1))
    p.add_measurement((0, 1), (0, 1))
    p.add(fqc.ops.Reset, (0, 1))
    return p


@pytest.mark.parametrize("parallel_mode,seed", [
    ("multiprocessing", 2026),
    ("loky", 7),
    ("auto", 99),
])
def test_parallel_counts_match_serial_for_same_seed(parallel_mode, seed):
    p = _random_dynamic_program()

    serial = StateVectorBackend(options={"max_workers": 1}).run(
        p, shots=40, seed=seed
    ).result().get_counts()
    parallel = StateVectorBackend(
        options={"max_workers": 2, "parallel_mode": parallel_mode}
    ).run(p, shots=40, seed=seed).result().get_counts()

    assert parallel == serial


def test_parallel_mode_serial_wins_over_max_workers():
    p = _random_dynamic_program()

    counts = StateVectorBackend(
        options={"max_workers": 2, "parallel_mode": "serial"}
    ).run(p, shots=12, seed=11).result().get_counts()

    assert sum(counts.values()) == 12


def test_unknown_parallel_mode_rejected_at_construction():
    # Option values are validated when the backend is constructed, so an
    # unknown parallel_mode fails fast here rather than being deferred to a
    # run and swallowed into a failed Job.
    with pytest.raises(fqc.errors.BackendValidationError, match="unsupported parallel_mode"):
        StateVectorBackend(options={"max_workers": 2, "parallel_mode": "not-a-mode"})
