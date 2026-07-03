import pytest

import qnsim as qs
from qnsim import operations as ops
from qnsim.backends import StateVectorBackend


def _random_dynamic_program():
    p = qs.Program(2, 2)
    p.add(ops.H, 0)
    p.add(ops.CX, (0, 1))
    p.add_measurement((0, 1), (0, 1))
    p.add(qs.ops.Reset, (0, 1))
    return p


@pytest.mark.parametrize("parallel_backend,seed", [
    ("multiprocessing", 2026),
    ("loky", 7),
    ("auto", 99),
])
def test_parallel_counts_match_serial_for_same_seed(parallel_backend, seed):
    p = _random_dynamic_program()

    serial = StateVectorBackend(options={"max_workers": 1}).run(
        p, shots=40, seed=seed
    ).result().get_counts()
    parallel = StateVectorBackend(
        options={"max_workers": 2, "parallel_backend": parallel_backend}
    ).run(p, shots=40, seed=seed).result().get_counts()

    assert parallel == serial


def test_parallel_backend_serial_wins_over_max_workers():
    p = _random_dynamic_program()

    counts = StateVectorBackend(
        options={"max_workers": 2, "parallel_backend": "serial"}
    ).run(p, shots=12, seed=11).result().get_counts()

    assert sum(counts.values()) == 12


def test_unknown_parallel_backend_fails_when_consumed():
    p = _random_dynamic_program()

    job = StateVectorBackend(
        options={"max_workers": 2, "parallel_backend": "not-a-backend"}
    ).run(p, shots=4, seed=11)

    with pytest.raises(qs.BackendValidationError, match="unsupported parallel_backend"):
        job.result()
