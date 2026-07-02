import pytest

import qnsim as qs
from qnsim.backends import (
    StateVectorBackend,
    _BackendConfig,
    _ResultRequest,
    _planned_workers,
)


def test_backend_accepts_known_options():
    backend = StateVectorBackend(
        options={"max_workers": 1, "parallel_backend": "serial"}
    )

    assert backend._config.max_workers == 1
    assert backend._config.parallel_backend == "serial"


def test_backend_warns_and_ignores_unknown_options():
    with pytest.warns(UserWarning, match="ignored unsupported backend options"):
        backend = StateVectorBackend(options={"gpu": True, "foo": 3})

    assert backend._config.max_workers is None
    assert backend._config.parallel_backend == "auto"


def test_serial_backend_option_runs_dynamic_program():
    p = qs.Program(1, 1)
    p.add(qs.ops.H, 0)
    p.add_measurement(0, 0)
    p.add(qs.ops.Reset(), 0)

    counts = (
        StateVectorBackend(options={"max_workers": 4, "parallel_backend": "serial"})
        .run(p, shots=16, seed=123)
        .result()
        .get_counts()
    )

    assert sum(counts.values()) == 16


def test_planned_workers_disables_parallel_serial_backend():
    workers = _planned_workers(
        _BackendConfig(max_workers=4, parallel_backend="serial"),
        _ResultRequest(counts=True, statevector=False),
        n_iters=16,
    )

    assert workers is None


def test_planned_workers_clamps_explicit_workers_to_iterations():
    workers = _planned_workers(
        _BackendConfig(max_workers=8, parallel_backend="multiprocessing"),
        _ResultRequest(counts=True, statevector=False),
        n_iters=3,
    )

    assert workers == 3


def test_backend_rejects_non_dict_options():
    with pytest.raises(TypeError, match="dict or None"):
        StateVectorBackend(options=object())
