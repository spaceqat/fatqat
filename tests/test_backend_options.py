import pytest

import qnsim as qs
from qnsim.backends import StateVectorBackend


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
