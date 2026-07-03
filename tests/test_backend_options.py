import numpy as np
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
    p.add(qs.ops.Reset, 0)

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


def test_backend_accepts_custom_implementation_map():
    from qnsim.implementation import default_implementation_map

    # Start from the defaults and override just one gate, to prove override
    # replaces one rule while the rest of the default map keeps working.
    m = default_implementation_map()
    m.register(qs.ops.X, np.eye(2, dtype=complex))  # override X with identity
    backend = StateVectorBackend(implementation_map=m)

    p = qs.Program(2, 2)
    p.add(qs.ops.X, 0)  # overridden: identity, so qubit 0 stays |0>
    p.add(qs.ops.H, 1)  # still a default rule: H|0> is an equal superposition
    p.add_measurement(0, 0)
    p.add_measurement(1, 1)
    counts = backend.run(p, shots=200, seed=0).result().get_counts()

    assert set(counts) <= {"00", "10"}  # c1 (H) varies; c0 (overridden X) is always 0
    assert counts.get("00", 0) + counts.get("10", 0) == 200


def test_backend_none_implementation_map_uses_defaults():
    backend = StateVectorBackend(implementation_map=None)

    p = qs.Program(1, 1)
    p.add(qs.ops.X, 0)
    p.add_measurement(0, 0)
    counts = backend.run(p, shots=10, seed=0).result().get_counts()

    assert counts == {"1": 10}


def test_backend_copies_implementation_map_defensively():
    from qnsim.implementation import default_implementation_map

    m = default_implementation_map()
    backend = StateVectorBackend(implementation_map=m)

    m.unregister(qs.ops.X)  # mutate the caller's map after construction

    p = qs.Program(1, 1)
    p.add(qs.ops.X, 0)
    p.add_measurement(0, 0)
    counts = backend.run(p, shots=10, seed=0).result().get_counts()

    assert counts == {"1": 10}
