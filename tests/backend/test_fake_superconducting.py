"""Tests the fake 4x4 superconducting backend's device-shape and native-gate constraints."""

import numpy as np
import pytest

from fatqat import operations as ops
from fatqat.backends import FakeSuperconducting4x4Backend
from fatqat.backends.fake_superconducting import fake_superconducting_4x4_implementation_map
from fatqat.errors import BackendValidationError, UnsupportedOperationError
from fatqat.program import Program
from fatqat.registers import QuantumRegister


def test_fake_superconducting_map_exposes_native_target_keys():
    m = fake_superconducting_4x4_implementation_map()

    assert m.supports(ops.RZ)
    assert m.supports(ops.SX)
    assert m.supports(ops.CZ)
    assert m.target_keys(ops.RZ) == frozenset((i,) for i in range(16))
    assert m.target_keys(ops.SX) == frozenset((i,) for i in range(16))
    assert (0, 1) in m.target_keys(ops.CZ)
    assert (1, 0) in m.target_keys(ops.CZ)
    assert (0, 4) in m.target_keys(ops.CZ)
    assert (4, 0) in m.target_keys(ops.CZ)
    assert (0, 5) not in m.target_keys(ops.CZ)


def test_fake_superconducting_map_has_no_default_class_keyed_rules():
    # Every entry must be target-aware; a stray register()-style rule would
    # let get() silently accept illegal target keys via fallback.
    m = fake_superconducting_4x4_implementation_map()
    assert m.get(ops.RZ) is None
    assert m.get(ops.SX) is None
    assert m.get(ops.CZ) is None


def test_fake_backend_runs_native_neighbor_operations():
    p = Program(16)
    p.add(ops.SX, 0)
    p.add(ops.RZ(np.pi / 3), 0)
    p.add(ops.CZ, (0, 1))

    result = (
        FakeSuperconducting4x4Backend()
        .run(p, result_config={"counts": False, "statevector": True})
        .result()
    )

    state = result.get_statevector()
    assert state.shape == (2**16,)
    assert np.isclose(np.linalg.norm(state), 1.0)


def test_fake_backend_rejects_non_neighbor_cz():
    p = Program(16)
    p.add(ops.CZ, (0, 5))

    # Same UnsupportedOperationError type as an unsupported family (see
    # test below) — only the message distinguishes "illegal target" from
    # "no rule at all."
    with pytest.raises(UnsupportedOperationError, match="target key") as excinfo:
        FakeSuperconducting4x4Backend().run(
            p,
            result_config={"counts": False, "statevector": True},
        )

    assert isinstance(excinfo.value, BackendValidationError)


def test_fake_backend_rejects_non_native_operation_family():
    p = Program(16)
    p.add(ops.CX, (0, 1))

    with pytest.raises(UnsupportedOperationError):
        FakeSuperconducting4x4Backend().run(
            p,
            result_config={"counts": False, "statevector": True},
        )


def test_fake_backend_requires_exactly_sixteen_qubits():
    p = Program(2)
    p.add(ops.SX, 0)

    with pytest.raises(BackendValidationError, match="exactly 16"):
        FakeSuperconducting4x4Backend().run(
            p,
            result_config={"counts": False, "statevector": True},
        )


def test_fake_backend_rejects_non_qubit_dimension_registers():
    p = Program([QuantumRegister(16, dim=3)])
    p.add(ops.SX, 0)

    with pytest.raises(BackendValidationError, match="qubit dimensions"):
        FakeSuperconducting4x4Backend().run(
            p,
            result_config={"counts": False, "statevector": True},
        )


def test_fake_backend_allows_measurement_and_reset_on_any_qubit():
    # Measurement and reset are dispatched by StateVectorBackend._lower before
    # any implementation-map lookup, so they are not gated by the fake
    # backend's native-gate-only map even though it declares no rule for
    # Measurement or Reset. This test pins that intentional bypass.
    p = Program(16, 1)
    p.add(ops.Reset, 3)
    p.add_measurement(3, 0)

    result = (
        FakeSuperconducting4x4Backend()
        .run(p, shots=4, result_config={"counts": True})
        .result()
    )

    assert result.get_counts()
