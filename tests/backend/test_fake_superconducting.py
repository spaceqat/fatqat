"""Tests the fake 4x4 superconducting backend's device-shape and native-gate constraints."""

import numpy as np
import pytest

from fatqat import operations as ops
from fatqat.backends import FakeSuperconducting4x4Backend
from fatqat.backends.fake_superconducting import fake_superconducting_4x4_implementation_map
from fatqat.errors import BackendValidationError, UnsupportedOperationError
from fatqat.program import Program
from fatqat.registers import QuantumRegister


def test_fake_superconducting_map_exposes_native_device_operands_for():
    m = fake_superconducting_4x4_implementation_map()

    assert m.supports(ops.RZ)
    assert m.supports(ops.SX)
    assert m.supports(ops.CZ)
    assert (0, 1) in m.device_operands_for(ops.CZ)
    assert (1, 0) in m.device_operands_for(ops.CZ)
    assert (0, 4) in m.device_operands_for(ops.CZ)
    assert (4, 0) in m.device_operands_for(ops.CZ)
    assert (0, 5) not in m.device_operands_for(ops.CZ)


def test_fake_superconducting_map_rz_and_sx_are_uniform():
    # RZ/SX are legal on any of the 16 qubits, so they are registered via
    # plain one unconstrained add() rather than explicit device-operand additions.
    # supports() + empty device_operands_for() is how a compiler infers "uniform,
    # legal on any target" instead of "not supported"; see
    # ImplementationMap.device_operands_for's docstring.
    m = fake_superconducting_4x4_implementation_map()
    assert m.supports(ops.RZ) and not m.device_operands_for(ops.RZ)
    assert m.supports(ops.SX) and not m.device_operands_for(ops.SX)
    assert m.implementation_for(ops.RZ) is not None
    assert m.implementation_for(ops.SX) is not None


def test_fake_superconducting_map_cz_has_no_class_keyed_rule():
    # CZ must be entirely target-aware; a stray unconstrained implementation would
    # let get() silently accept non-neighbor pairs via fallback.
    m = fake_superconducting_4x4_implementation_map()
    assert m.implementation_for(ops.CZ) is None


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
    # test below); only the message distinguishes "illegal target" from
    # "no rule at all."
    with pytest.raises(UnsupportedOperationError, match="device operands") as excinfo:
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


def test_fake_backend_accepts_fewer_than_sixteen_qubits():
    # Flat subsystem indices are assigned in declaration order, so a
    # smaller program maps deterministically onto physical qubits 0..N-1;
    # same rule as a full 16-qubit program, no ambiguity to guard against.
    p = Program(2)
    p.add(ops.SX, 0)
    p.add(ops.CZ, (0, 1))

    result = (
        FakeSuperconducting4x4Backend()
        .run(p, result_config={"counts": False, "statevector": True})
        .result()
    )

    state = result.get_statevector()
    assert state.shape == (2**2,)
    assert np.isclose(np.linalg.norm(state), 1.0)


def test_fake_backend_rejects_more_than_sixteen_qubits():
    p = Program(17)
    p.add(ops.SX, 0)

    with pytest.raises(BackendValidationError, match="at most 16"):
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
    # Measurement and reset are dispatched by SimulatorBackend._lower before
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
