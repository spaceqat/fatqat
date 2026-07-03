"""Tests matrix implementation rules and immutable matrix payloads."""

import numpy as np
import pytest

from qnsim import operations as ops
from qnsim.implementation import (
    ApplyMatrixStep,
    FixedMatrix,
    MatrixImplementation,
    MatrixImplementationMap,
    default_implementation_map,
)


def test_fixed_gate_matrices():
    m = default_implementation_map()
    x = m.get(type(ops.X))(ops.X)
    assert np.allclose(x, [[0, 1], [1, 0]])
    cz = m.get(type(ops.CZ))(ops.CZ)
    assert np.allclose(cz, np.diag([1, 1, 1, -1]))
    cx = m.get(type(ops.CX))(ops.CX)
    assert np.allclose(cx, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


def test_h_matrix_is_unitary_and_correct():
    m = default_implementation_map()
    h = m.get(type(ops.H))(ops.H)
    assert np.allclose(h, np.array([[1, 1], [1, -1]]) / np.sqrt(2))


def test_batch1_fixed_single_qubit_gate_matrices():
    m = default_implementation_map()
    i_matrix = m.get(type(ops.I))(ops.I)
    assert np.allclose(i_matrix, np.eye(2))
    s = m.get(type(ops.S))(ops.S)
    assert np.allclose(s, [[1, 0], [0, 1j]])
    sdg = m.get(type(ops.Sdg))(ops.Sdg)
    assert np.allclose(sdg, [[1, 0], [0, -1j]])
    tdg = m.get(type(ops.Tdg))(ops.Tdg)
    assert np.allclose(tdg, [[1, 0], [0, np.exp(-1j * np.pi / 4)]])


def test_parametric_rx_reads_theta():
    m = default_implementation_map()
    theta = 0.5
    rx = m.get(ops.RX)(ops.RX(theta))
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    assert np.allclose(rx, [[c, -1j * s], [-1j * s, c]])


def test_parametric_phase_reads_theta():
    m = default_implementation_map()
    theta = 0.9
    phase = m.get(ops.Phase)(ops.Phase(theta))
    assert np.allclose(phase, [[1, 0], [0, np.exp(1j * theta)]])


def test_unregistered_class_returns_none():
    m = MatrixImplementationMap()
    assert m.get(type(ops.X)) is None


def test_apply_matrix_step_value_object():
    step = ApplyMatrixStep(matrix=np.eye(2, dtype=complex), target_indices=(3,))
    assert step.target_indices == (3,)
    with pytest.raises(ValueError):
        step.matrix[0, 0] = 5.0


def test_batch1_fixed_two_qubit_gate_matrices():
    m = default_implementation_map()
    swap = m.get(type(ops.Swap))(ops.Swap)
    assert np.allclose(swap, [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    cy = m.get(type(ops.CY))(ops.CY)
    assert np.allclose(cy, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]])


def test_batch2_fixed_two_qubit_gate_matrices():
    m = default_implementation_map()
    cs = m.get(type(ops.CS))(ops.CS)
    assert np.allclose(cs, np.diag([1, 1, 1, 1j]))
    iswap = m.get(type(ops.iSwap))(ops.iSwap)
    assert np.allclose(
        iswap, [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]]
    )


def test_parametric_cphase_reads_theta():
    m = default_implementation_map()
    theta = 1.1
    cphase = m.get(ops.CPhase)(ops.CPhase(theta))
    assert np.allclose(cphase, np.diag([1, 1, 1, np.exp(1j * theta)]))


def test_batch2_fixed_three_qubit_gate_matrices():
    m = default_implementation_map()

    ccx = m.get(type(ops.CCX))(ops.CCX)
    expected_ccx = np.eye(8, dtype=complex)
    expected_ccx[[6, 7]] = expected_ccx[[7, 6]]
    assert ccx.shape == (8, 8)
    assert np.allclose(ccx, expected_ccx)

    cswap = m.get(type(ops.CSwap))(ops.CSwap)
    expected_cswap = np.eye(8, dtype=complex)
    expected_cswap[[5, 6]] = expected_cswap[[6, 5]]
    assert cswap.shape == (8, 8)
    assert np.allclose(cswap, expected_cswap)


def test_fixed_matrix_is_a_matrix_implementation():
    rule = FixedMatrix(np.eye(2, dtype=complex))
    assert isinstance(rule, MatrixImplementation)


def test_fixed_matrix_returns_stored_matrix_regardless_of_operation():
    rule = FixedMatrix(np.array([[0, 1], [1, 0]], dtype=complex))
    assert np.allclose(rule(ops.X), [[0, 1], [1, 0]])
    assert np.allclose(rule(ops.RX(1.23)), [[0, 1], [1, 0]])


def test_fixed_matrix_rejects_non_square_matrix():
    with pytest.raises(ValueError, match="square"):
        FixedMatrix(np.zeros((2, 3)))


def test_fixed_matrix_accepts_non_power_of_two_side_length():
    # Deliberately not restricted to qubit-only power-of-two sizes: FixedMatrix
    # has no way to know what dimension its caller intends (e.g. a qutrit's
    # dim=3 gate), so it only requires squareness, not a power-of-two side.
    rule = FixedMatrix(np.eye(3, dtype=complex))
    assert np.allclose(rule(ops.X), np.eye(3))


def test_fixed_matrix_rejects_side_length_below_two():
    with pytest.raises(ValueError, match="side length"):
        FixedMatrix(np.eye(1))


def test_fixed_matrix_buffer_is_read_only():
    rule = FixedMatrix(np.eye(2, dtype=complex))
    matrix = rule(ops.X)
    with pytest.raises(ValueError):
        matrix[0, 0] = 5.0


def test_fixed_matrix_copies_input_array():
    source = np.eye(2, dtype=complex)
    rule = FixedMatrix(source)
    source[0, 0] = 99.0
    assert rule(ops.X)[0, 0] == 1.0


def test_register_accepts_operation_instance_key():
    m = MatrixImplementationMap()
    rule = FixedMatrix(np.eye(2, dtype=complex))
    m.register(ops.X, rule)
    assert m.get(ops.X) is rule
    assert m.get(type(ops.X)) is rule


def test_register_accepts_operation_class_key():
    class MyGate(ops.Operation):
        name = "MyGate"
        _num_qubits = 1

    m = MatrixImplementationMap()
    rule = FixedMatrix(np.eye(2, dtype=complex))
    m.register(MyGate, rule)
    assert m.get(MyGate) is rule
    assert m.get(MyGate()) is rule


def test_register_rejects_non_operation_key():
    m = MatrixImplementationMap()
    rule = FixedMatrix(np.eye(2, dtype=complex))
    with pytest.raises(TypeError):
        m.register("not an operation", rule)


def test_register_rejects_variable_arity_operation():
    class VariableGate(ops.Operation):
        name = "VariableGate"
        _num_qubits = None

    m = MatrixImplementationMap()
    rule = FixedMatrix(np.eye(2, dtype=complex))
    with pytest.raises(TypeError, match="variable arity"):
        m.register(VariableGate, rule)


def test_unregister_removes_by_instance_or_class():
    m = default_implementation_map()
    m.unregister(ops.T)
    assert m.get(ops.T) is None
    assert m.get(type(ops.T)) is None

    m2 = default_implementation_map()
    m2.unregister(type(ops.T))
    assert m2.get(ops.T) is None


def test_unregister_missing_operation_is_a_noop():
    m = MatrixImplementationMap()
    m.unregister(ops.X)


def test_register_wraps_bare_ndarray_in_fixed_matrix():
    m = MatrixImplementationMap()
    matrix = np.array([[0, 1], [1, 0]], dtype=complex)
    m.register(ops.X, matrix)
    rule = m.get(ops.X)
    assert isinstance(rule, FixedMatrix)
    assert np.allclose(rule(ops.X), matrix)


def test_register_rejects_ndarray_with_wrong_shape_for_operation():
    m = MatrixImplementationMap()
    with pytest.raises(ValueError, match="shape"):
        m.register(ops.X, np.eye(4, dtype=complex))


def test_register_wraps_bare_callable():
    m = MatrixImplementationMap()

    def noisy_rx(gate):
        return np.eye(2, dtype=complex)

    m.register(ops.RX, noisy_rx)
    rule = m.get(ops.RX)
    assert not isinstance(rule, FixedMatrix)
    assert np.allclose(rule(ops.RX(0.3)), np.eye(2))


def test_register_rejects_callable_with_zero_positional_args():
    m = MatrixImplementationMap()

    def bad_rule():
        return np.eye(2, dtype=complex)

    with pytest.raises(TypeError, match="one positional argument"):
        m.register(ops.RX, bad_rule)


def test_register_rejects_callable_with_two_required_positional_args():
    m = MatrixImplementationMap()

    def bad_rule(op, extra):
        return np.eye(2, dtype=complex)

    with pytest.raises(TypeError, match="one positional argument"):
        m.register(ops.RX, bad_rule)


def test_register_accepts_args_only_callable():
    m = MatrixImplementationMap()

    def flexible_rule(*args):
        return np.eye(2, dtype=complex)

    m.register(ops.RX, flexible_rule)
    assert np.allclose(m.get(ops.RX)(ops.RX(0.3)), np.eye(2))


def test_register_accepts_callable_with_optional_second_argument():
    m = MatrixImplementationMap()

    def rule_with_default(op, extra=None):
        return np.eye(2, dtype=complex)

    m.register(ops.RX, rule_with_default)


def test_register_rejects_callable_with_required_keyword_only_argument():
    m = MatrixImplementationMap()

    def bad_rule(*, theta):
        return np.eye(2, dtype=complex)

    with pytest.raises(TypeError, match="one positional argument"):
        m.register(ops.RX, bad_rule)


def test_register_rejects_ndarray_for_variable_arity_operation():
    class VariableGate(ops.Operation):
        name = "VariableGate"
        _num_qubits = None

    m = MatrixImplementationMap()
    with pytest.raises(TypeError, match="variable arity"):
        m.register(VariableGate, np.eye(2, dtype=complex))


def test_register_rejects_callable_for_variable_arity_operation():
    class VariableGate(ops.Operation):
        name = "VariableGate"
        _num_qubits = None

    def some_rule(op):
        return np.eye(2, dtype=complex)

    m = MatrixImplementationMap()
    with pytest.raises(TypeError, match="variable arity"):
        m.register(VariableGate, some_rule)


def test_register_rejects_non_callable_non_ndarray_rule():
    m = MatrixImplementationMap()
    with pytest.raises(TypeError, match="MatrixImplementation, np.ndarray, or callable"):
        m.register(ops.X, "not a rule")


def test_register_accepts_callable_when_signature_raises_value_error(monkeypatch):
    import qnsim.implementation as implementation

    def raise_value_error(_rule):
        raise ValueError("no signature found")

    monkeypatch.setattr(implementation.inspect, "signature", raise_value_error)

    m = MatrixImplementationMap()

    def some_rule(op):
        return np.eye(2, dtype=complex)

    m.register(ops.RX, some_rule)  # must not raise despite uninspectable signature
    assert np.allclose(m.get(ops.RX)(ops.RX(0.3)), np.eye(2))


def test_register_accepts_callable_when_signature_raises_type_error(monkeypatch):
    import qnsim.implementation as implementation

    def raise_type_error(_rule):
        raise TypeError("unsupported callable")

    monkeypatch.setattr(implementation.inspect, "signature", raise_type_error)

    m = MatrixImplementationMap()

    def some_rule(op):
        return np.eye(2, dtype=complex)

    m.register(ops.RX, some_rule)  # must not raise despite uninspectable signature
    assert np.allclose(m.get(ops.RX)(ops.RX(0.3)), np.eye(2))
