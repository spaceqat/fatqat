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


# --- Fixed-gate registration ------------------------------------------------
# We deliberately do not re-assert each fixed gate's literal matrix: those
# constants live in the source and are unlikely to change, so a value check is
# a tautology. What is worth guarding is that every fixed gate is actually
# wired into the default map under its class key and returns a matrix of the
# right dimension.

@pytest.mark.parametrize("gate,n_qubits", [
    (ops.I, 1), (ops.H, 1), (ops.S, 1), (ops.Sdg, 1), (ops.X, 1),
    (ops.Y, 1), (ops.Z, 1), (ops.T, 1), (ops.Tdg, 1),
    (ops.CX, 2), (ops.CZ, 2), (ops.Swap, 2), (ops.CY, 2),
    (ops.CS, 2), (ops.iSwap, 2),
    (ops.CCX, 3), (ops.CSwap, 3),
])
def test_fixed_gate_is_registered_with_correct_shape(gate, n_qubits):
    m = default_implementation_map()
    matrix = m.get(type(gate))(gate)
    dim = 2 ** n_qubits
    assert matrix.shape == (dim, dim)


# --- Parametric rules read their operation's theta --------------------------
# Unlike the fixed gates above, these are not tautologies: they verify the rule
# reads `op.theta` off the bare Operation and builds the matrix from it.

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


def test_parametric_cphase_reads_theta():
    m = default_implementation_map()
    theta = 1.1
    cphase = m.get(ops.CPhase)(ops.CPhase(theta))
    assert np.allclose(cphase, np.diag([1, 1, 1, np.exp(1j * theta)]))


def test_unregistered_class_returns_none():
    m = MatrixImplementationMap()
    assert m.get(type(ops.X)) is None


def test_apply_matrix_step_value_object():
    step = ApplyMatrixStep(matrix=np.eye(2, dtype=complex), target_indices=(3,))
    assert step.target_indices == (3,)
    with pytest.raises(ValueError):
        step.matrix[0, 0] = 5.0


# --- FixedMatrix payload behavior -------------------------------------------

def test_fixed_matrix_is_a_matrix_implementation():
    rule = FixedMatrix(np.eye(2, dtype=complex))
    assert isinstance(rule, MatrixImplementation)


def test_fixed_matrix_returns_stored_matrix_regardless_of_operation():
    rule = FixedMatrix(np.array([[0, 1], [1, 0]], dtype=complex))
    assert np.allclose(rule(ops.X), [[0, 1], [1, 0]])
    assert np.allclose(rule(ops.RX(1.23)), [[0, 1], [1, 0]])


@pytest.mark.parametrize("matrix,match", [
    (np.zeros((2, 3)), "square"),
    (np.eye(1), "side length"),
], ids=["non_square", "side_below_two"])
def test_fixed_matrix_rejects_invalid_shape(matrix, match):
    with pytest.raises(ValueError, match=match):
        FixedMatrix(matrix)


def test_fixed_matrix_accepts_non_power_of_two_side_length():
    # Deliberately not restricted to qubit-only power-of-two sizes: FixedMatrix
    # has no way to know what dimension its caller intends (e.g. a qutrit's
    # dim=3 gate), so it only requires squareness, not a power-of-two side.
    rule = FixedMatrix(np.eye(3, dtype=complex))
    assert np.allclose(rule(ops.X), np.eye(3))


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


# --- register: key normalization --------------------------------------------

def test_register_accepts_operation_instance_key():
    m = MatrixImplementationMap()
    rule = FixedMatrix(np.eye(2, dtype=complex))
    m.register(ops.X, rule)
    assert m.get(ops.X) is rule
    assert m.get(type(ops.X)) is rule


def test_register_accepts_operation_class_key():
    class MyGate(ops.Operation):
        name = "MyGate"
        _num_subsystems = 1

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


def _callable_rule(op):
    return np.eye(2, dtype=complex)


@pytest.mark.parametrize("rule", [
    FixedMatrix(np.eye(2, dtype=complex)),
    np.eye(2, dtype=complex),
    _callable_rule,
], ids=["fixed_matrix", "ndarray", "callable"])
def test_register_rejects_variable_arity_operation(rule):
    class VariableGate(ops.Operation):
        name = "VariableGate"
        _num_subsystems = None

    m = MatrixImplementationMap()
    with pytest.raises(TypeError, match="variable arity"):
        m.register(VariableGate, rule)


# --- unregister -------------------------------------------------------------

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


# --- register: rule wrapping (ndarray) --------------------------------------

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


# --- register: rule wrapping (callable) -------------------------------------

def _accepts_plain(gate):
    return np.eye(2, dtype=complex)


def _accepts_varargs(*args):
    return np.eye(2, dtype=complex)


def _accepts_optional_second(op, extra=None):
    return np.eye(2, dtype=complex)


@pytest.mark.parametrize("rule", [
    _accepts_plain,
    _accepts_varargs,
    _accepts_optional_second,
], ids=["plain", "varargs", "optional_second"])
def test_register_accepts_valid_callable_signatures(rule):
    m = MatrixImplementationMap()
    m.register(ops.RX, rule)
    wrapped = m.get(ops.RX)
    assert not isinstance(wrapped, FixedMatrix)  # wrapped, not stored as a matrix
    assert np.allclose(wrapped(ops.RX(0.3)), np.eye(2))


def _rejects_zero_args():
    return np.eye(2, dtype=complex)


def _rejects_two_required(op, extra):
    return np.eye(2, dtype=complex)


def _rejects_keyword_only(*, theta):
    return np.eye(2, dtype=complex)


@pytest.mark.parametrize("rule", [
    _rejects_zero_args,
    _rejects_two_required,
    _rejects_keyword_only,
], ids=["zero_args", "two_required", "keyword_only"])
def test_register_rejects_invalid_callable_signatures(rule):
    m = MatrixImplementationMap()
    with pytest.raises(TypeError, match="one positional argument"):
        m.register(ops.RX, rule)


def test_register_rejects_non_callable_non_ndarray_rule():
    m = MatrixImplementationMap()
    with pytest.raises(TypeError, match="MatrixImplementation, np.ndarray, or callable"):
        m.register(ops.X, "not a rule")


@pytest.mark.parametrize("exc", [ValueError, TypeError])
def test_register_accepts_callable_when_signature_is_uninspectable(monkeypatch, exc):
    import qnsim.implementation as implementation

    def raise_exc(_rule):
        raise exc("uninspectable signature")

    monkeypatch.setattr(implementation.inspect, "signature", raise_exc)

    m = MatrixImplementationMap()

    def some_rule(op):
        return np.eye(2, dtype=complex)

    m.register(ops.RX, some_rule)  # must not raise despite uninspectable signature
    assert np.allclose(m.get(ops.RX)(ops.RX(0.3)), np.eye(2))


# --- copy -------------------------------------------------------------------

def test_copy_is_independent_of_original():
    m = default_implementation_map()
    clone = m.copy()

    clone.unregister(ops.X)

    assert m.get(ops.X) is not None
    assert clone.get(ops.X) is None


def test_copy_preserves_existing_registrations():
    m = default_implementation_map()
    clone = m.copy()
    assert clone.get(ops.X) is m.get(ops.X)
