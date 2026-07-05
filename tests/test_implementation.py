"""Tests matrix implementation rules and immutable matrix payloads."""

import numpy as np
import pytest

from qnsim import operations as ops
from qnsim.implementation import (
    ApplyMatrixStep,
    _DimMatrix,
    FixedMatrix,
    MatrixImplementation,
    MatrixImplementationMap,
    clock_matrix,
    default_implementation_map,
    sum_matrix,
)
from qnsim.registers import QuantumRegister


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
    rx = m.get(ops.RX)(ops.RX(theta), targets=())
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    assert np.allclose(rx, [[c, -1j * s], [-1j * s, c]])


def test_parametric_phase_reads_theta():
    m = default_implementation_map()
    theta = 0.9
    phase = m.get(ops.Phase)(ops.Phase(theta), targets=())
    assert np.allclose(phase, [[1, 0], [0, np.exp(1j * theta)]])


def test_parametric_cphase_reads_theta():
    m = default_implementation_map()
    theta = 1.1
    cphase = m.get(ops.CPhase)(ops.CPhase(theta), targets=())
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


def test_register_accepts_ndarray_of_any_square_shape_for_operation():
    # The registry no longer cross-checks a bare ndarray's shape against the
    # operation's arity (e.g. 2**n for an n-subsystem gate): a rule has no
    # way to know the operation's intended dimension in general (a custom
    # qutrit gate registered against a 1-subsystem op class is legitimate),
    # so only squareness (>= 2x2) is validated, in `FixedMatrix`.
    m = MatrixImplementationMap()
    m.register(ops.X, np.eye(4, dtype=complex))
    assert m.get(ops.X)(ops.X, targets=()).shape == (4, 4)


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
    assert np.allclose(wrapped(ops.RX(0.3), targets=()), np.eye(2))


def _wrong_shape_zero_args():
    return np.eye(2, dtype=complex)


def test_wrong_shape_callable_fails_at_use_not_registration():
    # register() no longer arity-checks callables; a wrong-shape rule registers
    # cleanly and instead fails the first time it is invoked (where the backend
    # wraps it in a MatrixImplementationError). Here we invoke the wrapped rule
    # directly and expect the raw call-time TypeError.
    m = MatrixImplementationMap()
    m.register(ops.RX, _wrong_shape_zero_args)  # accepted at registration
    with pytest.raises(TypeError):
        m.get(ops.RX)(ops.RX(0.3), targets=())


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
    assert np.allclose(m.get(ops.RX)(ops.RX(0.3), targets=()), np.eye(2))


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


# --- widened contract: rule(op, *, targets=...) -----------------------------

def _targets(dim, n=1):
    reg = QuantumRegister(n, dim=dim)
    return tuple(reg[i] for i in range(n))


def test_fixed_matrix_ignores_targets():
    fm = FixedMatrix(np.eye(2, dtype=complex))
    out = fm(ops.X, targets=_targets(2))
    assert np.allclose(out, np.eye(2))


def test_bare_callable_op_only_still_works():
    m = MatrixImplementationMap()
    m.register(ops.RX, lambda op: np.eye(2, dtype=complex))
    rule = m.get(ops.RX(0.1))
    assert np.allclose(rule(ops.RX(0.1), targets=_targets(2)), np.eye(2))


def test_targets_aware_callable_receives_refs():
    seen = {}

    def rule(op, targets):
        seen["dim"] = targets[0].register.dim
        return np.eye(targets[0].register.dim, dtype=complex)

    m = MatrixImplementationMap()
    m.register(ops.X, rule)
    out = m.get(ops.X)(ops.X, targets=_targets(3))
    assert seen["dim"] == 3
    assert out.shape == (3, 3)


def test_dimensioned_matrix_derives_dims():
    dm = _DimMatrix(lambda dims: np.eye(dims[0], dtype=complex))
    out = dm(ops.X, targets=_targets(3))
    assert out.shape == (3, 3)


def test_bare_ndarray_non_power_of_two_registers():
    m = MatrixImplementationMap()
    m.register(ops.X, np.eye(3, dtype=complex))  # 3x3 for a custom qutrit rule
    assert m.get(ops.X)(ops.X, targets=_targets(3)).shape == (3, 3)


def test_non_square_ndarray_still_rejected():
    m = MatrixImplementationMap()
    with pytest.raises(ValueError):
        m.register(ops.X, np.ones((2, 3), dtype=complex))


# --- Shift/Clock/Sum: dimension-generic gates -------------------------------


def _qutrit_targets(n):
    reg = QuantumRegister(n, dim=3)
    return tuple(reg[i] for i in range(n))


def test_clock_matrix_qutrit_phases():
    c = clock_matrix(3, 1)
    omega = np.exp(2j * np.pi / 3)
    assert np.allclose(np.diag(c), [1, omega, omega**2])


def test_sum_matrix_controlled_add_qutrits():
    s = sum_matrix((3, 3))
    # |i,j> -> |i, (i+j) mod 3>; local index = i*3 + j (operand0 = control = MSB).
    # |2,2> (index 8) -> |2, (2+2)%3=1> (index 2*3+1 = 7).
    vec = np.zeros(9, dtype=complex)
    vec[8] = 1.0
    out = s @ vec
    assert np.argmax(np.abs(out)) == 7


def test_sum_matrix_mismatched_dims_raises():
    with pytest.raises(ValueError):
        sum_matrix((3, 2))


def test_default_map_has_new_gates():
    m = default_implementation_map()
    assert m.get(ops.Shift(1))(ops.Shift(1), targets=_qutrit_targets(1)).shape == (3, 3)
    assert m.get(ops.Clock(1))(ops.Clock(1), targets=_qutrit_targets(1)).shape == (3, 3)
    assert m.get(ops.Sum)(ops.Sum, targets=_qutrit_targets(2)).shape == (9, 9)


def test_shift_reduces_to_x_at_dim2():
    m = default_implementation_map()
    qb = QuantumRegister(1, dim=2)
    got = m.get(ops.Shift(1))(ops.Shift(1), targets=(qb[0],))
    assert np.allclose(got, np.array([[0, 1], [1, 0]], dtype=complex))
