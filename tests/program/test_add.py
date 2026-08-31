"""Tests Program.add operation insertion, target resolution, and validation."""

from dataclasses import dataclass
from typing import ClassVar

import pytest

from fatqat.program import Program, _AppliedOperation
import fatqat.operations as ops
from fatqat.registers import GridRegister, QuantumRegister, RegisterView


def test_add_single_operand_int():
    p = Program(2)
    result = p.add(ops.H, 0)
    assert result is None
    assert len(p._instructions) == 1
    ao = p._instructions[0]
    assert isinstance(ao, _AppliedOperation)
    assert ao.operation is ops.H
    assert ao.targets == (p.quantum_registers[0][0],)


def test_add_multi_operand_tuple():
    p = Program(2)
    p.add(ops.CZ, (0, 1))
    ao = p._instructions[0]
    assert ao.targets == (p.quantum_registers[0][0], p.quantum_registers[0][1])


def test_add_parametric_gate():
    p = Program(1)
    p.add(ops.RX(0.2), 0)
    assert p._instructions[0].operation.theta == 0.2


def test_add_rejects_variadic_positional():
    p = Program(2)
    with pytest.raises(TypeError):
        p.add(ops.CZ, 0, 1)  # variadic not supported


def test_add_wrong_arity_raises():
    p = Program(2)
    with pytest.raises(ValueError):
        p.add(ops.CZ, 0)  # CZ needs 2 targets


def test_add_rejects_non_operation():
    p = Program(1)
    with pytest.raises(TypeError):
        p.add(ops.RX, 0)  # passed the class, not an instance


def test_add_rejects_int_target_with_multiple_quantum_registers():
    qr0 = QuantumRegister(1, name="a")
    qr1 = QuantumRegister(1, name="b")
    p = Program([qr0, qr1])

    with pytest.raises(TypeError, match="explicit RegisterRef"):
        p.add(ops.X, 0)


def test_add_rejects_duplicate_qubit_targets():
    p = Program(2)
    with pytest.raises(ValueError, match="more than once"):
        p.add(ops.CZ, (0, 0))


def test_add_accepts_explicit_refs_across_multiple_quantum_registers():
    qr0 = QuantumRegister(1, name="a")
    qr1 = QuantumRegister(1, name="b")
    p = Program([qr0, qr1])

    p.add(ops.X, qr0[0])
    p.add(ops.X, qr1[0])

    assert p._instructions[0].targets == (qr0[0],)
    assert p._instructions[1].targets == (qr1[0],)


def test_add_swap_levels_out_of_range_raises():
    qr = QuantumRegister(1, dim=3)
    p = Program([qr])
    with pytest.raises(ValueError, match="0 <= j, k < dim"):
        p.add(ops.SwapLevels(0, 5), 0)


def test_add_swap_levels_in_range_succeeds():
    qr = QuantumRegister(1, dim=3)
    p = Program([qr])
    p.add(ops.SwapLevels(0, 2), 0)
    assert p._instructions[0].operation.j == 0


@pytest.mark.parametrize("op_cls", [ops.SubspaceRX, ops.SubspaceRY, ops.SubspaceRZ])
def test_add_subspace_rotation_out_of_range_raises(op_cls):
    qr = QuantumRegister(1, dim=3)
    p = Program([qr])
    with pytest.raises(ValueError, match="0 <= j, k < dim"):
        p.add(op_cls(0.3, (1, 3)), 0)


@pytest.mark.parametrize("op_cls", [ops.SubspaceRX, ops.SubspaceRY, ops.SubspaceRZ])
def test_add_subspace_rotation_in_range_succeeds(op_cls):
    qr = QuantumRegister(1, dim=3)
    p = Program([qr])
    p.add(op_cls(0.3, (1, 2)), 0)
    assert p._instructions[0].operation.subspace == (1, 2)


# ---------------------------------------------------------------------------
# RegisterView target acceptance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op",
    [
        ops.I,
        ops.H,
        ops.S,
        ops.Sdg,
        ops.SX,
        ops.T,
        ops.Tdg,
        ops.X,
        ops.Y,
        ops.Z,
        ops.RX(0.1),
        ops.RY(0.1),
        ops.RZ(0.1),
        ops.Phase(0.1),
        ops.U(0.1, 0.2, 0.3),
        ops.U1(0.1),
        ops.U2(0.1, 0.2),
        ops.U3(0.1, 0.2, 0.3),
        ops.Shift(1),
        ops.Clock(1),
        ops.SwapLevels(0, 2),
        ops.Fourier,
        ops.InverseFourier,
        ops.SubspaceRX(0.1, (0, 2)),
        ops.SubspaceRY(0.1, (0, 2)),
        ops.SubspaceRZ(0.1, (0, 2)),
    ],
)
def test_add_accepts_view_for_unary_gates(op):
    # Program.add is dimension-agnostic for ordinary matrix gates; dim=3 also
    # accommodates the level-selective gates in this shared contract table.
    qudits = GridRegister(2, 2, name="qudits", dim=3)
    p = Program([qudits])
    p.add(op, qudits.row(0))
    ao = p._instructions[0]
    assert ao.targets == (qudits.row(0),)


@pytest.mark.parametrize(
    "op",
    [
        ops.CX,
        ops.CZ,
        ops.Swap,
        ops.CY,
        ops.CS,
        ops.iSwap,
        ops.CPhase(0.1),
        ops.Sum,
        ops.CClock(1),
    ],
)
def test_add_accepts_view_pair_for_binary_gates(op):
    qudits = GridRegister(2, 2, name="qudits", dim=3)
    p = Program([qudits])
    p.add(op, (qudits.row(0), qudits.row(1)))
    ao = p._instructions[0]
    assert ao.targets == (qudits.row(0), qudits.row(1))


@pytest.mark.parametrize("op", [ops.CCX, ops.CSwap])
def test_add_accepts_views_for_ternary_gates(op):
    qubits = GridRegister(3, 2, name="qubits")
    p = Program([qubits])
    views = tuple(qubits.row(row) for row in range(3))
    p.add(op, views)
    assert p._instructions[0].targets == views


@pytest.mark.parametrize("op", [ops.Reset, ops.Barrier, ops.Put, ops.Pair, ops.Unpair])
def test_add_rejects_view_for_structural_operations(op):
    qubits = GridRegister(2, 2, name="qubits")
    p = Program([qubits])
    targets = (
        (qubits.row(0), qubits.row(1)) if op.num_subsystems == 2 else qubits.row(0)
    )
    with pytest.raises(ValueError):
        p.add(op, targets)


def test_add_view_runs_target_dependent_validation():
    qudits = GridRegister(1, 2, name="qudits", dim=3)
    p = Program([qudits])
    with pytest.raises(ValueError, match="invalid for target dimension 3"):
        p.add(ops.SwapLevels(0, 3), qudits.row(0))


def test_add_rejects_view_from_foreign_program():
    qubits = GridRegister(2, 2, name="qubits")
    p = Program([qubits])
    foreign_qubits = GridRegister(2, 2, name="foreign")  # not in p.quantum_registers
    with pytest.raises(ValueError):
        p.add(ops.RX(0.1), foreign_qubits.row(0))


def test_add_view_target_is_not_treated_as_scalar_ref():
    qubits = GridRegister(1, 2, name="qubits")
    p = Program([qubits])
    p.add(ops.RX(0.1), qubits.row(0))
    ao = p._instructions[0]
    assert isinstance(ao.targets[0], RegisterView)


def test_add_rejects_scalar_view_mixture_for_two_target_op():
    qubits = GridRegister(2, 2, name="qubits")
    p = Program([qubits])
    with pytest.raises(ValueError, match="mixes a scalar target with a view"):
        p.add(ops.CX, (qubits.row(1), qubits[0]))


def test_add_targets_optional_for_zero_arity_operation():
    @dataclass(frozen=True)
    class _ZeroArityProbe(ops.Operation):
        name: ClassVar[str] = "ZeroArityProbe"
        num_subsystems: ClassVar[int] = 0

    p = Program(1)
    p.add(_ZeroArityProbe())
    assert p._instructions[0].targets == ()


def test_add_accepts_list_targets():
    p = Program(2)
    p.add(ops.CZ, [0, 1])
    assert p._instructions[0].targets == (
        p.quantum_registers[0][0],
        p.quantum_registers[0][1],
    )


def test_measure_accepts_list_operands():
    p = Program(2, 2)
    p.measure([0, 1], [0, 1])
    m = p._instructions[0]
    assert m.targets == (p.quantum_registers[0][0], p.quantum_registers[0][1])
    assert m.outputs == (p.classical_registers[0][0], p.classical_registers[0][1])


def test_integer_operand_with_no_registers_names_the_real_fix():
    p = Program(0, 2)
    with pytest.raises(TypeError, match="has no quantum register"):
        p.add(ops.H, 0)
    p2 = Program(1, 0)
    with pytest.raises(TypeError, match="has no classical register"):
        p2.add(ops.X, 0, condition=(0, 1))
