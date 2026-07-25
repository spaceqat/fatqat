"""Tests AppliedOperation and Measurement value object validation."""

from dataclasses import dataclass
from typing import ClassVar

import pytest

from fatqat.registers import GridRegister, QuantumRegister, ClassicalRegister
from fatqat import operations as ops
from fatqat.operations import Measurement
from fatqat.program import AppliedOperation


def test_applied_operation_accepts_correct_arity():
    qr = QuantumRegister(2)
    ao = AppliedOperation(operation=ops.CX, targets=(qr[0], qr[1]))
    assert ao.operation is ops.CX
    assert ao.targets == (qr[0], qr[1])
    assert ao.condition is None


def test_applied_operation_wrong_arity_raises():
    qr = QuantumRegister(2)
    with pytest.raises(ValueError):
        AppliedOperation(operation=ops.X, targets=(qr[0], qr[1]))  # X is 1-qubit
    with pytest.raises(ValueError):
        AppliedOperation(operation=ops.CX, targets=(qr[0],))  # CX is 2-qubit


def test_applied_operation_rejects_duplicate_targets():
    qr = QuantumRegister(1)
    with pytest.raises(ValueError, match="appears more than once"):
        AppliedOperation(operation=ops.CX, targets=(qr[0], qr[0]))


def test_measurement_fields():
    qr = QuantumRegister(1)
    cr = ClassicalRegister(1)
    m = Measurement(targets=(qr[0],), outputs=(cr[0],))
    assert m.targets == (qr[0],)
    assert m.outputs == (cr[0],)


def test_validate_targets_default_is_noop():
    qr = QuantumRegister(2)
    ao = AppliedOperation(operation=ops.CX, targets=(qr[0], qr[1]))
    assert ao.operation is ops.CX  # constructing did not raise


def test_validate_targets_hook_is_called_with_resolved_targets():
    @dataclass(frozen=True)
    class _Probe(ops.Operation):
        name: ClassVar[str] = "Probe"
        _num_subsystems: ClassVar[int] = 1

        def validate_targets(self, targets):
            raise ValueError(f"probe saw dim {targets[0].register.dim}")

    qr = QuantumRegister(1, dim=3)
    with pytest.raises(ValueError, match="probe saw dim 3"):
        AppliedOperation(operation=_Probe(), targets=(qr[0],))


# ---------------------------------------------------------------------------
# View-aware __post_init__ behavior
# ---------------------------------------------------------------------------


def test_applied_operation_accepts_view_for_view_capable_operation():
    atoms = GridRegister(2, 2, name="atoms")
    ao = AppliedOperation(operation=ops.RX(0.1), targets=(atoms.row(0),))
    assert ao.targets == (atoms.row(0),)


def test_applied_operation_rejects_view_for_scalar_only_operation():
    atoms = GridRegister(2, 2, name="atoms")
    with pytest.raises(ValueError):
        AppliedOperation(operation=ops.H, targets=(atoms.row(0),))


def test_applied_operation_view_arity_checked_before_scalar_validation():
    # CX/CZ expect exactly 2 target *expressions*; a view counts as one
    # expression regardless of how many members it selects.
    atoms = GridRegister(2, 2, name="atoms")
    with pytest.raises(ValueError):
        AppliedOperation(operation=ops.CX, targets=(atoms.row(0),))


def test_applied_operation_view_bearing_skips_validate_hook():
    atoms = GridRegister(2, 2, name="atoms")
    calls = []

    @dataclass(frozen=True)
    class _Probe(ops.Operation):
        name: ClassVar[str] = "ProbeCX"
        _num_subsystems: ClassVar[int] = 2
        _accepts_views: ClassVar[bool] = True

        def validate_targets(self, targets):
            calls.append(targets)
            raise AssertionError(
                "validate_targets must not run for view-bearing targets"
            )

    # Non-overlapping views (distinct rows): legal pairing, so only the
    # per-operation validate_targets() hook is at stake here - it must not
    # run for view-bearing targets, unlike the scalar path.
    ao = AppliedOperation(operation=_Probe(), targets=(atoms.row(0), atoms.row(1)))
    assert ao.targets == (atoms.row(0), atoms.row(1))
    assert calls == []


# ---------------------------------------------------------------------------
# View-pair legality (arity 2: matching selector kind, equal cardinality, no
# same-register overlap) - checked once here, at construction, not deferred
# to backend expansion. See registers._validate_view_pair.
# ---------------------------------------------------------------------------


def test_applied_operation_rejects_same_view_repeated():
    atoms = GridRegister(2, 2, name="atoms")
    with pytest.raises(ValueError, match="overlapping"):
        AppliedOperation(operation=ops.CX, targets=(atoms.row(0), atoms.row(0)))


def test_applied_operation_rejects_cross_selector_type_pairing():
    atoms = GridRegister(3, 3, name="atoms")
    # Equal cardinality (3 vs 3), but a row and a column are different
    # selector kinds - forbidden regardless of size match.
    with pytest.raises(ValueError, match="selector kind"):
        AppliedOperation(operation=ops.CX, targets=(atoms.row(0), atoms.column(0)))


def test_applied_operation_rejects_unequal_cardinality_across_registers():
    small = GridRegister(2, 3, name="small")
    large = GridRegister(2, 5, name="large")
    with pytest.raises(ValueError, match="cardinality"):
        AppliedOperation(operation=ops.CX, targets=(small.row(0), large.row(0)))


def test_applied_operation_accepts_disjoint_rows_same_register():
    atoms = GridRegister(2, 2, name="atoms")
    ao = AppliedOperation(operation=ops.CX, targets=(atoms.row(0), atoms.row(1)))
    assert ao.targets == (atoms.row(0), atoms.row(1))


def test_applied_operation_rejects_overlapping_blocks():
    atoms = GridRegister(2, 3, name="atoms")
    first = atoms.block(rows=(0, 2), cols=(0, 2))
    second = atoms.block(rows=(0, 2), cols=(1, 3))  # shares column 1
    with pytest.raises(ValueError, match="overlapping"):
        AppliedOperation(operation=ops.CX, targets=(first, second))


def test_applied_operation_accepts_non_overlapping_equal_size_blocks():
    atoms = GridRegister(2, 4, name="atoms")
    first = atoms.block(rows=(0, 2), cols=(0, 2))
    second = atoms.block(rows=(0, 2), cols=(2, 4))
    ao = AppliedOperation(operation=ops.CX, targets=(first, second))
    assert ao.targets == (first, second)


def test_applied_operation_rejects_all_paired_with_all():
    atoms = GridRegister(2, 2, name="atoms")
    with pytest.raises(ValueError, match="overlapping"):
        AppliedOperation(operation=ops.CX, targets=(atoms.all(), atoms.all()))


def test_zero_arity_operation_class_is_legal():
    @dataclass(frozen=True)
    class _ZeroArityProbe(ops.Operation):
        name: ClassVar[str] = "ZeroArityProbe"
        _num_subsystems: ClassVar[int] = 0

    ao = AppliedOperation(operation=_ZeroArityProbe(), targets=())
    assert ao.targets == ()


def test_zero_arity_operation_rejects_any_target():
    @dataclass(frozen=True)
    class _ZeroArityProbe2(ops.Operation):
        name: ClassVar[str] = "ZeroArityProbe2"
        _num_subsystems: ClassVar[int] = 0

    qr = QuantumRegister(1)
    with pytest.raises(ValueError, match="expects 0 target"):
        AppliedOperation(operation=_ZeroArityProbe2(), targets=(qr[0],))


def test_negative_arity_operation_class_rejected():
    with pytest.raises(ValueError, match="non-negative int or None"):

        @dataclass(frozen=True)
        class _NegativeArityProbe(ops.Operation):
            name: ClassVar[str] = "NegativeArityProbe"
            _num_subsystems: ClassVar[int] = -1
