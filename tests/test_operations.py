"""Tests operation objects, gate metadata, and immutability."""

import pytest

from qnsim import operations as ops
from qnsim.operations import Operation


@pytest.mark.parametrize("gate,name,n_subsystems", [
    (ops.I,  "I",  1),
    (ops.H,  "H",  1),
    (ops.S,  "S",  1),
    (ops.Sdg, "Sdg", 1),
    (ops.X,  "X",  1),
    (ops.Y,  "Y",  1),
    (ops.Z,  "Z",  1),
    (ops.T,  "T",  1),
    (ops.Tdg, "Tdg", 1),
    (ops.CX, "CX", 2),
    (ops.CZ, "CZ", 2),
    (ops.Swap, "Swap", 2),
    (ops.CY, "CY", 2),
    (ops.CS, "CS", 2),
    (ops.iSwap, "iSwap", 2),
    (ops.CCX, "CCX", 3),
    (ops.CSwap, "CSwap", 3),
])
def test_fixed_gate_name_and_arity(gate, name, n_subsystems):
    assert gate.name == name
    assert gate.num_subsystems == n_subsystems


def test_parametric_gate_is_class_storing_theta():
    g = ops.RX(0.2)
    assert isinstance(g, Operation)
    assert g.name == "RX"
    assert g.theta == 0.2
    assert g.num_subsystems == 1
    assert ops.RY(0.3).name == "RY"
    assert ops.RZ(0.4).name == "RZ"


def test_gates_distinguished_by_class():
    assert type(ops.X) is not type(ops.H)
    assert isinstance(ops.X, Operation)


def test_phase_gate_is_class_storing_theta():
    g = ops.Phase(0.7)
    assert isinstance(g, Operation)
    assert g.name == "Phase"
    assert g.theta == 0.7
    assert g.num_subsystems == 1


def test_operations_are_frozen():
    with pytest.raises(Exception):
        ops.RX(0.1).theta = 9.0


def test_cphase_gate_is_class_storing_theta():
    g = ops.CPhase(0.4)
    assert isinstance(g, Operation)
    assert g.name == "CPhase"
    assert g.theta == 0.4
    assert g.num_subsystems == 2


def test_shift_clock_are_single_subsystem_parametric():
    assert ops.Shift(1)._num_subsystems == 1
    assert ops.Clock(2)._num_subsystems == 1
    assert ops.Shift(1).power == 1


def test_sum_is_two_subsystem_singleton():
    assert ops.Sum._num_subsystems == 2
    assert isinstance(ops.Sum, ops.SumGate)


def test_new_gates_carry_no_dim_field():
    # No dim attribute on the symbols.
    assert not hasattr(ops.Shift(1), "dim")
    assert not hasattr(ops.Sum, "dim")


def test_swap_levels_is_single_subsystem_parametric():
    g = ops.SwapLevels(0, 2)
    assert g.name == "SwapLevels"
    assert g.num_subsystems == 1
    assert g.j == 0
    assert g.k == 2


def test_swap_levels_rejects_equal_indices():
    with pytest.raises(ValueError, match="j != k"):
        ops.SwapLevels(1, 1)


def test_swap_levels_rejects_negative_indices():
    with pytest.raises(ValueError, match="non-negative"):
        ops.SwapLevels(-1, 0)


def test_fourier_is_single_subsystem_singleton():
    assert ops.Fourier._num_subsystems == 1
    assert ops.Fourier.name == "Fourier"
    assert isinstance(ops.Fourier, Operation)
    assert not isinstance(ops.Fourier, type)
    assert ops.Fourierdg._num_subsystems == 1
    assert ops.Fourierdg.name == "Fourierdg"
    assert isinstance(ops.Fourierdg, Operation)
    assert not isinstance(ops.Fourierdg, type)


def test_fourier_gate_classes_are_not_public():
    # Unlike SumGate (attribute-accessible via qs.ops.SumGate, excluded only
    # from __all__), FourierGate/FourierdgGate are internal-only: they exist
    # in operations/qudit_gates.py to build the singletons and to key the
    # matrix registry, but are never imported into operations/__init__.py, so
    # qs.ops has no FourierGate/FourierdgGate attribute at all.
    assert not hasattr(ops, "FourierGate")
    assert not hasattr(ops, "FourierdgGate")


@pytest.mark.parametrize("cls,name", [
    (ops.SubspaceRX, "SubspaceRX"),
    (ops.SubspaceRY, "SubspaceRY"),
    (ops.SubspaceRZ, "SubspaceRZ"),
])
def test_subspace_rotation_is_single_subsystem_parametric(cls, name):
    g = cls(0.3, (0, 2))
    assert g.name == name
    assert g.num_subsystems == 1
    assert g.theta == 0.3
    assert g.subspace == (0, 2)


@pytest.mark.parametrize("cls", [ops.SubspaceRX, ops.SubspaceRY, ops.SubspaceRZ])
def test_subspace_rotation_rejects_equal_subspace_indices(cls):
    with pytest.raises(ValueError, match="distinct"):
        cls(0.1, (1, 1))


@pytest.mark.parametrize("cls", [ops.SubspaceRX, ops.SubspaceRY, ops.SubspaceRZ])
def test_subspace_rotation_rejects_negative_subspace_indices(cls):
    with pytest.raises(ValueError, match="non-negative"):
        cls(0.1, (-1, 0))


def test_cclock_is_two_subsystem_parametric():
    g = ops.CClock(1)
    assert g.name == "CClock"
    assert g.num_subsystems == 2
    assert g.power == 1


def test_cclock_carries_no_dim_field():
    assert not hasattr(ops.CClock(1), "dim")
