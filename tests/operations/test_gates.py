"""Tests operation objects, gate metadata, and immutability."""

import pytest

import fatqat.operations as ops
from fatqat import Parameter
from fatqat.operations import Operation


@pytest.mark.parametrize(
    "gate,name,n_subsystems",
    [
        (ops.I, "I", 1),
        (ops.H, "H", 1),
        (ops.S, "S", 1),
        (ops.Sdg, "Sdg", 1),
        (ops.SX, "SX", 1),
        (ops.X, "X", 1),
        (ops.Y, "Y", 1),
        (ops.Z, "Z", 1),
        (ops.T, "T", 1),
        (ops.Tdg, "Tdg", 1),
        (ops.CX, "CX", 2),
        (ops.CZ, "CZ", 2),
        (ops.Swap, "Swap", 2),
        (ops.CY, "CY", 2),
        (ops.CS, "CS", 2),
        (ops.iSwap, "iSwap", 2),
        (ops.CCX, "CCX", 3),
        (ops.CSwap, "CSwap", 3),
    ],
)
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


@pytest.mark.parametrize(
    "gate_factory",
    [
        ops.RX,
        ops.RY,
        ops.RZ,
        ops.Phase,
        ops.CPhase,
        lambda theta: ops.SubspaceRX(theta, (0, 1)),
        lambda theta: ops.SubspaceRY(theta, (0, 1)),
        lambda theta: ops.SubspaceRZ(theta, (0, 1)),
    ],
)
def test_angle_gates_accept_parameter_values(gate_factory):
    theta = Parameter("theta")

    assert gate_factory(theta).theta is theta


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
    assert ops.Shift(1).num_subsystems == 1
    assert ops.Clock(2).num_subsystems == 1
    assert ops.Shift(1).power == 1


def test_sum_is_two_subsystem_singleton():
    assert ops.Sum.num_subsystems == 2
    assert ops.Sum.name == "Sum"
    assert isinstance(ops.Sum, Operation)
    assert not isinstance(ops.Sum, type)


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
    assert ops.Fourier.num_subsystems == 1
    assert ops.Fourier.name == "Fourier"
    assert isinstance(ops.Fourier, Operation)
    assert not isinstance(ops.Fourier, type)
    assert ops.InverseFourier.num_subsystems == 1
    assert ops.InverseFourier.name == "InverseFourier"
    assert isinstance(ops.InverseFourier, Operation)
    assert not isinstance(ops.InverseFourier, type)
    assert type(ops.InverseFourier).__name__ == "InverseFourierGate"


@pytest.mark.parametrize(
    "cls,name",
    [
        (ops.SubspaceRX, "SubspaceRX"),
        (ops.SubspaceRY, "SubspaceRY"),
        (ops.SubspaceRZ, "SubspaceRZ"),
    ],
)
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


def test_num_subsystems_is_the_only_public_operation_width_name():
    class CustomTwoSubsystem(Operation):
        num_subsystems = 2

    assert CustomTwoSubsystem.num_subsystems == 2
    assert CustomTwoSubsystem().num_subsystems == 2
    assert not hasattr(Operation, "arity")
    assert not hasattr(ops.X, "arity")
    assert not hasattr(Operation, "num_targets")
    assert not hasattr(ops.X, "num_targets")


@pytest.mark.parametrize("retired_name", ["num_targets", "_num_subsystems"])
def test_custom_operation_rejects_retired_width_declarations(retired_name):
    with pytest.raises(TypeError, match="num_subsystems"):
        type("LegacySubsystemCount", (Operation,), {retired_name: 2})


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_custom_operation_rejects_invalid_num_subsystems(bad):
    with pytest.raises(ValueError, match="num_subsystems"):
        type("BadSubsystemCount", (Operation,), {"num_subsystems": bad})


class TestAngleValidationAtConstruction:
    """Angles follow the assign_parameters scalar policy from construction."""

    def test_scalars_and_parameters_accepted(self):
        import numpy as np

        import fatqat as fq

        ops.RX(1)
        ops.RX(0.5)
        ops.RX(np.float64(0.5))
        ops.RX(np.int32(2))
        ops.RX(fq.Parameter("t"))
        ops.U(0.1, 0.2, 0.3)
        ops.SubspaceRY(0.4, (0, 1))

    @pytest.mark.parametrize("bad", [True, "a", 1 + 2j, None, [0.1]])
    def test_invalid_angle_types_rejected(self, bad):
        with pytest.raises(TypeError, match="must be a real number"):
            ops.RX(bad)
        with pytest.raises(TypeError, match="must be a real number"):
            ops.U(0.1, bad, 0.3)
        with pytest.raises(TypeError, match="must be a real number"):
            ops.CPhase(bad)
        with pytest.raises(TypeError, match="must be a real number"):
            ops.SubspaceRZ(bad, (0, 1))

    def test_whole_parameter_vector_rejected(self):
        import fatqat as fq

        vec = fq.ParameterVector("a", 2)
        with pytest.raises(TypeError, match="must be a real number"):
            ops.RX(vec)
        ops.RX(vec[0])
