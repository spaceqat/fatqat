"""Tests for immutable program parameter binding."""

from dataclasses import dataclass
from fractions import Fraction
from typing import ClassVar

import numpy as np
import pytest

import fatqat as fq
from fatqat._parameter_binding import _raise_for_unbound_parameters
from fatqat.operations import Operation


def test_binds_one_parameter_without_mutating_template():
    theta = fq.Parameter("theta")
    program = fq.Program(1)
    program.add(fq.ops.RX(theta), 0)

    bound = program.assign_parameters({theta: 0.25})

    assert program.operations[0].operation.theta is theta
    assert bound.operations[0].operation.theta == 0.25
    assert bound is not program


@pytest.mark.parametrize("value", [1, 0.5, np.int64(1), np.float64(0.5)])
def test_binds_shared_parameter_in_every_direct_field(value):
    theta = fq.Parameter("theta")
    program = fq.Program(2)
    program.add(fq.ops.RX(theta), 0)
    program.add(fq.ops.RY(theta), 1)

    bound = program.assign_parameters({theta: value})

    assert bound.operations[0].operation.theta == value
    assert bound.operations[1].operation.theta == value


def test_binds_complete_vector_and_mixed_individual_parameter():
    angles = fq.ParameterVector("angles", 2)
    bias = fq.Parameter("bias")
    program = fq.Program(3)
    program.add(fq.ops.RX(angles[0]), 0)
    program.add(fq.ops.RY(angles[1]), 1)
    program.add(fq.ops.RZ(bias), 2)

    bound = program.assign_parameters({bias: 0.3, angles: [0.1, 0.2]})

    assert [instruction.operation.theta for instruction in bound.operations] == [
        0.1,
        0.2,
        0.3,
    ]


def test_partial_binding_leaves_other_parameters_intact():
    first = fq.Parameter("first")
    second = fq.Parameter("second")
    program = fq.Program(2)
    program.add(fq.ops.RX(first), 0)
    program.add(fq.ops.RY(second), 1)

    bound = program.assign_parameters({first: 0.1})

    assert bound.operations[0].operation.theta == 0.1
    assert bound.operations[1].operation.theta is second


def test_empty_binding_returns_independent_program_and_metadata():
    program = fq.Program(1, metadata={"owner": "template"})
    program.add(fq.ops.H, 0)

    bound = program.assign_parameters({})
    bound.metadata["owner"] = "bound"

    assert bound is not program
    assert bound.operations == program.operations
    assert bound.operations[0] is program.operations[0]
    assert program.metadata == {"owner": "template"}


def test_binding_preserves_program_structure_and_view_targets():
    atoms = fq.GridRegister(2, 2, name="atoms")
    classical = fq.ClassicalRegister(1, name="c")
    theta = fq.Parameter("theta")
    program = fq.Program([atoms], [classical], metadata={"kind": "view"})
    program.add(fq.ops.RX(theta), atoms.row(0), condition=(classical[0], 1))
    program.measure(atoms[2], classical[0])

    bound = program.assign_parameters({theta: 0.4})

    original_gate, original_measurement = program.operations
    bound_gate, bound_measurement = bound.operations
    assert bound.quantum_registers[0] is atoms
    assert bound.classical_registers[0] is classical
    assert bound_gate is not original_gate
    assert bound_gate.targets[0] is original_gate.targets[0]
    assert bound_gate.condition is original_gate.condition
    assert bound_measurement is original_measurement
    assert bound.metadata == program.metadata
    assert bound.metadata is not program.metadata


def test_generic_dataclass_operation_binding_preserves_other_fields():
    @dataclass(frozen=True)
    class CustomRotation(Operation):
        theta: float | fq.Parameter
        label: str
        name: ClassVar[str] = "CustomRotation"

    theta = fq.Parameter("theta")
    program = fq.Program(1)
    program.add(CustomRotation(theta, "kept"), 0)

    bound = program.assign_parameters({theta: 0.75})

    assert bound.operations[0].operation == CustomRotation(0.75, "kept")


def test_custom_operation_post_init_error_propagates_unchanged():
    @dataclass(frozen=True)
    class NonNegativeRotation(Operation):
        theta: float | fq.Parameter
        name: ClassVar[str] = "NonNegativeRotation"

        def __post_init__(self):
            if not isinstance(self.theta, fq.Parameter) and self.theta < 0:
                raise RuntimeError("custom negative angle")

    theta = fq.Parameter("theta")
    program = fq.Program(1)
    program.add(NonNegativeRotation(theta), 0)

    with pytest.raises(RuntimeError, match="custom negative angle"):
        program.assign_parameters({theta: -0.1})


@pytest.mark.parametrize("values", [None, [], [("theta", 0.1)]])
def test_binding_requires_mapping(values):
    program = fq.Program(1)

    with pytest.raises(TypeError, match="must be a mapping"):
        program.assign_parameters(values)


def test_binding_rejects_invalid_and_foreign_keys():
    theta = fq.Parameter("theta")
    program = fq.Program(1)
    program.add(fq.ops.RX(theta), 0)

    with pytest.raises(TypeError, match="keys must be"):
        program.assign_parameters({"theta": 0.1})
    with pytest.raises(ValueError, match="not present"):
        program.assign_parameters({fq.Parameter("theta"): 0.1})


def test_vector_key_requires_every_element_in_program():
    angles = fq.ParameterVector("angles", 2)
    program = fq.Program(1)
    program.add(fq.ops.RX(angles[0]), 0)

    with pytest.raises(ValueError, match="not fully present"):
        program.assign_parameters({angles: [0.1, 0.2]})


def test_zero_length_vector_cannot_be_bound():
    program = fq.Program(1)

    with pytest.raises(ValueError, match="zero-length"):
        program.assign_parameters({fq.ParameterVector("empty", 0): []})


def test_duplicate_vector_and_element_assignment_is_rejected():
    angles = fq.ParameterVector("angles", 2)
    program = fq.Program(2)
    program.add(fq.ops.RX(angles[0]), 0)
    program.add(fq.ops.RY(angles[1]), 1)

    with pytest.raises(ValueError, match="assigned more than once"):
        program.assign_parameters({angles: [0.1, 0.2], angles[0]: 0.3})


@pytest.mark.parametrize(
    "value", [True, 1 + 2j, "0.1", [0.1], np.bool_(True), Fraction(1, 3)]
)
def test_binding_rejects_invalid_scalar_values(value):
    theta = fq.Parameter("theta")
    program = fq.Program(1)
    program.add(fq.ops.RX(theta), 0)

    with pytest.raises(TypeError, match="real scalars"):
        program.assign_parameters({theta: value})


@pytest.mark.parametrize("value", ["values", b"values", {0: 0.1}, 0.1])
def test_binding_rejects_invalid_vector_containers(value):
    angles = fq.ParameterVector("angles", 2)
    program = fq.Program(2)
    program.add(fq.ops.RX(angles[0]), 0)
    program.add(fq.ops.RY(angles[1]), 1)

    with pytest.raises(TypeError, match="one-dimensional sequences"):
        program.assign_parameters({angles: value})


@pytest.mark.parametrize(
    "value",
    [np.array(0.1), [[0.1, 0.2]], [[0.1], [0.2]]],
)
def test_binding_rejects_vector_rank_errors(value):
    angles = fq.ParameterVector("angles", 2)
    program = fq.Program(2)
    program.add(fq.ops.RX(angles[0]), 0)
    program.add(fq.ops.RY(angles[1]), 1)

    with pytest.raises(ValueError, match="one-dimensional"):
        program.assign_parameters({angles: value})


def test_binding_rejects_nonrectangular_vector_container():
    angles = fq.ParameterVector("angles", 2)
    program = fq.Program(2)
    program.add(fq.ops.RX(angles[0]), 0)
    program.add(fq.ops.RY(angles[1]), 1)

    with pytest.raises(ValueError, match="rectangular container"):
        program.assign_parameters({angles: [np.zeros((2, 2)), np.zeros((2, 3))]})


def test_binding_rejects_wrong_vector_length_and_bad_element():
    angles = fq.ParameterVector("angles", 2)
    program = fq.Program(2)
    program.add(fq.ops.RX(angles[0]), 0)
    program.add(fq.ops.RY(angles[1]), 1)

    with pytest.raises(ValueError, match="expects 2 values"):
        program.assign_parameters({angles: [0.1]})
    with pytest.raises(TypeError, match="real scalars"):
        program.assign_parameters({angles: [0.1, "bad"]})


def test_unbound_diagnostic_uses_stable_order_and_deduplicates_identity():
    shared = fq.Parameter("shared")
    first = fq.Parameter("theta")
    second = fq.Parameter("theta")
    program = fq.Program(4)
    program.add(fq.ops.RX(shared), 0)
    program.add(fq.ops.RY(first), 1)
    program.add(fq.ops.RZ(shared), 2)
    program.add(fq.ops.Phase(second), 3)

    with pytest.raises(
        fq.errors.BackendValidationError,
        match=r"^program has unbound parameters: shared, theta#1, theta#2$",
    ):
        _raise_for_unbound_parameters(program.operations)
