"""Tests for scalar qubit-resource mapping and grouped-operation expansion."""

import pytest

from fatqat import operations as ops
from fatqat.backends import ApplyMatrixStep, SimulatorBackend
from fatqat.backends.fake_atom_grid import FakeAtomGridBackend
from fatqat.backends.resource_binding import BoundResource
from fatqat.backends.simulator_backend import _break_grouped_operations
from fatqat.errors import BackendValidationError
from fatqat.implementation import ImplementationMap, default_matrix_implementation_map
from fatqat.program import Program
from fatqat.registers import GridRegister


def _prepared(backend, program):
    layout = backend.resolve_layout(program)
    resources = backend._build_qubit_resource_map(program, layout)
    operations = _break_grouped_operations(program.operations)
    return operations, layout, resources


def _matrix_steps(plan):
    return [step for step in plan if isinstance(step, ApplyMatrixStep)]


def test_bound_resource_is_immutable():
    program = Program(1)
    bound = BoundResource(ref=program.qreg[0][0], engine_index=0, device_label=0)
    with pytest.raises(AttributeError):
        bound.engine_index = 1


def test_identity_resource_map_contains_flat_index_and_device_label():
    program = Program(2)
    backend = SimulatorBackend()
    layout = backend.resolve_layout(program)
    resources = backend._build_qubit_resource_map(program, layout)
    ref = program.qreg[0][1]
    assert resources[ref] == BoundResource(ref=ref, engine_index=1, device_label=1)


def test_fake_grid_resource_map_keeps_flat_and_hardware_indices_distinct():
    atoms = GridRegister(2, 3, name="atoms")
    program = Program([atoms])
    backend = FakeAtomGridBackend(rows=4, cols=5)
    layout = backend.resolve_layout(program)
    resources = backend._build_qubit_resource_map(program, layout)
    assert tuple(resources[atoms[index]].engine_index for index in range(6)) == tuple(
        range(6)
    )
    assert tuple(resources[atoms[index]].device_label for index in range(6)) == (
        0,
        1,
        2,
        5,
        6,
        7,
    )


def test_scalar_only_and_measurement_instructions_pass_through_unchanged():
    program = Program(2, 1)
    program.add(ops.RX(0.3), 1)
    program.add(ops.CZ, (0, 1))
    program.add_measurement(0, 0)
    broken = _break_grouped_operations(program.operations)
    assert broken == program.operations
    assert all(left is right for left, right in zip(broken, program.operations))


@pytest.mark.parametrize(
    ("selector_name", "expected_indices"),
    [("row", (0, 1, 2)), ("column", (0, 3))],
)
def test_grouped_rotation_expands_in_view_order_and_preserves_operation_data(
    selector_name, expected_indices
):
    atoms = GridRegister(2, 3, name="atoms")
    program = Program([atoms], 1)
    operation = ops.RX(0.3)
    program.add(operation, getattr(atoms, selector_name)(0), condition=(0, 1))
    broken = _break_grouped_operations(program.operations)
    assert [step.targets for step in broken] == [
        (atoms[index],) for index in expected_indices
    ]
    assert all(step.operation is operation for step in broken)
    assert all(step.condition == program.operations[0].condition for step in broken)


def test_grouped_two_target_operation_zips_views_in_order():
    atoms = GridRegister(2, 2, name="atoms")
    program = Program([atoms])
    program.add(ops.CX, (atoms.row(0), atoms.row(1)))
    broken = _break_grouped_operations(program.operations)
    assert [step.targets for step in broken] == [
        (atoms[0], atoms[2]),
        (atoms[1], atoms[3]),
    ]


def test_grouped_operation_rejects_unequal_view_cardinality():
    atoms = GridRegister(2, 3, name="atoms")
    program = Program([atoms])
    program.add(ops.CX, (atoms.row(0), atoms.column(0)))
    with pytest.raises(BackendValidationError):
        _break_grouped_operations(program.operations)


def test_grouped_operation_rejects_scalar_view_mixture():
    atoms = GridRegister(2, 2, name="atoms")
    program = Program([atoms])
    program.add(ops.CX, (atoms.row(1), atoms[0]))
    with pytest.raises(BackendValidationError):
        _break_grouped_operations(program.operations)


def test_grouped_operation_rejects_self_pair():
    atoms = GridRegister(2, 2, name="atoms")
    program = Program([atoms])
    program.add(ops.CZ, (atoms.row(0), atoms.column(0)))
    with pytest.raises(BackendValidationError):
        _break_grouped_operations(program.operations)


def test_grouped_operation_does_not_mutate_program():
    atoms = GridRegister(2, 2, name="atoms")
    program = Program([atoms])
    program.add(ops.RX(0.3), atoms.row(0))
    before = program.operations
    broken = _break_grouped_operations(program.operations)
    assert program.operations is before
    assert list(program.operations) != list(broken)


def test_base_simulator_executes_grouped_views_with_identity_mapping():
    atoms = GridRegister(2, 2, name="atoms")
    program = Program([atoms])
    program.add(ops.RX(0.3), atoms.row(0))
    result = (
        SimulatorBackend()
        .run(
            program,
            result_config={"counts": False, "statevector": True},
        )
        .result()
    )
    assert result.get_statevector() is not None


def test_lower_uses_device_labels_for_lookup_and_flat_indices_for_steps():
    program = Program(2)
    program.add(ops.CZ, (0, 1))
    backend = SimulatorBackend()
    layout = backend.resolve_layout(program)
    q0, q1 = program.qreg[0][0], program.qreg[0][1]
    resources = {
        q0: BoundResource(ref=q0, engine_index=0, device_label=99),
        q1: BoundResource(ref=q1, engine_index=1, device_label=100),
    }
    cz_rule = default_matrix_implementation_map().implementation_for(ops.CZ)
    implementation_map = ImplementationMap()
    implementation_map.add(ops.CZ, cz_rule, device_operands=(99, 100))
    backend = SimulatorBackend(implementation_map=implementation_map)
    operations = _break_grouped_operations(program.operations)
    plan, _facts = backend._lower(operations, layout, resources)
    assert _matrix_steps(plan)[0].target_indices == (0, 1)


def test_lower_scalar_rotation_emits_one_step():
    program = Program(2)
    program.add(ops.RX(0.3), 1)
    plan, _facts = SimulatorBackend()._lower_program(program)
    assert [step.target_indices for step in _matrix_steps(plan)] == [(1,)]
