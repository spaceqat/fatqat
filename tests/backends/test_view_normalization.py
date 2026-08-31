"""Tests for grouped-view/operation expansion and the private lowering context.

`_break_grouped_operations()` is the temporary scalar view-normalization
stage: it must run before scalar lowering and must not carry any resource-
mapping policy of its own (that lives in `ResourceLayout`/`_EngineAllocation`
and the private `_LoweringContext` that pairs them for one run's lowering).

Pairing legality (matching selector kind, equal cardinality, no same-register
overlap) is validated at `Program.add()`/`_AppliedOperation` construction time
(see tests/program/test_add.py and tests/program/test_applied_operation.py),
not here - by the time a step reaches expansion, its targets are already
guaranteed legal.
"""

import pytest

import fatqat.operations as ops
from fatqat._backends.steps import ApplyMatrixStep
from fatqat.simulator import Simulator
from fatqat._backends.backend_utils import _LoweringContext
from fatqat._index_allocation import _ClassicalAllocation
from fatqat.simulator.simulator import _break_grouped_operations
from fatqat.implementation import (
    MatrixImplementationMap,
    default_matrix_implementation_map,
)
from fatqat.program import Program
from fatqat.registers import GridRegister, QuantumRegister
from fatqat.resource_layout import ResourceLayout


def _matrix_steps(plan):
    return [step for step in plan if isinstance(step, ApplyMatrixStep)]


def test_scalar_only_and_measurement_instructions_pass_through_unchanged():
    program = Program(2, 1)
    program.add(ops.RX(0.3), 1)
    program.add(ops.CZ, (0, 1))
    program.measure(0, 0)
    broken = _break_grouped_operations(program._instructions)
    assert broken == program._instructions
    assert all(left is right for left, right in zip(broken, program._instructions))


@pytest.mark.parametrize(
    ("selector_name", "expected_indices"),
    [("row", (0, 1, 2)), ("column", (0, 3))],
)
def test_grouped_rotation_expands_in_view_order_and_preserves_operation_data(
    selector_name, expected_indices
):
    qubits = GridRegister(2, 3, name="qubits")
    program = Program([qubits], 1)
    operation = ops.RX(0.3)
    program.add(operation, getattr(qubits, selector_name)(0), condition=(0, 1))
    broken = _break_grouped_operations(program._instructions)
    assert [step.targets for step in broken] == [
        (qubits[index],) for index in expected_indices
    ]
    assert all(step.operation is operation for step in broken)
    assert all(step.condition == program._instructions[0].condition for step in broken)


def test_flat_register_all_expands_in_index_order():
    qubits = QuantumRegister(3)
    program = Program([qubits])
    program.add(ops.X, qubits.all())
    broken = _break_grouped_operations(program._instructions)
    assert [step.targets for step in broken] == [(qubits[index],) for index in range(3)]


def test_grouped_two_target_operation_zips_views_in_order():
    qubits = GridRegister(2, 2, name="qubits")
    program = Program([qubits])
    program.add(ops.CX, (qubits.row(0), qubits.row(1)))
    broken = _break_grouped_operations(program._instructions)
    assert [step.targets for step in broken] == [
        (qubits[0], qubits[2]),
        (qubits[1], qubits[3]),
    ]


def test_grouped_three_target_operation_zips_views_in_order():
    qubits = GridRegister(3, 2, name="qubits")
    program = Program([qubits])
    program.add(ops.CCX, tuple(qubits.row(row) for row in range(3)))
    broken = _break_grouped_operations(program._instructions)
    assert [step.targets for step in broken] == [
        (qubits[0], qubits[2], qubits[4]),
        (qubits[1], qubits[3], qubits[5]),
    ]


def test_grouped_operation_does_not_mutate_program():
    qubits = GridRegister(2, 2, name="qubits")
    program = Program([qubits])
    program.add(ops.RX(0.3), qubits.row(0))
    before = program._instructions
    broken = _break_grouped_operations(program._instructions)
    assert program._instructions is before
    assert list(program._instructions) != list(broken)


def test_base_backend_executes_grouped_views_with_identity_mapping():
    qubits = GridRegister(2, 2, name="qubits")
    program = Program([qubits])
    program.add(ops.RX(0.3), qubits.row(0))
    result = (
        Simulator()
        .run(
            program,
            result_config={"counts": False, "final_state": True},
        )
        .result()
    )
    assert result.get_statevector() is not None


# --- private lowering context: ResourceLayout for lookup, _EngineAllocation
# for execution indices -----------------------------------------------------


def test_lower_uses_resource_layout_device_labels_for_lookup_and_engine_indices_for_steps():
    # A deliberately divergent mapping: device labels 99/100 have nothing to
    # do with the engine indices 0/1. The CZ rule is only registered for
    # device operands (99, 100), so lowering can only have succeeded by using
    # `ResourceLayout.device_labels_for()` for the implementation-map lookup;
    # the resulting step must still carry the *engine* indices (0, 1), from
    # `_EngineAllocation`, not the device labels used for lookup.
    program = Program(2)
    program.add(ops.CZ, (0, 1))
    q0, q1 = program.quantum_registers[0][0], program.quantum_registers[0][1]

    resource_layout = ResourceLayout({q0: 99, q1: 100})
    engine_allocation = Simulator()._allocate_engine_indices(program, resource_layout)

    cz_rule = default_matrix_implementation_map().implementation_for(ops.CZ)
    implementation_map = MatrixImplementationMap()
    implementation_map.add(ops.CZ, cz_rule, device_operands=(99, 100))
    backend = Simulator(implementation_map=implementation_map)

    context = _LoweringContext(
        resource_layout=resource_layout,
        engine_allocation=engine_allocation,
        classical_allocation=_ClassicalAllocation.from_program(program),
    )
    operations = _break_grouped_operations(program._instructions)
    plan = backend._lower(operations, context)
    assert _matrix_steps(plan)[0].target_indices == (0, 1)


def test_lower_scalar_rotation_emits_one_step():
    program = Program(2)
    program.add(ops.RX(0.3), 1)
    plan, _facts = Simulator()._lower_program(program)
    assert [step.target_indices for step in _matrix_steps(plan)] == [(1,)]
