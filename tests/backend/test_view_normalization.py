"""Tests for grouped-view/operation expansion and the private lowering context.

`_break_grouped_operations()` is the temporary scalar view-normalization
stage: it must run before scalar lowering and must not carry any resource-
mapping policy of its own (that lives in `ResourceLayout`/`_EngineIndexAllocation`
and the private `_LoweringContext` that pairs them for one run's lowering).

Pairing legality (matching selector kind, equal cardinality, no same-register
overlap) is validated at `Program.add()`/`AppliedOperation` construction time
(see tests/program/test_add.py and tests/program/test_applied_operation.py),
not here - by the time a step reaches expansion, its targets are already
guaranteed legal.
"""

import pytest

from fatqat import operations as ops
from fatqat.backends import ApplyMatrixStep, SimulatorBackend
from fatqat.backends.backend_utils import _LoweringContext
from fatqat.backends.simulator_backend import _break_grouped_operations
from fatqat.implementation import ImplementationMap, default_matrix_implementation_map
from fatqat.program import Program
from fatqat.registers import GridRegister
from fatqat.resource_layout import ResourceLayout


def _matrix_steps(plan):
    return [step for step in plan if isinstance(step, ApplyMatrixStep)]


def test_scalar_only_and_measurement_instructions_pass_through_unchanged():
    program = Program(2, 1)
    program.add(ops.RX(0.3), 1)
    program.add(ops.CZ, (0, 1))
    program.measure(0, 0)
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
            result_config={"counts": False, "final_state": True},
        )
        .result()
    )
    assert result.get_statevector() is not None


# --- private lowering context: ResourceLayout for lookup, _EngineIndexAllocation
# for execution indices -----------------------------------------------------


def test_lower_uses_resource_layout_device_labels_for_lookup_and_engine_indices_for_steps():
    # A deliberately divergent mapping: device labels 99/100 have nothing to
    # do with the engine indices 0/1. The CZ rule is only registered for
    # device operands (99, 100), so lowering can only have succeeded by using
    # `ResourceLayout.device_labels_for()` for the implementation-map lookup;
    # the resulting step must still carry the *engine* indices (0, 1), from
    # `_EngineIndexAllocation`, not the device labels used for lookup.
    program = Program(2)
    program.add(ops.CZ, (0, 1))
    q0, q1 = program.quantum_registers[0][0], program.quantum_registers[0][1]

    engine_index_allocation = SimulatorBackend()._allocate_engine_indices(program)
    resource_layout = ResourceLayout({q0: 99, q1: 100})

    cz_rule = default_matrix_implementation_map().implementation_for(ops.CZ)
    implementation_map = ImplementationMap()
    implementation_map.add(ops.CZ, cz_rule, device_operands=(99, 100))
    backend = SimulatorBackend(implementation_map=implementation_map)

    context = _LoweringContext(
        resource_layout=resource_layout,
        engine_index_allocation=engine_index_allocation,
    )
    operations = _break_grouped_operations(program.operations)
    plan, _facts = backend._lower(operations, context)
    assert _matrix_steps(plan)[0].target_indices == (0, 1)


def test_lower_scalar_rotation_emits_one_step():
    program = Program(2)
    program.add(ops.RX(0.3), 1)
    plan, _facts = SimulatorBackend()._lower_program(program)
    assert [step.target_indices for step in _matrix_steps(plan)] == [(1,)]
