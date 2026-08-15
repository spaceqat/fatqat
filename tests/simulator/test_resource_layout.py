"""Tests for the public ResourceLayout: RegisterRef -> device resource label."""

import pytest

from fatqat import operations as ops
from fatqat._backends.backend_utils import _LoweringContext
from fatqat._backends.steps import ApplyMatrixStep, MeasurementStep
from fatqat._index_allocation import _ClassicalAllocation, _EngineAllocation
from fatqat.errors import BackendValidationError
from fatqat.simulator import AtomGridSimulator, Simulator
from fatqat.program import Program
from fatqat.registers import QuantumRegister
from fatqat.resource_layout import ResourceLayout


def test_device_label_lookup():
    qreg = QuantumRegister(2, name="q")
    layout = ResourceLayout({qreg[0]: 0, qreg[1]: 1})
    assert layout.device_label(qreg[0]) == 0
    assert layout.device_label(qreg[1]) == 1


def test_device_label_opaque_hashable_values():
    qreg = QuantumRegister(2, name="q")
    layout = ResourceLayout({qreg[0]: "site-A", qreg[1]: (3, 7)})
    assert layout.device_label(qreg[0]) == "site-A"
    assert layout.device_label(qreg[1]) == (3, 7)


def test_device_labels_for_preserves_operand_order():
    qreg = QuantumRegister(3, name="q")
    layout = ResourceLayout({qreg[0]: "a", qreg[1]: "b", qreg[2]: "c"})
    assert layout.device_labels_for((qreg[2], qreg[0])) == ("c", "a")
    assert layout.device_labels_for(()) == ()


def test_device_labels_membership():
    qreg = QuantumRegister(2, name="q")
    layout = ResourceLayout({qreg[0]: 5, qreg[1]: 9})
    assert layout.device_labels == frozenset({5, 9})


def test_private_reverse_lookup_requires_one_unique_device_label():
    qreg = QuantumRegister(2, name="q")
    first = qreg[0]
    second = qreg[1]
    unique = ResourceLayout({first: "a", second: "b"})
    duplicate = ResourceLayout({first: "shared", second: "shared"})

    assert unique._ref_for_label("b") is second
    with pytest.raises(KeyError, match="exactly one"):
        duplicate._ref_for_label("shared")


def test_foreign_ref_raises_on_device_label():
    qreg = QuantumRegister(1, name="q")
    foreign = QuantumRegister(1, name="x")
    layout = ResourceLayout({qreg[0]: 0})
    with pytest.raises(KeyError):
        layout.device_label(foreign[0])


def test_foreign_ref_raises_on_device_labels_for():
    qreg = QuantumRegister(1, name="q")
    foreign = QuantumRegister(1, name="x")
    layout = ResourceLayout({qreg[0]: 0})
    with pytest.raises(KeyError):
        layout.device_labels_for((qreg[0], foreign[0]))


def test_lookalike_register_is_not_part_of_the_layout():
    qreg = QuantumRegister(1, name="q")
    layout = ResourceLayout({qreg[0]: 0})
    lookalike = QuantumRegister(qreg.size, name=qreg.name)
    with pytest.raises(KeyError):
        layout.device_label(lookalike[0])


# --- Simulator's default resource-mapping hook -----------------------


def test_simulator_resolves_generic_identity_device_labels_in_declaration_order():
    program = Program(3)
    backend = Simulator()

    resource_layout = backend._resolve_resource_layout(program)

    refs = [program.quantum_registers[0][i] for i in range(3)]
    assert [resource_layout.device_label(ref) for ref in refs] == [0, 1, 2]
    assert isinstance(resource_layout, ResourceLayout)


def test_simulator_allocate_engine_indices_returns_a_separate_engine_allocation():
    program = Program(3)
    backend = Simulator()

    resource_layout = backend._resolve_resource_layout(program)
    engine_index_allocation = backend._allocate_engine_indices(program, resource_layout)

    refs = [program.quantum_registers[0][i] for i in range(3)]
    # Two independently-resolved values: not the same object...
    assert resource_layout is not engine_index_allocation
    assert isinstance(engine_index_allocation, _EngineAllocation)
    # ...but the generic simulator's trivial policy makes their current
    # numerical values coincide. That coincidence is not an API contract.
    assert [
        engine_index_allocation.engine_index(resource_layout.device_label(ref))
        for ref in refs
    ] == [resource_layout.device_label(ref) for ref in refs]


def test_supplied_layout_must_cover_unused_declared_quantum_refs():
    program = Program(2)
    partial = ResourceLayout({program.quantum_registers[0][0]: 0})

    with pytest.raises(BackendValidationError, match="every declared quantum ref"):
        Simulator().run(program, resource_layout=partial)


def test_supplied_layout_composes_sparse_device_labels_to_dense_engine_axes():
    left = QuantumRegister(1, name="q")
    right = QuantumRegister(1, name="q")
    program = Program([left, right], 2)
    program.add(ops.X, left[0])
    program.measure((left[0], right[0]), (0, 1))
    q0, q1 = left[0], right[0]
    supplied = ResourceLayout({q0: 9, q1: 3})
    backend = Simulator()

    result = backend.run(
        program,
        shots=1,
        resource_layout=supplied,
        result_config={"counts": False, "final_state": True},
    ).result()
    resolved = backend._resolve_resource_layout(program, supplied)
    engine = backend._allocate_engine_indices(program, resolved)
    context = _LoweringContext(
        resource_layout=resolved,
        engine_allocation=engine,
        classical_allocation=_ClassicalAllocation.from_program(program),
    )
    plan, _facts = backend._lower_program(program, context=context)

    assert engine.device_operands == (9, 3)
    matrix_step = next(step for step in plan if isinstance(step, ApplyMatrixStep))
    assert matrix_step.target_indices == (0,)
    measurement = next(step for step in plan if isinstance(step, MeasurementStep))
    assert measurement.measured_indices == (0, 1)
    assert result.metadata["state_axes"] == [
        {"device_operand": 9, "register_ref": left[0]},
        {"device_operand": 3, "register_ref": right[0]},
    ]
    assert all("engine_index" not in axis for axis in result.metadata["state_axes"])
    with pytest.raises(BackendValidationError, match="does not accept"):
        AtomGridSimulator().run(program, resource_layout=supplied)
