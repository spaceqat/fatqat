"""NoiseModel routing: selector validation, lookup precedence, dual addressing."""

import numpy as np
import pytest

import fatqat as fq
from fatqat.errors import BackendValidationError
from fatqat.noise import AmplitudeDamping, Depolarizing, NoiseModel, PhaseDamping
from fatqat.operations import ResetGate
from fatqat.registers import GridRegister
from fatqat.resource_layout import ResourceLayout


def _resource_layout_for(program):
    return fq.simulator.Simulator()._resolve_resource_layout(program)


def _two_qubit_program():
    program = fq.Program(2)
    program.add(fq.ops.X, 0)
    return program


def test_all_target_entry_matches_every_occurrence():
    noise = NoiseModel()
    channel = Depolarizing(p=0.1)
    noise.add_channel(channel, operation=fq.ops.X)
    program = _two_qubit_program()
    layout = _resource_layout_for(program)
    q = program.quantum_registers[0]

    assert noise.channels_for(fq.ops.X, (q[0],), layout) == [(channel, (q[0],))]
    assert noise.channels_for(fq.ops.X, (q[1],), layout) == [(channel, (q[1],))]
    assert noise.channels_for(fq.ops.Y, (q[0],), layout) == []


def test_operation_keyword_records_operation_scope():
    program = fq.Program(1)
    target = program.quantum_registers[0][0]
    layout = ResourceLayout({target: "q0"})
    channel = Depolarizing(p=0.1)
    noise = NoiseModel()

    noise.add_channel(channel, operation=fq.ops.X)

    assert noise.channel_registrations() == ((channel, type(fq.ops.X)),)
    assert noise.channels_for(fq.ops.X, (target,), layout) == [(channel, (target,))]
    assert noise.always_on_channels_for(target, "q0") == ()


def test_omitted_operation_records_always_on_scope():
    program = fq.Program(1)
    target = program.quantum_registers[0][0]
    channel = PhaseDamping(rate=0.01)
    noise = NoiseModel()

    noise.add_channel(channel, targets=target)

    assert noise.channel_registrations() == ((channel, None),)
    assert noise.always_on_channels_for(target, "q0") == (channel,)


def test_always_on_channel_rejects_slots_and_multi_target_selector():
    noise = NoiseModel()
    channel = PhaseDamping(rate=0.01)

    with pytest.raises(ValueError, match="does not accept slots"):
        noise.add_channel(channel, slots=0)
    with pytest.raises(ValueError, match="exactly one subsystem"):
        noise.add_channel(channel, targets=(0, 1))


def test_specific_ref_entry_replaces_default_only_on_its_target():
    noise = NoiseModel()
    default = Depolarizing(p=0.1)
    specific = Depolarizing(p=0.5)
    program = _two_qubit_program()
    layout = _resource_layout_for(program)
    q = program.quantum_registers[0]
    noise.add_channel(default, operation=fq.ops.X)
    noise.add_channel(specific, operation=fq.ops.X, targets=(q[1],))

    assert noise.channels_for(fq.ops.X, (q[1],), layout) == [(specific, (q[1],))]
    assert noise.channels_for(fq.ops.X, (q[0],), layout) == [(default, (q[0],))]


def test_physical_label_selector_matches_device_operands():
    noise = NoiseModel()
    channel = PhaseDamping(p=0.2)
    noise.add_channel(channel, operation=fq.ops.X, targets=(1,))  # device label 1
    program = _two_qubit_program()
    layout = _resource_layout_for(program)
    q = program.quantum_registers[0]

    assert noise.channels_for(fq.ops.X, (q[1],), layout) == [(channel, (q[1],))]
    assert noise.channels_for(fq.ops.X, (q[0],), layout) == []


def test_physical_selector_accepts_non_int_hashable_label():
    program = _two_qubit_program()
    q = program.quantum_registers[0]
    layout = ResourceLayout({q[0]: "trap-a", q[1]: "trap-b"})

    noise = NoiseModel()
    channel = Depolarizing(p=0.1)
    noise.add_channel(channel, operation=fq.ops.X, targets=("trap-b",))

    assert noise.channels_for(fq.ops.X, (q[1],), layout) == [(channel, (q[1],))]
    assert noise.channels_for(fq.ops.X, (q[0],), layout) == []


def test_logical_and_physical_entries_resolving_to_same_target_accumulate():
    noise = NoiseModel()
    by_label = Depolarizing(p=0.1)
    by_ref = PhaseDamping(p=0.2)
    program = _two_qubit_program()
    layout = _resource_layout_for(program)
    q = program.quantum_registers[0]
    noise.add_channel(by_label, operation=fq.ops.X, targets=(0,))
    noise.add_channel(by_ref, operation=fq.ops.X, targets=(q[0],))

    assert noise.channels_for(fq.ops.X, (q[0],), layout) == [
        (by_label, (q[0],)),
        (by_ref, (q[0],)),
    ]


def test_repeated_add_channel_accumulates_in_registration_order():
    noise = NoiseModel()
    first = Depolarizing(p=0.1)
    second = PhaseDamping(p=0.2)
    noise.add_channel(first, operation=fq.ops.X)
    noise.add_channel(second, operation=fq.ops.X)
    program = _two_qubit_program()
    layout = _resource_layout_for(program)
    q = program.quantum_registers[0]

    assert noise.channels_for(fq.ops.X, (q[0],), layout) == [
        (first, (q[0],)),
        (second, (q[0],)),
    ]


def test_ref_selector_from_foreign_register_never_matches():
    noise = NoiseModel()
    foreign = fq.QuantumRegister(2, name="q")  # same shape as the program's
    noise.add_channel(Depolarizing(p=0.9), operation=fq.ops.X, targets=(foreign[0],))
    program = _two_qubit_program()
    layout = _resource_layout_for(program)
    q = program.quantum_registers[0]

    assert noise.channels_for(fq.ops.X, (q[0],), layout) == []


def test_two_subsystem_selector_matches_operand_order():
    noise = NoiseModel()
    channel = Depolarizing(p=0.05)
    noise.add_channel(channel, operation=fq.ops.CX, targets=(0, 1))
    program = _two_qubit_program()
    layout = _resource_layout_for(program)
    q = program.quantum_registers[0]

    assert noise.channels_for(fq.ops.CX, (q[0], q[1]), layout) == [
        (channel, (q[0], q[1]))
    ]
    assert noise.channels_for(fq.ops.CX, (q[1], q[0]), layout) == []


def test_mixed_logical_and_physical_tuple_selector_rejected():
    noise = NoiseModel()
    ref = fq.QuantumRegister(2)[0]
    with pytest.raises(TypeError, match="not mixed"):
        noise.add_channel(Depolarizing(p=0.1), operation=fq.ops.CX, targets=(ref, 0))


def test_register_view_selector_rejected():
    atoms = GridRegister(2, 3, name="atoms")
    noise = NoiseModel()
    with pytest.raises(TypeError, match="RegisterView"):
        noise.add_channel(
            Depolarizing(p=0.1), operation=fq.ops.RX, targets=(atoms.row(0),)
        )


def test_add_channel_rejects_barrier():
    with pytest.raises(ValueError, match="Barrier"):
        NoiseModel().add_channel(Depolarizing(p=0.1), operation=fq.ops.Barrier)


def test_add_channel_accepts_reset():
    noise = NoiseModel()
    noise.add_channel(Depolarizing(p=0.1), operation=fq.ops.Reset)
    assert noise.has_noise_for(ResetGate)


def test_add_channel_rejects_non_channel():
    with pytest.raises(TypeError, match="Channel"):
        NoiseModel().add_channel("not a channel", operation=fq.ops.X)


def test_add_channel_selector_validation():
    noise = NoiseModel()
    with pytest.raises(ValueError, match="non-empty"):
        noise.add_channel(Depolarizing(p=0.1), operation=fq.ops.X, targets=())
    with pytest.raises(ValueError, match="length"):
        noise.add_channel(Depolarizing(p=0.1), operation=fq.ops.CX, targets=(0,))
    program = fq.Program(1, 1)
    with pytest.raises(TypeError, match="QuantumRegister"):
        noise.add_channel(
            Depolarizing(p=0.1),
            operation=fq.ops.X,
            targets=(program.classical_registers[0][0],),
        )
    # A physical selector is an opaque label: negative ints, strings, and
    # even bools are all legal device-resource labels now, not flat indices.
    noise.add_channel(Depolarizing(p=0.1), operation=fq.ops.X, targets=(-1,))
    noise.add_channel(Depolarizing(p=0.1), operation=fq.ops.Y, targets=("zone-a",))


def test_add_channel_accepts_bare_label_for_arity_one_operation():
    noise = NoiseModel()
    channel = Depolarizing(p=0.1)
    noise.add_channel(channel, operation=fq.ops.X, targets=0)  # bare device label
    program = _two_qubit_program()
    layout = _resource_layout_for(program)
    q = program.quantum_registers[0]

    assert noise.channels_for(fq.ops.X, (q[0],), layout) == [(channel, (q[0],))]
    assert noise.channels_for(fq.ops.X, (q[1],), layout) == []


def test_add_channel_accepts_bare_ref_for_arity_one_operation():
    noise = NoiseModel()
    channel = Depolarizing(p=0.1)
    program = _two_qubit_program()
    layout = _resource_layout_for(program)
    q = program.quantum_registers[0]
    noise.add_channel(channel, operation=fq.ops.X, targets=q[1])  # bare RegisterRef

    assert noise.channels_for(fq.ops.X, (q[1],), layout) == [(channel, (q[1],))]
    assert noise.channels_for(fq.ops.X, (q[0],), layout) == []


def test_add_channel_accepts_bare_label_for_variable_arity_operation():
    noise = NoiseModel()
    noise.add_channel(Depolarizing(p=0.1), operation=fq.ops.Reset, targets=0)
    assert noise.has_noise_for(ResetGate)


def test_add_channel_bare_target_rejected_for_multi_target_operation():
    noise = NoiseModel()
    with pytest.raises(ValueError, match="length"):
        noise.add_channel(Depolarizing(p=0.1), operation=fq.ops.CX, targets=0)


def test_add_channel_bare_register_view_rejected():
    atoms = GridRegister(2, 3, name="atoms")
    noise = NoiseModel()
    with pytest.raises(TypeError, match="RegisterView"):
        noise.add_channel(
            Depolarizing(p=0.1), operation=fq.ops.RX, targets=atoms.row(0)
        )


def test_slots_resolve_a_single_subsystem_of_a_two_qubit_gate():
    noise = NoiseModel()
    damping = AmplitudeDamping(p=(0.1,))
    noise.add_channel(damping, operation=fq.ops.CZ, slots=1)
    program = _two_qubit_program()
    q = program.quantum_registers[0]

    assert noise.channels_for(
        fq.ops.CZ, (q[0], q[1]), _resource_layout_for(program)
    ) == [(damping, (q[1],))]


def test_slot_precedence_is_grouped_by_extent():
    noise = NoiseModel()
    joint = Depolarizing(p=0.1)
    scoped = AmplitudeDamping(p=(0.1,))
    override = Depolarizing(p=0.2)
    program = _two_qubit_program()
    q = program.quantum_registers[0]
    layout = _resource_layout_for(program)
    noise.add_channel(joint, operation=fq.ops.CZ)
    noise.add_channel(scoped, operation=fq.ops.CZ, slots=(0,))
    noise.add_channel(override, operation=fq.ops.CZ, targets=(q[0], q[1]))

    assert noise.channels_for(fq.ops.CZ, (q[0], q[1]), layout) == [
        (scoped, (q[0],)),
        (override, (q[0], q[1])),
    ]


def test_slots_reject_bad_extent_and_whole_gate_single_channel():
    noise = NoiseModel()
    damping = AmplitudeDamping(p=(0.1,))
    with pytest.raises(ValueError, match="slots="):
        noise.add_channel(damping, operation=fq.ops.CZ)
    with pytest.raises(ValueError, match="out of range"):
        noise.add_channel(damping, operation=fq.ops.CZ, slots=(2,))
    with pytest.raises(ValueError, match="strictly increasing"):
        noise.add_channel(damping, operation=fq.ops.CZ, slots=(1, 0))


def test_channel_types_lists_every_attached_descriptor_type():
    noise = NoiseModel()
    noise.add_channel(Depolarizing(p=0.1), operation=fq.ops.X)
    noise.add_channel(PhaseDamping(p=0.2), operation=fq.ops.H)

    assert noise.channel_types() == frozenset({Depolarizing, PhaseDamping})


# --- readout error registration and lookup ---
#
# Readout selectors mirror the gate-channel identity spaces (logical
# RegisterRef vs. physical device-resource label), but the stored selector is
# scalar (None | RegisterRef | device label), not a tuple, and matching
# selection is single-winner (most-recently-registered specific match wins),
# not accumulate-all like gate channels. `readout_error_for()` takes the
# measured ref plus the run's resource layout - never an engine index.

_C_MILD = np.array([[0.9, 0.2], [0.1, 0.8]])
_C_STRONG = np.array([[0.5, 0.5], [0.5, 0.5]])


def test_readout_error_all_target_default_and_ref_override():
    noise = NoiseModel()
    program = _two_qubit_program()
    layout = _resource_layout_for(program)
    q = program.quantum_registers[0]
    noise.add_readout_error(_C_MILD)
    noise.add_readout_error(_C_STRONG, target=q[1])

    assert np.array_equal(noise.readout_error_for(q[0], layout), _C_MILD)
    assert np.array_equal(noise.readout_error_for(q[1], layout), _C_STRONG)
    assert noise.has_readout_error()


def test_readout_error_physical_label_selector_matches_device_label():
    noise = NoiseModel()
    program = _two_qubit_program()
    layout = _resource_layout_for(program)
    q = program.quantum_registers[0]
    noise.add_readout_error(_C_STRONG, target=1)  # device label 1

    assert np.array_equal(noise.readout_error_for(q[1], layout), _C_STRONG)
    assert noise.readout_error_for(q[0], layout) is None


def test_readout_error_ref_selector_from_foreign_register_never_matches():
    noise = NoiseModel()
    program = _two_qubit_program()
    layout = _resource_layout_for(program)
    q = program.quantum_registers[0]
    noise.add_readout_error(_C_MILD, target=fq.QuantumRegister(2)[1])

    assert noise.readout_error_for(q[1], layout) is None


def test_readout_error_last_specific_entry_wins_among_physical_selectors():
    noise = NoiseModel()
    program = _two_qubit_program()
    layout = _resource_layout_for(program)
    q = program.quantum_registers[0]
    noise.add_readout_error(_C_MILD, target=0)
    noise.add_readout_error(_C_STRONG, target=0)

    assert np.array_equal(noise.readout_error_for(q[0], layout), _C_STRONG)


def test_readout_error_last_specific_entry_wins_across_logical_and_physical():
    program = _two_qubit_program()
    layout = _resource_layout_for(program)
    q = program.quantum_registers[0]

    logical_then_physical = NoiseModel()
    logical_then_physical.add_readout_error(_C_MILD, target=q[0])
    logical_then_physical.add_readout_error(_C_STRONG, target=0)  # same subsystem
    assert np.array_equal(
        logical_then_physical.readout_error_for(q[0], layout), _C_STRONG
    )

    physical_then_logical = NoiseModel()
    physical_then_logical.add_readout_error(_C_STRONG, target=0)
    physical_then_logical.add_readout_error(_C_MILD, target=q[0])  # same subsystem
    assert np.array_equal(
        physical_then_logical.readout_error_for(q[0], layout), _C_MILD
    )


def test_readout_error_matrix_is_copied_and_frozen():
    noise = NoiseModel()
    source = _C_MILD.copy()
    noise.add_readout_error(source)
    program = _two_qubit_program()
    layout = _resource_layout_for(program)
    q = program.quantum_registers[0]
    source[0, 0] = 0.0  # caller mutation must not reach the stored matrix

    stored = noise.readout_error_for(q[0], layout)
    assert stored[0, 0] == 0.9
    assert not stored.flags.writeable


def test_readout_error_validation():
    noise = NoiseModel()
    with pytest.raises(ValueError, match="square"):
        noise.add_readout_error(np.ones((2, 3)))
    with pytest.raises(ValueError, match="column-stochastic"):
        noise.add_readout_error(np.array([[0.9, 0.1], [0.2, 0.8]]))
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        noise.add_readout_error(np.array([[1.5, 0.0], [-0.5, 1.0]]))
    with pytest.raises(TypeError, match="RegisterView"):
        atoms = GridRegister(2, 3, name="atoms")
        noise.add_readout_error(_C_MILD, target=atoms.row(0))
    program = fq.Program(1, 1)
    with pytest.raises(TypeError, match="QuantumRegister"):
        noise.add_readout_error(_C_MILD, target=program.classical_registers[0][0])
    # A physical selector is an opaque label now, not a flat index: negative
    # ints and strings are both legal device-resource labels.
    noise.add_readout_error(_C_MILD, target=-1)
    noise.add_readout_error(_C_MILD, target="q0")


# --- validate_for: strict run-time selector-identity legality ---
#
# validate_for() checks identity legality (does the ref/label denote
# something real for this program/layout), never occurrence matching (does
# any gate/measurement actually use it). Both stored shapes - tuple gate
# selectors and scalar readout selectors - must be checked.


def _three_qubit_program_and_layout():
    program = fq.Program(3)
    layout = _resource_layout_for(program)
    return program, layout


def test_validate_for_accepts_valid_selectors_that_match_no_occurrence():
    # A valid ref/label with no matching gate occurrence or measurement is a
    # permitted no-effect entry, not a validation error.
    program, layout = _three_qubit_program_and_layout()
    q = program.quantum_registers[0]
    noise = NoiseModel()
    noise.add_channel(
        Depolarizing(p=0.1), operation=fq.ops.X, targets=(q[0],)
    )  # no X in program
    noise.add_channel(
        Depolarizing(p=0.1), operation=fq.ops.Y, targets=(2,)
    )  # no Y in program
    noise.add_readout_error(_C_MILD, target=q[1])  # never measured
    noise.add_readout_error(_C_MILD, target=2)  # never measured

    noise.validate_for(program, layout)  # must not raise


def test_validate_for_rejects_foreign_logical_gate_selector():
    program, layout = _three_qubit_program_and_layout()
    foreign = fq.QuantumRegister(3, name="q")
    noise = NoiseModel()
    noise.add_channel(Depolarizing(p=0.1), operation=fq.ops.X, targets=(foreign[0],))

    with pytest.raises(BackendValidationError):
        noise.validate_for(program, layout)


def test_validate_for_rejects_foreign_logical_readout_selector():
    program, layout = _three_qubit_program_and_layout()
    foreign = fq.QuantumRegister(3, name="q")
    noise = NoiseModel()
    noise.add_readout_error(_C_MILD, target=foreign[0])

    with pytest.raises(BackendValidationError):
        noise.validate_for(program, layout)


def test_validate_for_rejects_unmapped_physical_gate_label():
    # The (99,) case from the spec: not a member of the effective layout's
    # device labels for this three-subsystem generic-simulator program.
    program, layout = _three_qubit_program_and_layout()
    noise = NoiseModel()
    noise.add_channel(Depolarizing(p=0.1), operation=fq.ops.X, targets=(99,))

    with pytest.raises(BackendValidationError):
        noise.validate_for(program, layout)


def test_validate_for_rejects_unmapped_physical_readout_label():
    program, layout = _three_qubit_program_and_layout()
    noise = NoiseModel()
    noise.add_readout_error(_C_MILD, target=99)

    with pytest.raises(BackendValidationError):
        noise.validate_for(program, layout)


def test_validate_for_checks_both_stored_selector_shapes_in_one_model():
    # A gate selector (tuple-shaped) and a readout selector (scalar-shaped)
    # are stored differently; both must be validated, not just one.
    program, layout = _three_qubit_program_and_layout()
    noise = NoiseModel()
    noise.add_channel(Depolarizing(p=0.1), operation=fq.ops.X, targets=(99,))
    noise.add_readout_error(_C_MILD, target=99)

    with pytest.raises(BackendValidationError):
        noise.validate_for(program, layout)


def test_validate_for_accepts_noise_free_model():
    program, layout = _three_qubit_program_and_layout()
    NoiseModel().validate_for(program, layout)  # must not raise
