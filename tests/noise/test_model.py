"""NoiseModel routing: selector validation, lookup precedence, dual addressing."""

import pytest

import fatqat as fq
from fatqat.noise import Depolarizing, NoiseModel, PhaseDamping
from fatqat.operations import ResetGate


def _layout_for(program):
    return fq.backends.SimulatorBackend().resolve_layout(program)


def _two_qubit_program():
    program = fq.Program(2)
    program.add(fq.ops.X, 0)
    return program


def test_all_target_entry_matches_every_occurrence():
    noise = NoiseModel()
    channel = Depolarizing(p=0.1)
    noise.add_noise(fq.ops.X, channel)
    program = _two_qubit_program()
    layout = _layout_for(program)

    assert noise.channels_for(fq.ops.X, (0,), layout) == [channel]
    assert noise.channels_for(fq.ops.X, (1,), layout) == [channel]
    assert noise.channels_for(fq.ops.Y, (0,), layout) == []


def test_specific_entry_replaces_default_only_on_its_target():
    noise = NoiseModel()
    default = Depolarizing(p=0.1)
    specific = Depolarizing(p=0.5)
    program = _two_qubit_program()
    layout = _layout_for(program)
    q = program.qreg[0]
    noise.add_noise(fq.ops.X, default)
    noise.add_noise(fq.ops.X, specific, targets=(q[1],))

    assert noise.channels_for(fq.ops.X, (1,), layout) == [specific]
    assert noise.channels_for(fq.ops.X, (0,), layout) == [default]


def test_int_selector_matches_flat_indices_without_any_register():
    noise = NoiseModel()
    channel = PhaseDamping(p=0.2)
    noise.add_noise(fq.ops.X, channel, targets=(1,))
    program = _two_qubit_program()
    layout = _layout_for(program)

    assert noise.channels_for(fq.ops.X, (1,), layout) == [channel]
    assert noise.channels_for(fq.ops.X, (0,), layout) == []


def test_int_and_ref_entries_resolving_to_same_target_accumulate():
    noise = NoiseModel()
    by_index = Depolarizing(p=0.1)
    by_ref = PhaseDamping(p=0.2)
    program = _two_qubit_program()
    layout = _layout_for(program)
    noise.add_noise(fq.ops.X, by_index, targets=(0,))
    noise.add_noise(fq.ops.X, by_ref, targets=(program.qreg[0][0],))

    assert noise.channels_for(fq.ops.X, (0,), layout) == [by_index, by_ref]


def test_repeated_add_noise_accumulates_in_registration_order():
    noise = NoiseModel()
    first = Depolarizing(p=0.1)
    second = PhaseDamping(p=0.2)
    noise.add_noise(fq.ops.X, first)
    noise.add_noise(fq.ops.X, second)
    layout = _layout_for(_two_qubit_program())

    assert noise.channels_for(fq.ops.X, (0,), layout) == [first, second]


def test_ref_selector_from_foreign_register_never_matches():
    noise = NoiseModel()
    foreign = fq.QuantumRegister(2, name="q")  # same shape as the program's
    noise.add_noise(fq.ops.X, Depolarizing(p=0.9), targets=(foreign[0],))
    layout = _layout_for(_two_qubit_program())

    assert noise.channels_for(fq.ops.X, (0,), layout) == []


def test_two_subsystem_selector_matches_operand_order():
    noise = NoiseModel()
    channel = Depolarizing(p=0.05)
    noise.add_noise(fq.ops.CX, channel, targets=(0, 1))
    layout = _layout_for(_two_qubit_program())

    assert noise.channels_for(fq.ops.CX, (0, 1), layout) == [channel]
    assert noise.channels_for(fq.ops.CX, (1, 0), layout) == []


def test_add_noise_rejects_barrier():
    with pytest.raises(ValueError, match="Barrier"):
        NoiseModel().add_noise(fq.ops.Barrier, Depolarizing(p=0.1))


def test_add_noise_accepts_reset_for_forward_compatibility():
    noise = NoiseModel()
    noise.add_noise(fq.ops.Reset, Depolarizing(p=0.1))
    assert noise.has_noise_for(ResetGate)


def test_add_noise_rejects_non_channel():
    with pytest.raises(TypeError, match="Channel"):
        NoiseModel().add_noise(fq.ops.X, "not a channel")


def test_add_noise_selector_validation():
    noise = NoiseModel()
    ref = fq.QuantumRegister(1)[0]
    with pytest.raises(ValueError, match="non-empty"):
        noise.add_noise(fq.ops.X, Depolarizing(p=0.1), targets=())
    with pytest.raises(TypeError, match="all flat indices"):
        noise.add_noise(fq.ops.CX, Depolarizing(p=0.1), targets=(0, ref))
    with pytest.raises(ValueError, match=">= 0"):
        noise.add_noise(fq.ops.X, Depolarizing(p=0.1), targets=(-1,))
    with pytest.raises(TypeError, match="all flat indices"):
        noise.add_noise(fq.ops.X, Depolarizing(p=0.1), targets=(True,))
    with pytest.raises(ValueError, match="length"):
        noise.add_noise(fq.ops.CX, Depolarizing(p=0.1), targets=(0,))
    program = fq.Program(1, 1)
    with pytest.raises(TypeError, match="QuantumRegister"):
        noise.add_noise(fq.ops.X, Depolarizing(p=0.1), targets=(program.clreg[0][0],))


def test_channel_types_lists_every_attached_descriptor_type():
    noise = NoiseModel()
    noise.add_noise(fq.ops.X, Depolarizing(p=0.1))
    noise.add_noise(fq.ops.H, PhaseDamping(p=0.2))

    assert noise.channel_types() == frozenset({Depolarizing, PhaseDamping})
