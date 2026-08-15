"""Gate-keyed AmplitudeDamping/PhaseDamping lowered into pulse intervals."""

from math import sqrt

import numpy as np
import pytest
from qutip import basis, ket2dm, tensor

import fatqat as fq
from fatqat._pulse_values import PulseControl
from fatqat._index_allocation import _EngineAllocation
from fatqat.emulator.superconducting.backend import TransmonEmulator
from fatqat.emulator._core.engine import _ShotContext
from fatqat.emulator._core.scheduling import schedule_pulse_run
from fatqat.emulator._core.lindblad import ResolvedLindbladTerm
from fatqat.emulator.superconducting.qutip_adapter import _TransmonQutipAdapter
from fatqat.emulator.superconducting.target import _TransmonTarget
from fatqat.emulator._core.pulse import (
    PulseBlock,
    PulseDefinition,
    PulseImplementationMap,
)
from fatqat.emulator._core.target import _PreparedControlBinding
from fatqat.waveforms import SampledWaveform
from fatqat.errors import BackendValidationError
from fatqat.noise import (
    AmplitudeDamping,
    Depolarizing,
    LindbladImplementationMap,
    NoiseModel,
    PhaseDamping,
    ThermalRelaxation,
)
from fatqat.noise.lindblad import phase_damping_lindblad_rule


@pytest.fixture(name="make_backend")
def make_backend_fixture(model, calibration):
    """Build a backend on the shared model with an optional noise model."""

    def build(noise=None):
        return TransmonEmulator(model, noise=noise)

    return build


def _idle_block(adapter, subsystem_id, *, duration, noise=(), condition=None):
    """A driven-but-zero-amplitude block: a nonzero-duration slot with no
    coherent dynamics, so only the attached collapse terms act."""
    target = adapter._target
    model = target.model
    controls = (
        PulseControl(
            model.drive_control(subsystem_id),
            SampledWaveform((0.0, duration), (0.0, 0.0)),
        ),
    )
    target_binding = target.bind_control(controls[0].channel)
    bindings = (
        _PreparedControlBinding(
            target_binding.kind,
            tuple(
                target.device_labels.index(value)
                for value in target_binding.device_operands
            ),
        ),
    )
    return PulseBlock(
        duration,
        controls,
        bindings,
        target_binding.claims,
        condition=condition,
        noise=noise,
    )


def _context(adapter, state):
    return _ShotContext(state=state, classical_memory=[], rng=np.random.default_rng(0))


def _adapter(model, **kwargs):
    target = _TransmonTarget(model)
    return _TransmonQutipAdapter(
        target,
        engine_allocation=_EngineAllocation(
            target.device_labels,
            (target.local_dimension,) * len(target.device_labels),
        ),
        **kwargs,
    )


def _amplitude_term(rates, *, ordinal=0):
    operator = np.zeros((3, 3), dtype=complex)
    for level, rate in enumerate(rates, start=1):
        operator[level - 1, level] = sqrt(rate)
    return ResolvedLindbladTerm(operator, (ordinal,))


def _phase_term(rate, *, ordinal=0):
    return ResolvedLindbladTerm(sqrt(2 * rate) * np.diag((0.0, 1.0, 2.0)), (ordinal,))


def _evolve(adapter, blocks, context, *, boundary=0.0):
    run = schedule_pulse_run(blocks, boundary_time=boundary)
    adapter.evolve(run, context, (True,) * len(run.blocks))
    context.time = run.end_time
    return context


# --- validate_noise / capability reporting ---------------------------------


def test_pulse_backend_accepts_gate_keyed_damping_in_either_mode(make_backend):
    noise = NoiseModel()
    noise.add_channel(AmplitudeDamping(p=(0.01, 0.02)), operation=fq.ops.RX)
    noise.add_channel(PhaseDamping(rate=0.001), operation=fq.ops.RX)
    report = make_backend(noise).validate_noise(noise)

    assert report.supported is True
    assert set(report.accepted_sources) == {
        "AmplitudeDamping(p)",
        "PhaseDamping(rate)",
    }
    assert report.rejected_sources == ()


@pytest.mark.parametrize(
    "channel",
    (AmplitudeDamping(p=(0.1,)), AmplitudeDamping(rate=(0.1,))),
)
def test_pulse_backend_rejects_qutrit_amplitude_damping_with_wrong_arity(
    make_backend, channel
):
    invalid = NoiseModel()
    invalid.add_channel(channel, operation=fq.ops.RX)
    report = make_backend(invalid).validate_noise(invalid)
    assert not report.supported
    assert "arity-1" in report.rejected_sources[0]
    assert "requires 2 damping values" in report.warnings[0]

    valid = NoiseModel()
    valid.add_channel(AmplitudeDamping(rate=(0.1, 0.2)), operation=fq.ops.RX)
    assert make_backend(valid).validate_noise(valid).supported


def test_pulse_backend_accepts_always_on_rate_and_rejects_probability(make_backend):
    noise = NoiseModel()
    noise.add_channel(PhaseDamping(rate=0.001), targets="q0")
    noise.add_channel(AmplitudeDamping(p=(0.01, 0.02)), targets="q1")

    report = make_backend(noise).validate_noise(noise)

    assert report.accepted_sources == ("PhaseDamping(rate, always-on)",)
    assert report.rejected_sources == ("AmplitudeDamping(p, always-on)",)
    assert "requires rate mode" in report.warnings[0]


def test_always_on_rate_lowers_to_the_same_lindblad_term(make_backend):
    noise = NoiseModel()
    noise.add_channel(PhaseDamping(rate=0.0025), targets="q0")
    backend = make_backend(noise)

    bindings = backend._prepare_program(fq.Program(1)).always_on_noise

    assert len(bindings) == 1
    assert bindings[0].engine_indices == (0,)
    assert np.allclose(bindings[0].local_operator, _phase_term(0.0025).local_operator)


def test_pulse_backend_still_rejects_channel_types_without_a_pulse_implementation(
    make_backend,
):
    noise = NoiseModel()
    noise.add_channel(Depolarizing(p=0.1), operation=fq.ops.RX)
    backend = make_backend(noise)

    report = backend.validate_noise(noise)
    assert report.supported is False
    assert report.rejected_sources == ("Depolarizing",)

    program = fq.Program(1)
    program.add(fq.ops.RX(0.3), 0)
    with pytest.raises(BackendValidationError, match="Depolarizing"):
        backend.run(program)


def test_lindblad_implementation_map_declares_pulse_noise_capability(
    model, calibration
):
    implementations = LindbladImplementationMap()
    implementations.register(PhaseDamping, phase_damping_lindblad_rule)
    noise = NoiseModel()
    noise.add_channel(AmplitudeDamping(rate=(0.01, 0.02)), operation=fq.ops.RX)
    backend = TransmonEmulator(
        model,
        noise=noise,
        lindblad_implementation_map=implementations,
    )

    assert backend.validate_noise(noise).rejected_sources == ("AmplitudeDamping(rate)",)

    implementations.register(AmplitudeDamping, phase_damping_lindblad_rule)
    assert backend.validate_noise(noise).rejected_sources == ("AmplitudeDamping(rate)",)


def test_run_rejects_unsupported_channel_from_lowering_even_bypassing_validate_noise(
    make_backend,
):
    noise = NoiseModel()
    noise.add_channel(Depolarizing(p=0.1), operation=fq.ops.RX)
    backend = make_backend(noise)
    program = fq.Program(1)
    program.add(fq.ops.RX(0.3), 0)

    with pytest.raises(BackendValidationError, match="Depolarizing"):
        backend._prepare_program(program)


# --- lowering: rate resolution ----------------------------------------------


def test_probability_mode_damping_lowers_to_the_converted_rate(make_backend):
    backend = make_backend()
    backend._noise_model.add_channel(
        AmplitudeDamping(p=(0.01, 0.02)), operation=fq.ops.RX
    )
    program = fq.Program(1)
    program.add(fq.ops.RX(0.3), 0)
    plan = backend._prepare_program(program).plan
    (block,) = [step for step in plan if isinstance(step, PulseBlock)]

    (binding,) = block.noise
    expected = AmplitudeDamping(p=(0.01, 0.02)).as_rate(block.duration)
    assert binding.engine_indices == (0,)
    assert np.allclose(binding.local_operator, _amplitude_term(expected).local_operator)


def test_rate_mode_damping_lowers_unchanged(make_backend):
    backend = make_backend()
    backend._noise_model.add_channel(PhaseDamping(rate=0.0025), operation=fq.ops.RX)
    program = fq.Program(1)
    program.add(fq.ops.RX(0.3), 0)
    plan = backend._prepare_program(program).plan
    (block,) = [step for step in plan if isinstance(step, PulseBlock)]

    (binding,) = block.noise
    assert np.allclose(binding.local_operator, _phase_term(0.0025).local_operator)


def test_gate_scoped_noise_resolves_using_the_custom_rules_realized_duration(
    model, calibration
):
    # Gate-scoped noise must key off whatever duration the *selected* rule
    # actually realizes, not a duration baked into the default recipe - the
    # rule is chosen through PulseImplementationMap, so a custom rule with a
    # different duration must change the resolved rate exactly as the
    # default rule's own duration does.
    custom_duration = 7.0

    def custom_rx(operation, *, device_operands):
        del operation
        (subsystem_id,) = device_operands
        return PulseDefinition(
            custom_duration,
            (
                PulseControl(
                    model.drive_control(subsystem_id),
                    SampledWaveform((0.0, custom_duration), (0.0, 0.0)),
                ),
            ),
        )

    implementations = PulseImplementationMap()
    implementations.add(fq.ops.RX, custom_rx)
    backend = TransmonEmulator(model, gate_implementation_map=implementations)
    backend._noise_model.add_channel(
        AmplitudeDamping(p=(0.01, 0.02)), operation=fq.ops.RX
    )

    program = fq.Program(1)
    program.add(fq.ops.RX(0.3), 0)
    plan = backend._prepare_program(program).plan
    (block,) = plan

    assert block.duration == custom_duration
    (binding,) = block.noise
    expected = AmplitudeDamping(p=(0.01, 0.02)).as_rate(custom_duration)
    assert np.allclose(binding.local_operator, _amplitude_term(expected).local_operator)


def test_nonzero_probability_on_zero_duration_gate_is_rejected_at_lowering(
    make_backend,
):
    backend = make_backend()
    backend._noise_model.add_channel(PhaseDamping(p=0.1), operation=fq.ops.RZ)
    program = fq.Program(1)
    program.add(fq.ops.RZ(0.2), 0)

    with pytest.raises(BackendValidationError, match="zero duration"):
        backend._prepare_program(program)


def test_zero_probability_on_zero_duration_gate_is_a_silent_no_op(make_backend):
    backend = make_backend()
    backend._noise_model.add_channel(PhaseDamping(p=0.0), operation=fq.ops.RZ)
    program = fq.Program(1)
    program.add(fq.ops.RZ(0.2), 0)

    plan = backend._prepare_program(program).plan
    (block,) = [step for step in plan if isinstance(step, PulseBlock)]
    (binding,) = block.noise
    assert np.allclose(binding.local_operator, 0.0)


# --- collapse-operator physics ----------------------------------------------


def test_amplitude_damping_rate_reproduces_exponential_population_decay(model):
    adapter = _adapter(model)
    duration = 5.0
    rate = 0.3
    binding = _amplitude_term((0.0, rate))  # only the 2 -> 1 transition is active
    block = _idle_block(adapter, "q0", duration=duration, noise=(binding,))
    initial = ket2dm(tensor(basis(3, 2), basis(3, 0)))
    context = _evolve(adapter, (block,), _context(adapter, initial))

    population_2 = context.state.ptrace(0).diag()[2].real
    assert population_2 == pytest.approx(np.exp(-rate * duration), abs=2e-4)


def test_amplitude_damping_uses_one_combined_ladder_jump(model):
    adapter = _adapter(model)
    t1 = 10.0
    binding = _amplitude_term((1 / t1, 2 / t1))

    ((jump, ordinal),) = adapter._lindblad_ops(binding)
    assert ordinal == 0
    assert np.allclose(
        jump.full(), np.sqrt(1 / t1) * adapter._local_annihilation.full()
    )


def test_phase_damping_rate_reproduces_exact_coherence_decay(model):
    adapter = _adapter(model)
    duration = 5.0
    rate = 0.2
    binding = _phase_term(rate)
    block = _idle_block(adapter, "q0", duration=duration, noise=(binding,))
    plus = (basis(3, 0) + basis(3, 1)).unit()
    initial = ket2dm(tensor(plus, basis(3, 0)))
    context = _evolve(adapter, (block,), _context(adapter, initial))

    coherence = context.state.ptrace(0).full()[0, 1]
    assert coherence == pytest.approx(0.5 * np.exp(-rate * duration), abs=2e-4)


def test_probability_and_rate_mode_produce_the_same_collapse_coefficient(model):
    adapter = _adapter(model)
    duration = 5.0
    rate = 0.2
    p = PhaseDamping(rate=rate).as_probability(duration)

    from_rate = _phase_term(rate)
    from_p = _phase_term(PhaseDamping(p=p).as_rate(duration))
    (op_rate,) = adapter._lindblad_ops(from_rate)
    (op_p,) = adapter._lindblad_ops(from_p)
    assert np.allclose(op_rate[0].full(), op_p[0].full())


# --- interval scoping and conditional disable -------------------------------


def test_collapse_terms_are_active_only_during_their_own_placed_block(model):
    adapter = _adapter(model)
    rate = 0.3
    binding = _amplitude_term((0.0, rate))
    noisy = _idle_block(adapter, "q0", duration=2.0, noise=(binding,))
    # Both blocks claim "q0", so ASAP scheduling serializes `quiet` right
    # after `noisy` without needing an explicit start time.
    quiet = _idle_block(adapter, "q0", duration=3.0)
    initial = ket2dm(tensor(basis(3, 2), basis(3, 0)))
    context = _evolve(adapter, (noisy, quiet), _context(adapter, initial))

    population_2 = context.state.ptrace(0).diag()[2].real
    # Only `noisy`'s own 2.0ns window contributes decay, not the full 5.0ns run.
    assert population_2 == pytest.approx(np.exp(-rate * 2.0), abs=2e-4)


def test_disabled_conditional_block_contributes_neither_control_nor_noise(model):
    adapter = _adapter(model)
    rate = 0.5
    binding = _amplitude_term((0.0, rate))
    block = _idle_block(
        adapter, "q0", duration=4.0, noise=(binding,), condition=((0, 1),)
    )
    initial = ket2dm(tensor(basis(3, 2), basis(3, 0)))
    context = _context(adapter, initial)
    run = schedule_pulse_run((block,), boundary_time=0.0)

    adapter.evolve(run, context, (False,))  # condition not met: disabled

    assert np.allclose(context.state.full(), initial.full())


def test_overlapping_disjoint_pulses_each_keep_their_own_noise_binding(model):
    adapter = _adapter(model)
    rate_q0 = 0.4
    rate_q1 = 0.1
    binding_q0 = _amplitude_term((0.0, rate_q0))
    binding_q1 = _amplitude_term((0.0, rate_q1), ordinal=1)
    duration = 3.0
    block_q0 = _idle_block(adapter, "q0", duration=duration, noise=(binding_q0,))
    block_q1 = _idle_block(adapter, "q1", duration=duration, noise=(binding_q1,))
    initial = ket2dm(tensor(basis(3, 2), basis(3, 2)))
    context = _evolve(adapter, (block_q0, block_q1), _context(adapter, initial))

    state = context.state
    assert state.ptrace(0).diag()[2].real == pytest.approx(
        np.exp(-rate_q0 * duration), abs=2e-4
    )
    assert state.ptrace(1).diag()[2].real == pytest.approx(
        np.exp(-rate_q1 * duration), abs=2e-4
    )


# --- always-on and operation-scoped bindings compose ------------------------


def test_operation_scoped_and_always_on_noise_accumulate_then_idle_is_global_only(
    model, make_backend
):
    t1 = 10.0
    rate_gate = 0.2
    # T2 = 2*T1 carries zero residual dephasing, keeping this test's population
    # dynamics governed purely by the two T1-type decay channels below.
    noise = NoiseModel()
    noise.add_channel(ThermalRelaxation(t1=t1, t2=2 * t1), targets="q0")
    backend = make_backend(noise)
    adapter = _adapter(
        model,
        always_on_noise=backend._prepare_program(fq.Program(1)).always_on_noise,
    )
    binding = _amplitude_term((0.0, rate_gate))
    gated_duration = 2.0
    idle_duration = 3.0
    gated = _idle_block(adapter, "q0", duration=gated_duration, noise=(binding,))
    idle_after = _idle_block(adapter, "q0", duration=idle_duration)  # global only
    initial = ket2dm(tensor(basis(3, 2), basis(3, 0)))
    context = _evolve(adapter, (gated, idle_after), _context(adapter, initial))

    # The natural annihilation operator's sqrt(n) scaling doubles T1's rate at
    # level 2 (`a|2> = sqrt(2)|1>`), so the gated interval decays at both
    # channels' combined rate, and the idle interval at the global rate alone.
    global_rate = 2.0 / t1
    expected = np.exp(-(global_rate + rate_gate) * gated_duration) * np.exp(
        -global_rate * idle_duration
    )
    population_2 = context.state.ptrace(0).diag()[2].real
    assert population_2 == pytest.approx(expected, abs=2e-4)
