"""Gate-keyed AmplitudeDamping/PhaseDamping lowered into pulse intervals."""

from dataclasses import dataclass
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
    Channel,
    Depolarizing,
    LindbladImplementationMap,
    NoiseModel,
    PauliChannel,
    PhaseDamping,
    ThermalRelaxation,
)
from fatqat.noise.lindblad import phase_damping_lindblad_rule


@dataclass(frozen=True)
class _DriveBroadening(Channel):
    """Test-only local generator whose physics is not expressed as a rate field."""

    _num_subsystems = 1
    strength: float


@dataclass(frozen=True)
class _TwoBodyGenerator(Channel):
    _num_subsystems = 2
    strength: float


@dataclass(frozen=True)
class _VariableWidthGenerator(Channel):
    strength: float


def _broadening_rule(channel, *, physical_dimension):
    return (
        np.sqrt(channel.strength) * np.diag(np.arange(physical_dimension, dtype=float)),
    )


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
            model.control.drive(subsystem_id),
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


def _qutip_tensor(*canonical_factors):
    """Construct a QuTiP value from canonical least-significant-first factors."""
    return tensor(*reversed(canonical_factors))


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


# --- check_noise_support / capability reporting ----------------------------


def test_pulse_backend_accepts_gate_keyed_rate_and_rejects_probability(make_backend):
    noise = NoiseModel()
    noise.add(AmplitudeDamping(p=(0.01, 0.02)), operation=fq.ops.RX)
    noise.add(PhaseDamping(rate=0.001), operation=fq.ops.RX)
    report = make_backend().check_noise_support(noise)

    assert report.supported is False
    assert report.accepted_sources == ("PhaseDamping(rate)",)
    assert report.rejected_sources == ("AmplitudeDamping(p)",)
    assert "finite probability mode" in report.warnings[0]


def test_pulse_backend_rejects_qutrit_amplitude_damping_with_wrong_arity(
    make_backend,
):
    invalid = NoiseModel()
    invalid.add(AmplitudeDamping(rate=(0.1,)), operation=fq.ops.RX)
    report = make_backend().check_noise_support(invalid)
    assert not report.supported
    assert "arity-1" in report.rejected_sources[0]
    assert "requires 2 damping values" in report.warnings[0]

    valid = NoiseModel()
    valid.add(AmplitudeDamping(rate=(0.1, 0.2)), operation=fq.ops.RX)
    assert make_backend().check_noise_support(valid).supported


def test_pulse_backend_accepts_background_rate_and_rejects_probability(make_backend):
    noise = NoiseModel()
    noise.add(PhaseDamping(rate=0.001), targets="q0")
    noise.add(AmplitudeDamping(p=(0.01, 0.02)), targets="q1")

    report = make_backend().check_noise_support(noise)

    assert report.accepted_sources == ("PhaseDamping(rate, background)",)
    assert report.rejected_sources == ("AmplitudeDamping(p, background)",)
    assert "finite probability mode" in report.warnings[0]


def test_background_rate_lowers_to_the_same_lindblad_term(make_backend):
    noise = NoiseModel()
    noise.add(PhaseDamping(rate=0.0025), targets="q0")
    backend = make_backend(noise)

    bindings = backend._prepare_program(fq.Program(1)).background_noise

    assert len(bindings) == 1
    assert bindings[0].engine_indices == (0,)
    assert np.allclose(bindings[0].local_operator, _phase_term(0.0025).local_operator)


def test_pulse_backend_still_rejects_channel_types_without_a_pulse_implementation(
    make_backend,
):
    noise = NoiseModel()
    noise.add(Depolarizing(p=0.1), operation=fq.ops.RX)
    backend = make_backend()

    report = backend.check_noise_support(noise)
    assert report.supported is False
    assert report.rejected_sources == ("Depolarizing(p)",)

    with pytest.raises(BackendValidationError, match="Depolarizing"):
        make_backend(noise)


def test_pulse_backend_rejects_finite_pauli_channel_without_inferred_generator(
    make_backend,
):
    noise = NoiseModel()
    noise.add(PauliChannel({"X": 0.1}), operation=fq.ops.RX)

    report = make_backend().check_noise_support(noise)

    assert report.rejected_sources == ("PauliChannel",)
    assert "finite-only" in report.warnings[0]


def test_custom_generator_fields_are_interpreted_only_by_the_registered_rule(
    model,
):
    implementations = LindbladImplementationMap()
    implementations.register(_DriveBroadening, _broadening_rule)
    noise = NoiseModel()
    noise.add(_DriveBroadening(strength=0.25), operation=fq.ops.RX)
    backend = TransmonEmulator(
        model,
        noise=noise,
        lindblad_implementation_map=implementations,
    )
    program = fq.Program(1)
    program.add(fq.ops.RX(0.2), 0)

    report = backend.check_noise_support(noise)
    plan = backend._prepare_program(program).plan
    (block,) = [step for step in plan if isinstance(step, PulseBlock)]

    assert report.supported
    assert np.allclose(
        block.noise[0].local_operator,
        _broadening_rule(
            _DriveBroadening(strength=0.25),
            physical_dimension=3,
        )[0],
    )


def test_known_two_body_generator_is_rejected_during_capability_validation(model):
    implementations = LindbladImplementationMap()
    implementations.register(_TwoBodyGenerator, _broadening_rule)
    noise = NoiseModel()
    noise.add(_TwoBodyGenerator(strength=0.1), operation=fq.ops.CZ)
    backend = TransmonEmulator(
        model,
        lindblad_implementation_map=implementations,
    )

    report = backend.check_noise_support(noise)

    assert report.rejected_sources == ("_TwoBodyGenerator",)
    assert "single-subsystem" in report.warnings[0]


def test_variable_width_generator_rejects_a_nonlocal_occurrence_at_lowering(model):
    implementations = LindbladImplementationMap()
    implementations.register(_VariableWidthGenerator, _broadening_rule)
    noise = NoiseModel()
    noise.add(_VariableWidthGenerator(strength=0.1), operation=fq.ops.CZ)
    backend = TransmonEmulator(
        model,
        noise=noise,
        lindblad_implementation_map=implementations,
    )
    program = fq.Program(2)
    program.add(fq.ops.CZ, (0, 1))

    assert backend.check_noise_support(noise).supported
    with pytest.raises(BackendValidationError, match="local to one subsystem"):
        backend._prepare_program(program)


def test_lindblad_implementation_map_declares_pulse_noise_capability(
    model, calibration
):
    implementations = LindbladImplementationMap()
    implementations.register(PhaseDamping, phase_damping_lindblad_rule)
    noise = NoiseModel()
    noise.add(AmplitudeDamping(rate=(0.01, 0.02)), operation=fq.ops.RX)
    backend = TransmonEmulator(
        model,
        lindblad_implementation_map=implementations,
    )

    assert backend.check_noise_support(noise).rejected_sources == (
        "AmplitudeDamping(rate)",
    )

    implementations.register(AmplitudeDamping, phase_damping_lindblad_rule)
    assert backend.check_noise_support(noise).rejected_sources == (
        "AmplitudeDamping(rate)",
    )


# --- lowering: rate resolution ----------------------------------------------


def test_probability_mode_damping_is_rejected_at_construction(make_backend):
    noise = NoiseModel()
    noise.add(AmplitudeDamping(p=(0.01, 0.02)), operation=fq.ops.RX)
    with pytest.raises(BackendValidationError, match="finite probability mode"):
        make_backend(noise)


def test_rate_mode_damping_lowers_unchanged(make_backend):
    noise = NoiseModel()
    noise.add(PhaseDamping(rate=0.0025), operation=fq.ops.RX)
    backend = make_backend(noise)
    program = fq.Program(1)
    program.add(fq.ops.RX(0.3), 0)
    plan = backend._prepare_program(program).plan
    (block,) = [step for step in plan if isinstance(step, PulseBlock)]

    (binding,) = block.noise
    assert np.allclose(binding.local_operator, _phase_term(0.0025).local_operator)


def test_gate_scoped_rate_is_independent_of_the_realized_block_duration(
    model, calibration
):
    custom_duration = 7.0

    def custom_rx(operation, *, device_operands):
        del operation
        (subsystem_id,) = device_operands
        return PulseDefinition(
            custom_duration,
            (
                PulseControl(
                    model.control.drive(subsystem_id),
                    SampledWaveform((0.0, custom_duration), (0.0, 0.0)),
                ),
            ),
        )

    implementations = PulseImplementationMap()
    implementations.add(fq.ops.RX, custom_rx)
    noise = NoiseModel()
    noise.add(AmplitudeDamping(rate=(0.01, 0.02)), operation=fq.ops.RX)
    backend = TransmonEmulator(
        model,
        noise=noise,
        gate_implementation_map=implementations,
    )

    program = fq.Program(1)
    program.add(fq.ops.RX(0.3), 0)
    plan = backend._prepare_program(program).plan
    (block,) = plan

    assert block.duration == custom_duration
    (binding,) = block.noise
    assert np.allclose(
        binding.local_operator,
        _amplitude_term((0.01, 0.02)).local_operator,
    )


def test_generator_rate_on_zero_duration_gate_is_retained_without_conversion(
    make_backend,
):
    noise = NoiseModel()
    noise.add(PhaseDamping(rate=0.1), operation=fq.ops.RZ)
    backend = make_backend(noise)
    program = fq.Program(1)
    program.add(fq.ops.RZ(0.2), 0)

    plan = backend._prepare_program(program).plan
    (block,) = [step for step in plan if isinstance(step, PulseBlock)]
    (binding,) = block.noise
    assert block.duration == 0.0
    assert np.allclose(binding.local_operator, _phase_term(0.1).local_operator)


# --- collapse-operator physics ----------------------------------------------


def test_amplitude_damping_rate_reproduces_exponential_population_decay(model):
    adapter = _adapter(model)
    duration = 5.0
    rate = 0.3
    binding = _amplitude_term((0.0, rate))  # only the 2 -> 1 transition is active
    block = _idle_block(adapter, "q0", duration=duration, noise=(binding,))
    initial = ket2dm(_qutip_tensor(basis(3, 2), basis(3, 0)))
    context = _evolve(adapter, (block,), _context(adapter, initial))

    population_2 = context.state.ptrace(1).diag()[2].real
    assert population_2 == pytest.approx(np.exp(-rate * duration), abs=2e-4)


def test_amplitude_damping_uses_one_combined_ladder_jump(model):
    adapter = _adapter(model)
    t1 = 10.0
    binding = _amplitude_term((1 / t1, 2 / t1))

    ((jump, _factor_index),) = adapter._lindblad_ops(binding)
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
    initial = ket2dm(_qutip_tensor(plus, basis(3, 0)))
    context = _evolve(adapter, (block,), _context(adapter, initial))

    coherence = context.state.ptrace(1).full()[0, 1]
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
    initial = ket2dm(_qutip_tensor(basis(3, 2), basis(3, 0)))
    context = _evolve(adapter, (noisy, quiet), _context(adapter, initial))

    population_2 = context.state.ptrace(1).diag()[2].real
    # Only `noisy`'s own 2.0ns window contributes decay, not the full 5.0ns run.
    assert population_2 == pytest.approx(np.exp(-rate * 2.0), abs=2e-4)


def test_disabled_conditional_block_keeps_only_background_noise_active(model):
    local_rate = 0.5
    background_rate = 0.2
    adapter = _adapter(
        model,
        background_noise=(_amplitude_term((0.0, background_rate)),),
    )
    binding = _amplitude_term((0.0, local_rate))
    block = _idle_block(
        adapter, "q0", duration=4.0, noise=(binding,), condition=((0, 1),)
    )
    initial = ket2dm(_qutip_tensor(basis(3, 2), basis(3, 0)))
    context = _context(adapter, initial)
    run = schedule_pulse_run((block,), boundary_time=0.0)

    adapter.evolve(run, context, (False,))  # condition not met: disabled

    population_2 = context.state.ptrace(1).diag()[2].real
    assert population_2 == pytest.approx(
        np.exp(-background_rate * block.duration),
        abs=2e-4,
    )


def test_overlapping_disjoint_pulses_each_keep_their_own_noise_binding(model):
    adapter = _adapter(model)
    rate_q0 = 0.4
    rate_q1 = 0.1
    binding_q0 = _amplitude_term((0.0, rate_q0))
    binding_q1 = _amplitude_term((0.0, rate_q1), ordinal=1)
    duration = 3.0
    block_q0 = _idle_block(adapter, "q0", duration=duration, noise=(binding_q0,))
    block_q1 = _idle_block(adapter, "q1", duration=duration, noise=(binding_q1,))
    initial = ket2dm(_qutip_tensor(basis(3, 2), basis(3, 2)))
    context = _evolve(adapter, (block_q0, block_q1), _context(adapter, initial))

    state = context.state
    assert state.ptrace(1).diag()[2].real == pytest.approx(
        np.exp(-rate_q0 * duration), abs=2e-4
    )
    assert state.ptrace(0).diag()[2].real == pytest.approx(
        np.exp(-rate_q1 * duration), abs=2e-4
    )


# --- background and operation-scoped bindings compose ------------------------


def test_operation_scoped_and_background_noise_accumulate_then_idle_is_global_only(
    model, make_backend
):
    t1 = 10.0
    rate_gate = 0.2
    # T2 = 2*T1 carries zero residual dephasing, keeping this test's population
    # dynamics governed purely by the two T1-type decay channels below.
    noise = NoiseModel()
    noise.add(ThermalRelaxation(t1=t1, t2=2 * t1), targets="q0")
    backend = make_backend(noise)
    adapter = _adapter(
        model,
        background_noise=backend._prepare_program(fq.Program(1)).background_noise,
    )
    binding = _amplitude_term((0.0, rate_gate))
    gated_duration = 2.0
    idle_duration = 3.0
    gated = _idle_block(adapter, "q0", duration=gated_duration, noise=(binding,))
    idle_after = _idle_block(adapter, "q0", duration=idle_duration)  # global only
    initial = ket2dm(_qutip_tensor(basis(3, 2), basis(3, 0)))
    context = _evolve(adapter, (gated, idle_after), _context(adapter, initial))

    # The natural annihilation operator's sqrt(n) scaling doubles T1's rate at
    # level 2 (`a|2> = sqrt(2)|1>`), so the gated interval decays at both
    # channels' combined rate, and the idle interval at the global rate alone.
    global_rate = 2.0 / t1
    expected = np.exp(-(global_rate + rate_gate) * gated_duration) * np.exp(
        -global_rate * idle_duration
    )
    population_2 = context.state.ptrace(1).diag()[2].real
    assert population_2 == pytest.approx(expected, abs=2e-4)
