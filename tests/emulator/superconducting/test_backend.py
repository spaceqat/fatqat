"""Physical qutrit boundaries, guards, and dynamic replay tests."""

from math import pi

import numpy as np
import pytest
from qutip import Qobj, basis, ket2dm, tensor

import fatqat as fq
from fatqat._pulse_values import PulseControl
from fatqat._index_allocation import _EngineAllocation
from fatqat._backends.steps import MeasurementStep, ResetStep
from fatqat.emulator.superconducting.backend import TransmonEmulator
from fatqat.emulator._core.engine import PulseEngine, _ShotContext
from fatqat.emulator._core.lindblad import bind_lindblad_operators
from fatqat.noise import default_lindblad_implementation_map
from fatqat.noise.lindblad import resolve_lindblad_operators
from fatqat.emulator.superconducting.qutip_adapter import _TransmonQutipAdapter
from fatqat.emulator.superconducting.target import _TransmonTarget
from fatqat.emulator._core.pulse import (
    PhaseShift,
    PulseBlock,
    PulseDefinition,
)
from fatqat.emulator._core.target import _PreparedControlBinding
from fatqat.waveforms import SampledWaveform
from fatqat.emulator.superconducting.realization import (
    default_transmon_gate_implementation_map,
)
from fatqat.noise import NoiseModel, ThermalRelaxation


@pytest.fixture(name="make_backend")
def make_backend_fixture(model, calibration):
    """Build a backend on the shared model with an optional noise model."""

    def build(noise=None):
        return TransmonEmulator(model, noise=noise)

    return build


def _context(adapter, state, *, classical=(0,), seed=1):
    return _ShotContext(
        state=state,
        classical_memory=list(classical),
        rng=np.random.default_rng(seed),
    )


def _adapter(model, *, kind=_TransmonQutipAdapter, **kwargs):
    target = _TransmonTarget(model)
    return kind(
        target,
        engine_allocation=_EngineAllocation(
            target.device_labels,
            (target.local_dimension,) * len(target.device_labels),
        ),
        **kwargs,
    )


def test_partial_entangled_measurement_collapses_the_physical_posterior(model):
    adapter = _adapter(model)
    ket = (tensor(basis(3, 0), basis(3, 0)) + tensor(basis(3, 1), basis(3, 2))).unit()
    context = _context(adapter, ket2dm(ket), seed=4)
    adapter.execute_boundary(
        MeasurementStep((0,), (0,), reported_digit_maps=((0, 1, 2),)),
        context,
    )

    outcome = context.classical_memory[0]
    assert outcome in (0, 1)
    posterior = context.state.ptrace(1)
    expected = ket2dm(basis(3, 0 if outcome == 0 else 2))
    assert np.allclose(posterior.full(), expected.full())


def test_grouped_measurement_preserves_declared_outcome_order(model):
    adapter = _adapter(model)
    context = _context(
        adapter,
        ket2dm(tensor(basis(3, 1), basis(3, 2))),
        classical=(0, 0),
    )
    adapter.execute_boundary(
        MeasurementStep(
            (1, 0),
            (0, 1),
            reported_digit_maps=((0, 1, 2), (0, 1, 2)),
        ),
        context,
    )
    assert context.classical_memory == [2, 1]


def test_leakage_reports_one_then_confusion_changes_only_classical_value(model):
    adapter = _adapter(model)
    leaked = ket2dm(tensor(basis(3, 2), basis(3, 0)))
    context = _context(adapter, leaked)
    adapter.execute_boundary(
        MeasurementStep(
            (0,),
            (0,),
            reported_digit_maps=((0, 1, 1),),
            confusions=(np.array([[0.0, 1.0], [1.0, 0.0]]),),
        ),
        context,
    )
    assert context.classical_memory == [0]
    assert np.allclose(context.state.full(), leaked.full())


def test_reset_reprepares_only_target_and_guard_can_skip_it(model):
    adapter = _adapter(model)
    entangled = (
        tensor(basis(3, 1), basis(3, 0)) + tensor(basis(3, 2), basis(3, 2))
    ).unit()
    context = _context(adapter, ket2dm(entangled))
    adapter.execute_boundary(ResetStep((0,), condition=((0, 1),)), context)
    assert np.allclose(context.state.full(), ket2dm(entangled).full())

    context.classical_memory[0] = 1
    adapter.execute_boundary(ResetStep((0,), condition=((0, 1),)), context)
    expected_other = (ket2dm(basis(3, 0)) + ket2dm(basis(3, 2))) / 2
    expected = tensor(ket2dm(basis(3, 0)), expected_other)
    assert np.allclose(context.state.full(), expected.full())


def test_confused_reported_value_drives_later_guarded_pulse(make_backend):
    noise = NoiseModel()
    noise.add_readout_error(np.array([[0.0, 1.0], [1.0, 0.0]]), target="q0")
    backend = make_backend(noise)
    program = fq.Program(2, 1)
    program.measure(0, 0)
    program.add(fq.ops.RX(pi), 1, condition=(0, 1))
    result = backend.run(
        program,
        shots=1,
        result_config={"counts": True, "final_state": True},
    ).result()

    assert result.get_counts() == {"1": 1}
    density = Qobj(result.get_density_matrix(), dims=[[3, 3], [3, 3]])
    assert density.ptrace(1).diag()[1].real > 0.8
    assert np.allclose(density.ptrace(0).full(), ket2dm(basis(3, 0)).full())


def test_seeded_dynamic_replay_is_reproducible(make_backend):
    backend = make_backend()
    program = fq.Program(1, 1)
    program.add(fq.ops.RX(pi / 2), 0)
    program.measure(0, 0)
    config = {"counts": True, "final_state": False}
    first = backend.run(
        program, shots=40, simulation_config={"seed": 19}, result_config=config
    ).result()
    second = backend.run(
        program, shots=40, simulation_config={"seed": 19}, result_config=config
    ).result()
    assert first.get_counts_as_tuples() == second.get_counts_as_tuples()


def test_real_boundary_preserves_frame_ledger_for_later_drive(make_backend):
    backend = make_backend()
    with_boundary = fq.Program(2, 1)
    with_boundary.add(fq.ops.RZ(0.3), 0)
    with_boundary.measure(1, 0)
    with_boundary.add(fq.ops.RX(0.7), 0)
    boundary_state = (
        backend.run(
            with_boundary,
            shots=1,
            result_config={"counts": False, "final_state": True},
        )
        .result()
        .get_density_matrix()
    )

    continuous = fq.Program(2)
    continuous.add(fq.ops.RZ(0.3), 0)
    continuous.add(fq.ops.RX(0.7), 0)
    continuous_state = (
        backend.run(
            continuous,
            result_config={"counts": False, "final_state": True},
        )
        .result()
        .get_density_matrix()
    )
    assert np.allclose(boundary_state, continuous_state, atol=2e-7)


def test_reset_and_both_guarded_boundary_outcomes_preserve_later_frame_use(
    make_backend,
):
    backend = make_backend()

    def q0_state(program, *, shots=0):
        density = (
            backend.run(
                program,
                shots=shots,
                result_config={"counts": False, "final_state": True},
            )
            .result()
            .get_density_matrix()
        )
        return Qobj(density, dims=[[3, 3], [3, 3]]).ptrace(0).full()

    continuous = fq.Program(2)
    continuous.add(fq.ops.RZ(0.3), 0)
    continuous.add(fq.ops.RX(0.7), 0)
    expected = q0_state(continuous)

    reset = fq.Program(2)
    reset.add(fq.ops.RZ(0.3), 0)
    reset.add(fq.ops.Reset, 1)
    reset.add(fq.ops.RX(0.7), 0)
    assert np.allclose(q0_state(reset), expected, atol=2e-7)

    for required_digit in (0, 1):
        guarded = fq.Program(2, 1)
        guarded.add(fq.ops.RZ(0.3), 0)
        guarded.measure(1, 0)
        guarded.add(fq.ops.Reset, 1, condition=(0, required_digit))
        guarded.add(fq.ops.RX(0.7), 0)
        assert np.allclose(q0_state(guarded, shots=1), expected, atol=2e-7)


def test_both_guarded_pulse_outcomes_flush_before_later_frame_aware_drive(make_backend):
    backend = make_backend()

    def q0_state(program, *, shots):
        density = (
            backend.run(
                program,
                shots=shots,
                result_config={"counts": False, "final_state": True},
            )
            .result()
            .get_density_matrix()
        )
        return Qobj(density, dims=[[3, 3], [3, 3]]).ptrace(0).full()

    continuous = fq.Program(2)
    continuous.add(fq.ops.RZ(0.3), 0)
    continuous.add(fq.ops.RX(0.7), 0)
    expected = q0_state(continuous, shots=0)

    for required_digit in (0, 1):
        guarded = fq.Program(2, 1)
        guarded.add(fq.ops.RZ(0.3), 0)
        guarded.measure(1, 0)
        guarded.add(fq.ops.RX(0.2), 1, condition=(0, required_digit))
        guarded.add(fq.ops.RX(0.7), 0)
        assert np.allclose(q0_state(guarded, shots=1), expected, atol=2e-7)


class _ExcitedAdapter(_TransmonQutipAdapter):
    def initial_state(self):
        return ket2dm(tensor(basis(3, 2), basis(3, 0)))

    def finish_shot(self, context):
        result = super().finish_shot(context)
        return result, dict(context.frame_angles)


def test_false_guard_reserves_noisy_idle_and_skips_controls_and_frames(model):
    thermal = ThermalRelaxation(t1=5, t2=10)
    adapter = _adapter(
        model,
        kind=_ExcitedAdapter,
        always_on_noise=bind_lindblad_operators(
            resolve_lindblad_operators(
                thermal,
                implementation_map=default_lindblad_implementation_map(),
                physical_dimension=model.physical_dimension,
                duration=None,
            ),
            engine_indices=(0,),
        ),
    )
    target = adapter._target
    frame = model.frame("q0")
    controls = (
        PulseControl(
            model.drive_control("q0"),
            SampledWaveform([0.0, 20.0], [10.0, 10.0]),
        ),
    )
    target_binding = target.bind_control(controls[0].channel)
    bindings = (_PreparedControlBinding("drive", (0,)),)
    block = PulseBlock(
        20.0,
        controls,
        bindings,
        target_binding.claims,
        post_actions=(PhaseShift(frame, 0.7),),
        condition=((0, 1),),
    )
    (outcome,) = PulseEngine(adapter).run(
        (block,), shots=1, n_clbits=1, rng=np.random.default_rng(5)
    )
    shot, frames = outcome
    density = Qobj(shot.final_state, dims=[[3, 3], [3, 3]])
    assert density.ptrace(0).diag()[2].real < 0.1
    assert frames == {}


def test_custom_cz_rule_executes_end_to_end_and_yields_a_valid_physical_state(
    model, calibration
):
    # A minimal custom CZ, registered through PulseImplementationMap instead
    # of the default calibrated recipe: structural model-authored
    # detuning/exchange controls and optional frame actions. The bound target
    # derives occurrence claims during shared preparation.

    def custom_cz(operation, *, device_operands):
        del operation
        first, second = device_operands
        duration = 40.0
        detuning_grid = np.linspace(0.0, duration, 65)
        detuning = np.full_like(detuning_grid, 2 * pi * 0.05)
        exchange_grid = np.linspace(0.0, duration, 65)
        exchange = 0.02 * np.sin(pi * exchange_grid / duration) ** 2
        return PulseDefinition(
            duration,
            (
                PulseControl(
                    model.detuning_control(first),
                    SampledWaveform(detuning_grid, detuning),
                ),
                PulseControl(
                    model.exchange_control(first, second),
                    SampledWaveform(exchange_grid, exchange),
                ),
            ),
            (
                PhaseShift(model.frame(first), 0.1),
                PhaseShift(model.frame(second), 0.05),
            ),
        )

    implementations = default_transmon_gate_implementation_map(
        model=model, calibration=calibration
    )
    implementations.remove(fq.ops.CZ)
    implementations.add(fq.ops.CZ, custom_cz)
    backend = TransmonEmulator(model, gate_implementation_map=implementations)

    program = fq.Program(2)
    program.add(fq.ops.CZ, (0, 1))
    plan = backend._prepare_program(program).plan
    (block,) = plan
    assert block.duration == 40.0
    assert len(block.controls) == 2
    assert {c.channel.kind for c in block.controls} == {"detuning", "exchange"}

    # Stable physical signal without overspecifying solver tolerances: the
    # custom realization must still evolve to a valid (trace-one, Hermitian)
    # physical state, exactly like any built-in gate's realization.
    result = backend.run(
        program, result_config={"counts": False, "final_state": True}
    ).result()
    density = result.get_density_matrix()
    assert np.isclose(np.trace(density).real, 1.0, atol=1e-6)
    assert np.allclose(density, density.conj().T, atol=1e-9)
