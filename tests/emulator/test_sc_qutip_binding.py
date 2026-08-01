"""Private qutip-qip binding and ideal full-qutrit evolution tests."""

import json
from dataclasses import replace
from math import pi
from pathlib import Path

import numpy as np
from qutip import Qobj, basis, ket2dm, mesolve, qeye, tensor
from scipy.interpolate import CubicSpline

import fatqat as fq
from fatqat.backends import MeasurementStep
from fatqat.emulator.backend import PulseBackend
from fatqat.emulator.engine import PulseEngine, _ShotContext
from fatqat.emulator.scheduling import schedule_pulse_run
from fatqat.emulator.qutip_adapter import FRAME_CONVENTION, SCQutipAdapter
from fatqat.emulator.pulse import (
    PhaseShift,
    PulseBlock,
    SampledControl,
)
from fatqat.emulator.superconducting_realization import realize_calibrated_operation
from fatqat.emulator.superconducting import (
    load_calibration_spec,
    load_physics_model,
)
from fatqat.operations import CZ, RZ, iSwap
from fatqat.resource_layout import ResourceLayout

_FIXTURES = Path(__file__).parent / "fixtures"


def _documents():
    model_document = json.loads((_FIXTURES / "sc_transmon_exchange.json").read_text())
    calibration_document = json.loads(
        (_FIXTURES / "sc_transmon_exchange_calibration.json").read_text()
    )
    return model_document, calibration_document


def _model_and_calibration():
    model_document, calibration_document = _documents()
    model = load_physics_model(model_document)
    return model, load_calibration_spec(calibration_document, model)


def _drive_block(
    model,
    subsystem_id,
    *,
    duration=1.0,
    coefficients=(0.1, 0.1),
    tlist=None,
    condition=None,
    post_actions=(),
):
    if tlist is None:
        tlist = (0.0, duration)
    return PulseBlock(
        model,
        duration,
        (
            SampledControl(
                model.drive_control(subsystem_id),
                tlist,
                coefficients,
            ),
        ),
        (model.resource(subsystem_id),),
        post_actions=post_actions,
        condition=condition,
    )


def _context(adapter, state=None):
    return _ShotContext(
        state=adapter.initial_state() if state is None else state,
        classical_memory=[],
        rng=np.random.default_rng(1),
    )


def _evolve(adapter, blocks, context=None, *, boundary=0.0):
    context = _context(adapter) if context is None else context
    run = schedule_pulse_run(blocks, boundary_time=boundary)
    adapter.evolve(run, context, (True,) * len(run.blocks))
    context.time = run.end_time
    return context


def test_child_binding_uses_one_cubic_qip_pulse_and_native_endpoints():
    model, _ = _model_and_calibration()
    adapter = SCQutipAdapter(model)
    child = SampledControl(
        model.drive_control("q0"),
        [0.0, 0.3, 0.7, 1.0],
        [1.0 + 2.0j, 3.0 + 4.0j, -2.0 + 1.0j, 5.0 + 6.0j],
        start_offset=0.5,
    )
    phase = 0.2
    pulse = adapter._bind_child(child, 4.0, {model.frame("q0"): phase})
    expected = np.exp(1j * phase) * child.coefficients

    assert type(pulse).__module__ == "qutip_qip.pulse"
    assert pulse.spline_kind == "cubic"
    assert np.array_equal(pulse.tlist, np.array([4.5, 4.8, 5.2, 5.5]))
    assert np.allclose(pulse.coeff, expected.real)
    assert len(pulse.coherent_noise) == 1
    assert np.allclose(pulse.coherent_noise[0].coeff, expected.imag)

    evolution, collapse = pulse.get_noisy_qobjevo([3, 3])
    assert collapse == []
    assert np.allclose(evolution(4.0).full(), evolution(4.5).full())
    assert np.allclose(evolution(6.0).full(), evolution(5.5).full())
    time = 4.95
    spline = CubicSpline(pulse.tlist, expected)
    x_operator = Qobj(model.annihilation) + Qobj(model.annihilation).dag()
    y_operator = -1j * (Qobj(model.annihilation) - Qobj(model.annihilation).dag())
    expected_hamiltonian = spline(time).real * tensor(x_operator, qeye(3)) + spline(
        time
    ).imag * tensor(y_operator, qeye(3))
    assert np.allclose(evolution(time).full(), expected_hamiltonian.full())


def test_constant_drive_matches_an_independent_full_model_hamiltonian():
    model, _ = _model_and_calibration()
    adapter = SCQutipAdapter(model)
    duration = 0.8
    amplitude = 0.07
    context = _evolve(
        adapter,
        (_drive_block(model, "q0", duration=duration, coefficients=(amplitude,) * 2),),
    )

    annihilation = Qobj(model.annihilation)
    drift = adapter._drift.get_ideal_qobjevo([3, 3])(0.0)
    hamiltonian = drift + amplitude * tensor(annihilation + annihilation.dag(), qeye(3))
    initial = adapter.initial_state()
    unitary = (-1j * hamiltonian * duration).expm()
    expected = unitary * initial * unitary.dag()
    assert np.allclose(context.state.full(), expected.full(), atol=2e-7)


def test_exchange_keeps_both_qutrit_leakage_paths_and_matches_reference():
    model, _ = _model_and_calibration()
    adapter = SCQutipAdapter(model)
    amplitude = 0.12
    duration = 0.4
    exchange = PulseBlock(
        model,
        duration,
        (
            SampledControl(
                model.exchange_control("q0", "q1"),
                [0.0, duration],
                [amplitude, amplitude],
            ),
        ),
        (
            model.resource("q0"),
            model.resource("q1"),
            model.coupling("q0", "q1"),
        ),
    )
    initial = ket2dm(tensor(basis(3, 1), basis(3, 1)))
    context = _evolve(adapter, (exchange,), _context(adapter, initial))

    annihilation = Qobj(model.annihilation)
    exchange_operator = tensor(annihilation.dag(), annihilation) + tensor(
        annihilation, annihilation.dag()
    )
    drift = adapter._drift.get_ideal_qobjevo([3, 3])(0.0)
    unitary = (-1j * (drift + amplitude * exchange_operator) * duration).expm()
    expected = unitary * initial * unitary.dag()
    density = context.state.full()
    assert density[6, 6].real > 1e-4  # |20>
    assert density[2, 2].real > 1e-4  # |02>
    assert np.allclose(density, expected.full(), atol=2e-7)


def test_realized_iswap_matches_the_public_positive_i_phase_convention():
    model, calibration = _model_and_calibration()
    adapter = SCQutipAdapter(model)
    block = realize_calibrated_operation(
        iSwap,
        (model.resource("q0"), model.resource("q1")),
        model=model,
        calibration=calibration,
    )
    ket_00 = tensor(basis(3, 0), basis(3, 0))
    ket_01 = tensor(basis(3, 0), basis(3, 1))
    ket_10 = tensor(basis(3, 1), basis(3, 0))
    initial = ket2dm((ket_00 + ket_01 + 0.3 * ket_10).unit())
    actual = _evolve(adapter, (block,), _context(adapter, initial)).state
    expected = ket2dm((ket_00 + 1j * ket_10 + 0.3j * ket_01).unit())
    assert np.allclose(actual.full(), expected.full(), atol=2e-7)


def test_drift_and_detuning_match_independent_qutrit_phase_and_exchange_facts():
    model, _ = _model_and_calibration()
    adapter = SCQutipAdapter(model)

    duration = 0.17
    ket_00 = tensor(basis(3, 0), basis(3, 0))
    ket_20 = tensor(basis(3, 2), basis(3, 0))
    initial = ket2dm((ket_00 + ket_20).unit())
    idle = _drive_block(model, "q0", duration=duration, coefficients=(0.0, 0.0))
    actual = _evolve(adapter, (idle,), _context(adapter, initial)).state
    phase = np.exp(-1j * 2 * pi * model.subsystems[0].anharmonicity_ghz * duration)
    expected = ket2dm((ket_00 + phase * ket_20).unit())
    assert np.allclose(actual.full(), expected.full(), atol=2e-7)

    residual_document, _ = _documents()
    residual_document["parameters"]["couplings"][0]["residual_exchange"] = 0.04
    residual_model = load_physics_model(residual_document)
    residual_adapter = SCQutipAdapter(residual_model)
    exchange_duration = 1 / (8 * 0.04)
    exchange = _drive_block(
        residual_model,
        "q0",
        duration=exchange_duration,
        coefficients=(0.0, 0.0),
    )
    exchanged = _evolve(
        residual_adapter,
        (exchange,),
        _context(residual_adapter, ket2dm(tensor(basis(3, 1), basis(3, 0)))),
    ).state
    assert np.isclose(exchanged[1, 1].real, 0.5, atol=2e-7)

    detuning = 0.09
    detuned = PulseBlock(
        model,
        duration,
        (
            SampledControl(
                model.detuning_control("q0"),
                [0.0, duration],
                [detuning, detuning],
            ),
        ),
        (model.resource("q0"),),
    )
    actual = _evolve(adapter, (detuned,), _context(adapter, initial)).state
    phase = np.exp(
        -1j * (2 * pi * model.subsystems[0].anharmonicity_ghz + 2 * detuning) * duration
    )
    expected = ket2dm((ket_00 + phase * ket_20).unit())
    assert np.allclose(actual.full(), expected.full(), atol=2e-7)


def test_backend_keeps_model_order_when_program_binds_a_nonprefix_transmon():
    model, calibration = _model_and_calibration()

    class _NonprefixBackend(PulseBackend):
        def _resolve_resource_layout(self, program):
            return ResourceLayout({program.quantum_registers[0][0]: "q1"})

    program = fq.Program(1)
    program.add(fq.ops.RX(pi / 2), 0)
    density = (
        _NonprefixBackend(model, calibration)
        .run(program, result_config={"counts": False, "final_state": True})
        .result()
        .get_density_matrix()
    )
    state = Qobj(density, dims=[[3, 3], [3, 3]])
    assert np.allclose(state.ptrace(0).full(), ket2dm(basis(3, 0)).full())
    assert state.ptrace(1).diag()[0].real < 0.9


def test_drift_covers_leading_internal_and_trailing_idle_intervals():
    model, _ = _model_and_calibration()
    adapter = SCQutipAdapter(model)
    first = _drive_block(
        model,
        "q0",
        duration=1.0,
        coefficients=(0.0, 0.0),
    )
    second = _drive_block(
        model,
        "q1",
        duration=1.0,
        coefficients=(0.0, 0.0),
    )
    first = replace(first, start_time=1.0)
    second = replace(second, start_time=3.0)
    ket = tensor((basis(3, 0) + basis(3, 2)).unit(), basis(3, 0))
    initial = ket2dm(ket)
    context = _context(adapter, initial)
    run = schedule_pulse_run((first, second), boundary_time=0.0)
    adapter.evolve(run, context, (True, True))

    drift = adapter._drift.get_ideal_qobjevo([3, 3])(0.0)
    unitary = (-1j * drift * 4.0).expm()
    assert np.allclose(
        context.state.full(),
        (unitary * initial * unitary.dag()).full(),
        atol=2e-7,
    )


def test_local_frame_fixes_nominal_cz_crossing_but_calibration_remains_data():
    model, calibration = _model_and_calibration()
    adapter = SCQutipAdapter(model)
    recipe = calibration.recipe("cz")["edges"][0]
    assert FRAME_CONVENTION.endswith("(Delta_i = 0)")
    assert recipe["detuning_ghz"] == -model.subsystems[0].anharmonicity_ghz

    drift = adapter._drift.get_ideal_qobjevo([3, 3])(0.0)
    parked = drift + 2 * pi * recipe["detuning_ghz"] * adapter._number[0]
    state_20 = tensor(basis(3, 2), basis(3, 0))
    state_11 = tensor(basis(3, 1), basis(3, 1))
    energy_20 = complex(state_20.dag() * parked * state_20).real
    energy_11 = complex(state_11.dag() * parked * state_11).real
    assert np.isclose(energy_20, energy_11)

    cz = realize_calibrated_operation(
        CZ,
        (model.resource("q0"), model.resource("q1")),
        model=model,
        calibration=calibration,
    )
    assert cz.controls[1].start_offset == recipe["ramp_duration_ns"]


def test_realized_cz_matches_an_independent_synchronized_hamiltonian():
    model, calibration = _model_and_calibration()
    adapter = SCQutipAdapter(model)
    block = realize_calibrated_operation(
        CZ,
        (model.resource("q0"), model.resource("q1")),
        model=model,
        calibration=calibration,
    )
    detuning, exchange = block.controls
    detuning_spline = CubicSpline(detuning.tlist, detuning.coefficients.real)
    exchange_spline = CubicSpline(
        exchange.start_offset + exchange.tlist, exchange.coefficients.real
    )

    def parked_exchange(time, _args=None):
        if time < exchange.start_offset or time > (
            exchange.start_offset + exchange.duration
        ):
            return 0.0
        return float(exchange_spline(time))

    assert parked_exchange(exchange.start_offset / 2) == 0.0
    assert parked_exchange(block.duration - exchange.start_offset / 2) == 0.0
    number = Qobj(model.number)
    annihilation = Qobj(model.annihilation)
    identity = qeye(3)
    drift = sum(
        2
        * pi
        * subsystem.anharmonicity_ghz
        * tensor(
            number * (number - identity) / 2 if ordinal == 0 else identity,
            number * (number - identity) / 2 if ordinal == 1 else identity,
        )
        for ordinal, subsystem in enumerate(model.subsystems)
    )
    exchange_operator = tensor(annihilation.dag(), annihilation) + tensor(
        annihilation, annihilation.dag()
    )
    initial = ket2dm(tensor(basis(3, 1), basis(3, 1)))
    expected = mesolve(
        [
            drift,
            [tensor(number, identity), detuning_spline],
            [exchange_operator, parked_exchange],
        ],
        initial,
        [0.0, block.duration],
        options={"atol": 1e-11, "rtol": 1e-9, "nsteps": 10000},
    ).states[-1]
    actual = _evolve(adapter, (block,), _context(adapter, initial)).state
    assert np.allclose(actual.full(), expected.full(), atol=2e-7)


def test_residual_exchange_is_bound_only_when_declared():
    model_document, calibration_document = _documents()
    ideal_model = load_physics_model(model_document)
    ideal = SCQutipAdapter(ideal_model)
    assert len(ideal._drift.drift_hamiltonians) == len(ideal_model.subsystems)

    model_document["parameters"]["couplings"][0]["residual_exchange"] = 0.01
    residual_model = load_physics_model(model_document)
    load_calibration_spec(calibration_document, residual_model)
    residual = SCQutipAdapter(residual_model)
    assert len(residual._drift.drift_hamiltonians) == len(residual_model.subsystems) + 1


class _NoOpBoundaryAdapter(SCQutipAdapter):
    def execute_boundary(self, step, context):
        del step, context


class _RecordingAdapter(_NoOpBoundaryAdapter):
    def __init__(self, model):
        super().__init__(model)
        self.bound_phases = []

    def _bind_child(self, child, block_start_time, frames):
        if getattr(child.channel, "kind", None) == "drive":
            ordinal = self._model.bind_control(child.channel)
            frame = self._model.frame(self._model.subsystem_ids[ordinal])
            self.bound_phases.append((ordinal, frames.get(frame, 0.0)))
        return super()._bind_child(child, block_start_time, frames)


def test_frame_ledger_survives_boundary_and_respects_post_action_time():
    model, calibration = _model_and_calibration()
    adapter = _RecordingAdapter(model)
    frame = model.frame("q0")
    rz = realize_calibrated_operation(
        RZ(0.2),
        (model.resource("q0"),),
        model=model,
        calibration=calibration,
    )
    overlapping = _drive_block(
        model,
        "q1",
        duration=2.0,
        coefficients=(0.0, 0.0),
        post_actions=(PhaseShift(frame, 0.3),),
    )
    q0_first = _drive_block(model, "q0", coefficients=(0.0, 0.0))
    q0_second = _drive_block(model, "q0", coefficients=(0.0, 0.0))
    q0_after = _drive_block(model, "q0", coefficients=(0.0, 0.0))
    plan = (
        rz,
        MeasurementStep((0,), (0,), reported_digit_maps=((0, 1, 1),)),
        overlapping,
        q0_first,
        q0_second,
        q0_after,
    )
    PulseEngine(adapter).run(plan, shots=1, n_clbits=1, rng=np.random.default_rng(2))

    q0_phases = [phase for ordinal, phase in adapter.bound_phases if ordinal == 0]
    assert np.allclose(q0_phases, [0.2, 0.2, 0.5])


def test_full_model_state_keeps_unused_nonprefix_transmons_in_ground_state():
    model, _ = _model_and_calibration()
    adapter = SCQutipAdapter(model)
    outcomes = PulseEngine(adapter).run(
        (_drive_block(model, "q1"),),
        shots=1,
        n_clbits=0,
        rng=np.random.default_rng(3),
    )
    density = outcomes[0].density_matrix
    assert density.shape == (9, 9)
    physical = Qobj(density, dims=[[3, 3], [3, 3]])
    assert np.allclose(physical.ptrace(0).full(), ket2dm(basis(3, 0)).full())
