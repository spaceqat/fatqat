"""Two-level atom Lindblad support, binding, mode, and ensemble tests."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qutip import Qobj, basis, ket2dm, mesolve

import fatqat as fq
from fatqat._pulse_values import PulseControl
from fatqat.emulator._core.engine import _ShotContext
from fatqat.emulator._core.pulse import PulseDefinition, PulseImplementationMap
from fatqat.emulator._core.scheduling import schedule_pulse_run
from fatqat.emulator.atom_2level import (
    Atom2LevelModel,
    Atom2LevelEmulator,
)
from fatqat.emulator.atom_2level.qutip_adapter import _Atom2LevelQutipAdapter
from fatqat.errors import BackendValidationError
from fatqat.noise import (
    AmplitudeDamping,
    Depolarizing,
    Loss,
    LindbladImplementationMap,
    PhaseDamping,
    ReadoutConfusion,
    ThermalRelaxation,
)
from fatqat.noise.lindblad import (
    amplitude_damping_lindblad_rule,
    phase_damping_lindblad_rule,
)
from fatqat.waveforms import SampledWaveform

from tests.emulator.atom_2level.reference.two_level_hamiltonian import (
    dense_hamiltonian,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "atom_2level_reference.json"


@pytest.fixture(name="model")
def model_fixture():
    return Atom2LevelModel.from_document(
        json.loads(_FIXTURE.read_text(encoding="utf-8"))
    )


def _backend(
    model,
    noise=None,
    *,
    sites=1,
    gate_map=None,
    lindblad_map=None,
):
    return Atom2LevelEmulator(
        model,
        arrangement=fq.emulator.AtomArrangement.rectangular(1, sites, 2.0),
        noise=noise,
        gate_implementation_map=gate_map,
        lindblad_implementation_map=lindblad_map,
    )


def _noise(channel, *, operation=None, targets=None):
    noise = fq.NoiseModel()
    if operation is None and targets is None:
        targets = 0
    noise.add(channel, operation=operation, targets=targets)
    return noise


def _pulse_program(sites=1, *, measured=False, amplitude=0.0, duration=1.0):
    model = Atom2LevelModel.from_document(
        json.loads(_FIXTURE.read_text(encoding="utf-8"))
    )
    program = fq.Program(sites, sites if measured else 0)
    program.add(
        fq.ops.PulseOperation(
            duration,
            (
                PulseControl(
                    model.control.drive(),
                    SampledWaveform((0.0, duration), (amplitude, amplitude)),
                ),
            ),
        )
    )
    if measured:
        program.measure(tuple(range(sites)), tuple(range(sites)))
    return program


def _explicit_damping_map():
    implementations = LindbladImplementationMap()
    implementations.register(
        AmplitudeDamping,
        amplitude_damping_lindblad_rule,
    )
    implementations.register(
        PhaseDamping,
        phase_damping_lindblad_rule,
    )
    return implementations


def _single_site_x_map(model):
    implementations = PulseImplementationMap()

    def realize(_operation, *, device_operands):
        del device_operands
        return PulseDefinition(
            0.5,
            (
                PulseControl(
                    model.control.drive(),
                    SampledWaveform((0.0, 0.5), (0.0, 0.0)),
                ),
            ),
        )

    implementations.add(fq.ops.X, realize)
    return implementations


@pytest.mark.parametrize(
    "channel",
    [
        AmplitudeDamping(rate=0.2),
        PhaseDamping(rate=0.3),
        ThermalRelaxation(t1=10.0, t2=15.0),
        Depolarizing(rate=0.2),
    ],
)
def test_support_accepts_builtin_background_generators(model, channel):
    backend = _backend(model)
    report = backend.check_noise_support(_noise(channel))
    assert report.supported
    mode = "rate, " if hasattr(channel, "rate") else ""
    assert report.accepted_sources == (f"{type(channel).__name__}({mode}background)",)
    assert _backend(model, _noise(channel)).model is model


@pytest.mark.parametrize(
    "noise",
    [
        _noise(AmplitudeDamping(p=0.2)),
        _noise(PhaseDamping(p=0.2)),
        _noise(Depolarizing(p=0.2), operation=fq.ops.X),
        _noise(AmplitudeDamping(rate=0.2), operation=fq.ops.X),
        _noise(AmplitudeDamping(rate=(0.1, 0.2))),
        _noise(Loss(p=0.2), operation=fq.ops.X),
    ],
)
def test_support_rejects_probability_unsupported_scoped_and_wrong_arity_noise(
    model, noise
):
    report = _backend(model).check_noise_support(noise)
    assert not report.supported
    assert report.rejected_sources
    with pytest.raises(BackendValidationError, match="not supported"):
        _backend(model, noise)


def test_support_accepts_binary_and_rejects_nonbinary_readout_confusion(model):
    noise = fq.NoiseModel()
    noise.add(ReadoutConfusion(np.eye(2)))
    report = _backend(model).check_noise_support(noise)
    assert report.supported
    assert report.accepted_sources == ("ReadoutConfusion",)

    invalid = fq.NoiseModel()
    invalid.add(ReadoutConfusion(np.eye(3)))
    invalid_report = _backend(model).check_noise_support(invalid)
    assert not invalid_report.supported
    assert invalid_report.rejected_sources == ("ReadoutConfusion",)


def test_explicit_equivalent_map_enables_rate_operation_scope(model):
    channel = AmplitudeDamping(rate=0.2)
    noise = _noise(channel, operation=fq.ops.X)
    assert not _backend(model).check_noise_support(noise).supported

    explicit = _backend(
        model,
        noise,
        gate_map=_single_site_x_map(model),
        lindblad_map=_explicit_damping_map(),
    )
    program = fq.Program(1)
    program.add(fq.ops.X, 0)
    prepared = explicit._prepare_program(program)

    term = prepared.plan[0].noise[0]
    expected_rate = channel.rate[0]
    assert term.engine_indices == (0,)
    assert abs(term.local_operator[0, 1]) ** 2 == pytest.approx(expected_rate)
    assert explicit.run(program).result().available_data == {"density_matrix"}


@pytest.mark.parametrize(
    "channel",
    (
        AmplitudeDamping(p=(0.1, 0.2)),
        AmplitudeDamping(rate=(0.1, 0.2)),
    ),
)
def test_explicit_map_rejects_wrong_probability_and_rate_arity(model, channel):
    noise = _noise(channel, operation=fq.ops.X)
    backend = _backend(
        model,
        gate_map=_single_site_x_map(model),
        lindblad_map=_explicit_damping_map(),
    )

    report = backend.check_noise_support(noise)

    assert not report.supported
    expected_label = (
        "AmplitudeDamping(p)"
        if channel.p is not None
        else "AmplitudeDamping(rate-arity-2)"
    )
    assert report.rejected_sources == (expected_label,)
    with pytest.raises(BackendValidationError, match="not supported"):
        _backend(
            model,
            noise,
            gate_map=_single_site_x_map(model),
            lindblad_map=_explicit_damping_map(),
        )


def test_explicit_map_keeps_background_rate_and_rejects_probability(model):
    implementations = _explicit_damping_map()
    assert (
        _backend(
            model,
            _noise(PhaseDamping(rate=0.2)),
            lindblad_map=implementations,
        )
        .check_noise_support(_noise(PhaseDamping(rate=0.2)))
        .supported
    )
    with pytest.raises(BackendValidationError, match="not supported"):
        _backend(
            model,
            _noise(PhaseDamping(p=0.2)),
            lindblad_map=implementations,
        )


def test_operation_scoped_terms_reach_the_adapter_time_window(model, monkeypatch):
    noise = _noise(AmplitudeDamping(rate=0.2), operation=fq.ops.X)
    backend = _backend(
        model,
        noise,
        gate_map=_single_site_x_map(model),
        lindblad_map=_explicit_damping_map(),
    )
    program = fq.Program(1)
    program.add(
        fq.ops.PulseOperation(
            0.3,
            (
                PulseControl(
                    model.control.drive(),
                    SampledWaveform((0.0, 0.3), (0.0, 0.0)),
                ),
            ),
        )
    )
    program.add(fq.ops.X, 0)
    captured = []

    def record(*args, c_ops=(), **kwargs):
        captured.append(tuple(c_ops))
        return mesolve(*args, c_ops=c_ops, **kwargs)

    monkeypatch.setattr(
        "fatqat.emulator.atom_2level.qutip_adapter.mesolve",
        record,
    )
    result = backend.run(program).result()

    assert captured and captured[0]
    assert captured[0][0](0.15).norm() == pytest.approx(0.0)
    assert captured[0][0](0.4).norm() > 0.0
    assert captured[0][0](0.8).norm() == pytest.approx(0.0)
    assert result.metadata["solver"]["solver"] == "mesolve"


def test_operation_scoped_terms_reach_terminal_trajectory_solver(model, monkeypatch):
    noise = _noise(AmplitudeDamping(rate=0.2), operation=fq.ops.X)
    backend = _backend(
        model,
        noise,
        gate_map=_single_site_x_map(model),
        lindblad_map=_explicit_damping_map(),
    )
    program = fq.Program(1, 1)
    program.add(fq.ops.X, 0)
    program.measure(0, 0)
    captured = {}

    def record(*_args, c_ops, ntraj, seeds, **_kwargs):
        captured["c_ops"] = tuple(c_ops)
        return SimpleNamespace(
            runs_final_states=[basis(2, 0) for _ in range(ntraj)],
            seeds=[SimpleNamespace(entropy=seed) for seed in seeds],
        )

    monkeypatch.setattr(
        "fatqat.emulator.atom_2level.qutip_adapter.mcsolve",
        record,
    )
    result = backend.run(program, shots=2).result()

    assert len(captured["c_ops"]) == 1
    assert captured["c_ops"][0](0.25).norm() > 0.0
    assert result.metadata["solver"]["solver"] == "mcsolve"


def test_invalid_noise_selector_is_rejected_against_the_run_layout(model):
    backend = _backend(
        model,
        _noise(AmplitudeDamping(rate=0.2), targets=7),
    )
    with pytest.raises(BackendValidationError, match="device resource"):
        backend.run(fq.Program(1))


def test_explicit_background_selectors_bind_independently_to_sites(model):
    per_site = fq.NoiseModel()
    per_site.add(AmplitudeDamping(rate=0.25), targets=0)
    per_site.add(AmplitudeDamping(rate=0.25), targets=1)
    backend = _backend(model, per_site, sites=2)
    program = fq.Program(2)
    terms = backend._prepare_program(program).background_noise
    assert [term.engine_indices for term in terms] == [(0,), (1,)]
    expected = np.asarray([[0.0, np.sqrt(0.25)], [0.0, 0.0]])
    assert all(term.local_operator == pytest.approx(expected) for term in terms)

    targeted = _backend(
        model,
        _noise(PhaseDamping(rate=0.5), targets=(1,)),
        sites=2,
    )
    terms = targeted._prepare_program(program).background_noise
    assert len(terms) == 1
    assert terms[0].engine_indices == (1,)
    assert terms[0].local_operator == pytest.approx(np.diag([0.0, np.sqrt(1.0)]))


def test_logical_one_element_selector_targets_exactly_one_site(model):
    program = fq.Program(2)
    selected = program.quantum_registers[0][0]
    backend = _backend(
        model,
        _noise(AmplitudeDamping(rate=0.2), targets=selected),
        sites=2,
    )
    terms = backend._prepare_program(program).background_noise
    assert [term.engine_indices for term in terms] == [(0,)]


def test_invalid_shots_raise_after_preparation_without_constructing_runner(
    model, monkeypatch
):
    backend = _backend(model, _noise(AmplitudeDamping(rate=0.2)))
    program = fq.Program(1, 1)
    program.measure(0, 0)

    monkeypatch.setattr(
        backend,
        "_create_runner",
        lambda *_args, **_kwargs: pytest.fail("runner was constructed"),
    )
    with pytest.raises(
        BackendValidationError,
        match="shots must be an int when requested results depend on it",
    ):
        backend.run(program, shots=1.5)


@pytest.mark.parametrize("kind", ["amplitude", "phase"])
def test_one_atom_damping_matches_analytic_master_equation(model, kind):
    gamma = 0.4
    channel = (
        AmplitudeDamping(rate=gamma)
        if kind == "amplitude"
        else PhaseDamping(rate=gamma)
    )
    backend = _backend(model, _noise(channel))
    program = _pulse_program(amplitude=0.0, duration=0.8)
    prepared = backend._prepare_program(program)
    adapter = _Atom2LevelQutipAdapter(
        backend._target,
        engine_allocation=prepared.engine_allocation,
        background_noise=prepared.background_noise,
        execution_mode="density_matrix",
    )
    if kind == "amplitude":
        initial = ket2dm(basis(2, 1))
        expected = np.diag([1 - np.exp(-gamma * 0.8), np.exp(-gamma * 0.8)])
    else:
        plus = (basis(2, 0) + basis(2, 1)).unit()
        initial = ket2dm(plus)
        coherence = 0.5 * np.exp(-gamma * 0.8)
        expected = np.asarray([[0.5, coherence], [coherence, 0.5]])
    context = _ShotContext(initial, [], np.random.default_rng(0))
    run = schedule_pulse_run(prepared.plan, boundary_time=0.0)

    adapter.evolve(run, context, (True,))

    assert context.state.full() == pytest.approx(expected, abs=2e-8)


def test_continuous_depolarization_matches_probability_law(model):
    rate = 0.4
    duration = 0.8
    backend = _backend(model, _noise(Depolarizing(rate=rate)))
    result = backend.run(_pulse_program(amplitude=0.0, duration=duration)).result()

    probability = 1.0 - np.exp(-rate * duration)
    expected = (1.0 - probability) * np.diag([1.0, 0.0]) + probability * np.eye(2) / 2
    assert result.get_density_matrix() == pytest.approx(expected, abs=2e-8)


def test_readout_confusion_changes_only_the_reported_digit(model):
    noise = fq.NoiseModel()
    noise.add(ReadoutConfusion(np.array([[0.0, 1.0], [1.0, 0.0]])))
    backend = _backend(model, noise)
    program = fq.Program(1, 1)
    program.measure(0, 0)

    result = backend.run(
        program,
        shots=1,
        result_config={"counts": True, "final_state": True},
    ).result()

    assert result.get_counts() == {"1": 1}
    assert result.get_statevector() == pytest.approx(np.asarray([1.0, 0.0]))


def test_unmeasured_noise_returns_exact_density_matrix(model):
    backend = _backend(model, _noise(AmplitudeDamping(rate=0.2)))
    result = backend.run(_pulse_program(amplitude=1.1, duration=0.5)).result()

    assert result.available_data == frozenset({"density_matrix"})
    density = result.get_density_matrix()
    assert density.shape == (2, 2)
    assert np.trace(density) == pytest.approx(1.0)
    assert result.metadata["solver"]["solver"] == "mesolve"


def test_two_atom_mesolve_matches_independently_assembled_master_equation(model):
    gamma = 0.15
    per_site = fq.NoiseModel()
    per_site.add(AmplitudeDamping(rate=gamma), targets=0)
    per_site.add(AmplitudeDamping(rate=gamma), targets=1)
    backend = _backend(model, per_site, sites=2)
    duration = 0.4
    amplitude = 0.8
    program = _pulse_program(
        sites=2, amplitude=amplitude, duration=duration, measured=False
    )
    actual = backend.run(program).result().get_density_matrix()

    hamiltonian = dense_hamiltonian(
        2,
        amplitude=amplitude,
        detuning=0.0,
        phase=0.0,
        interactions=((0, 1, model.c6_angular_per_us_um6 / 2.0**6),),
    )
    lowering = np.asarray([[0.0, np.sqrt(gamma)], [0.0, 0.0]])
    identity = np.eye(2)
    collapse = [
        Qobj(np.kron(lowering, identity), dims=[[2, 2], [2, 2]]),
        Qobj(np.kron(identity, lowering), dims=[[2, 2], [2, 2]]),
    ]
    initial = ket2dm(Qobj(np.asarray([1.0, 0.0, 0.0, 0.0]), dims=[[2, 2], [1, 1]]))
    expected = mesolve(
        Qobj(hamiltonian, dims=[[2, 2], [2, 2]]),
        initial,
        [0.0, duration],
        c_ops=collapse,
        options={"method": "vern9", "atol": 1e-11, "rtol": 1e-9},
    ).states[-1]

    assert actual == pytest.approx(expected.full(), abs=3e-8)


def test_explicit_counts_without_measurement_does_not_select_mcsolve(
    model, monkeypatch
):
    backend = _backend(model, _noise(AmplitudeDamping(rate=0.2)))

    def mcsolve_must_not_run(*_args, **_kwargs):
        raise AssertionError("mcsolve selected without terminal measurement")

    monkeypatch.setattr(
        "fatqat.emulator.atom_2level.qutip_adapter.mcsolve", mcsolve_must_not_run
    )
    result = backend.run(
        _pulse_program(amplitude=0.0),
        shots=3,
        result_config={"counts": True, "final_state": False},
    ).result()
    assert result.get_counts() == {"": 3}


def test_noisy_measurement_only_program_uses_no_dynamical_solver(model, monkeypatch):
    backend = _backend(model, _noise(AmplitudeDamping(rate=0.2)))
    program = fq.Program(1, 1)
    program.measure(0, 0)

    def solver_must_not_run(*_args, **_kwargs):
        raise AssertionError("a dynamical solver ran at zero elapsed time")

    for name in ("sesolve", "mesolve", "mcsolve"):
        monkeypatch.setattr(
            f"fatqat.emulator.atom_2level.qutip_adapter.{name}", solver_must_not_run
        )
    result = backend.run(program, shots=4).result()

    assert result.get_counts() == {"0": 4}
    assert result.metadata["solver"]["solver"] == "none"


def test_noisy_propagator_allows_empty_identity_but_rejects_elapsed_plan(model):
    backend = _backend(model, _noise(AmplitudeDamping(rate=0.2)))
    assert backend.propagator(fq.Program(1)) == pytest.approx(np.eye(2))

    with pytest.raises(BackendValidationError, match="dissipative Lindblad"):
        backend.propagator(_pulse_program(amplitude=0.0))
