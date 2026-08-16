"""Public Atom2LevelEmulator construction and program-boundary contracts."""

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import fatqat as fq
from fatqat._pulse_values import PulseControl
from fatqat.emulator.atom_2level import (
    Atom2LevelModel,
    Atom2LevelEmulator,
    GridInteractionPolicy,
)
from fatqat.emulator._core.backend import _PulseBackend
from fatqat.emulator._core.pulse import PulseDefinition, PulseImplementationMap
from fatqat.errors import BackendValidationError, UnsupportedOperationError
from fatqat.noise import (
    AmplitudeDamping,
    LindbladImplementationMap,
    PhaseDamping,
    ThermalRelaxation,
)
from fatqat.noise.lindblad import amplitude_damping_lindblad_rule
from fatqat.waveforms import SampledWaveform

_FIXTURE = Path(__file__).parent / "fixtures" / "atom_2level_reference.json"


@pytest.fixture(name="model")
def model_fixture():
    return Atom2LevelModel(json.loads(_FIXTURE.read_text(encoding="utf-8")))


def _backend(model, *, site_count=2, policy=None, noise=None):
    return Atom2LevelEmulator(
        model,
        arrangement=fq.AtomArrangement.rectangular(1, site_count, 2.0),
        interaction_policy=policy,
        noise=noise,
    )


def _pulse(duration=1.0, **components):
    model = Atom2LevelModel(json.loads(_FIXTURE.read_text(encoding="utf-8")))
    amplitude = components.pop("amplitude", 1.0)
    phase = components.pop("phase", 0.0)
    detuning = components.pop("detuning", None)
    if components:
        raise AssertionError(f"unknown test pulse components: {tuple(components)}")
    controls = [
        PulseControl(
            model.drive_control(),
            SampledWaveform(
                (0.0, duration),
                (amplitude * np.exp(1j * phase),) * 2,
            ),
        )
    ]
    if detuning is not None:
        controls.append(
            PulseControl(
                model.detuning_control(),
                SampledWaveform((0.0, duration), (detuning, detuning)),
            )
        )
    return fq.ops.PulseOperation(duration, tuple(controls))


def test_public_constructor_has_only_the_locked_two_level_arguments(model):
    signature = inspect.signature(Atom2LevelEmulator)
    assert tuple(signature.parameters) == (
        "model",
        "arrangement",
        "interaction_policy",
        "noise",
        "gate_implementation_map",
        "lindblad_implementation_map",
    )
    assert signature.parameters["arrangement"].kind is inspect.Parameter.KEYWORD_ONLY
    assert not any(name in signature.parameters for name in ("calibration", "solver"))

    arrangement = fq.AtomArrangement.rectangular(1, 2, 2.0)
    backend = Atom2LevelEmulator(model, arrangement=arrangement)
    assert backend.model is model
    assert backend.arrangement is arrangement
    assert backend.interaction_policy.mode == "nearest_neighbor"
    assert not hasattr(backend, "calibration")
    assert type(backend).__bases__ == (_PulseBackend,)
    assert not backend._gate_implementation_map.supported_operations()
    assert backend._lindblad_implementation_map.supported_channels() == {
        AmplitudeDamping,
        PhaseDamping,
    }
    assert not any(
        name in type(backend).__dict__
        for name in ("run", "propagator", "_prepare_program", "_execute")
    )
    assert not any(
        hasattr(backend, name)
        for name in (
            "_dele" + "gate",
            "_model" + "_value",
            "_arrangement" + "_value",
        )
    )


def test_explicit_policy_identity_and_constructor_types_are_preserved(model):
    arrangement = fq.AtomArrangement.rectangular(1, 2, 2.0)
    policy = GridInteractionPolicy.full_pair()
    backend = Atom2LevelEmulator(
        model, arrangement=arrangement, interaction_policy=policy
    )
    assert backend.arrangement is arrangement
    assert backend.interaction_policy is policy
    with pytest.raises(AttributeError):
        backend.arrangement = fq.AtomArrangement.rectangular(1, 2, 3.0)
    with pytest.raises(AttributeError):
        backend.interaction_policy = GridInteractionPolicy.nearest_neighbor()

    with pytest.raises(BackendValidationError, match="Atom2LevelModel"):
        Atom2LevelEmulator(object(), arrangement=arrangement)
    with pytest.raises(BackendValidationError, match="AtomArrangement"):
        Atom2LevelEmulator(model, arrangement=object())
    with pytest.raises(BackendValidationError, match="GridInteractionPolicy"):
        Atom2LevelEmulator(model, arrangement=arrangement, interaction_policy=object())
    with pytest.raises(BackendValidationError, match="gate_implementation_map"):
        Atom2LevelEmulator(
            model,
            arrangement=arrangement,
            gate_implementation_map=object(),
        )
    with pytest.raises(BackendValidationError, match="lindblad_implementation_map"):
        Atom2LevelEmulator(
            model,
            arrangement=arrangement,
            lindblad_implementation_map=object(),
        )


def _global_gate_map(model, operation=fq.ops.CZ):
    implementations = PulseImplementationMap()

    def global_drive(_operation, *, device_operands):
        del device_operands
        return PulseDefinition(
            0.2,
            (
                PulseControl(
                    model.drive_control(),
                    SampledWaveform((0.0, 0.2), (0.0, 0.0)),
                ),
            ),
        )

    implementations.add(operation, global_drive)
    return implementations


def test_maps_are_copied_once_and_explicit_empty_maps_stay_empty(model):
    arrangement = fq.AtomArrangement.rectangular(1, 2, 2.0)
    gate_map = _global_gate_map(model)
    lindblad_map = LindbladImplementationMap()
    lindblad_map.register(AmplitudeDamping, amplitude_damping_lindblad_rule)
    backend = Atom2LevelEmulator(
        model,
        arrangement=arrangement,
        gate_implementation_map=gate_map,
        lindblad_implementation_map=lindblad_map,
    )

    gate_map.remove(fq.ops.CZ)
    lindblad_map.register(AmplitudeDamping, lambda channel, **kwargs: ())

    assert backend._gate_implementation_map.supports(fq.ops.CZ)
    assert (
        backend._lindblad_implementation_map.get(AmplitudeDamping)
        is amplitude_damping_lindblad_rule
    )
    empty = Atom2LevelEmulator(
        model,
        arrangement=arrangement,
        gate_implementation_map=PulseImplementationMap(),
        lindblad_implementation_map=LindbladImplementationMap(),
    )
    assert not empty._gate_implementation_map.supported_operations()
    assert not empty._lindblad_implementation_map.supported_channels()


def test_invalid_attached_noise_rejects_before_target_construction(model, monkeypatch):
    arrangement = fq.AtomArrangement.rectangular(1, 2, 2.0)
    noise = fq.NoiseModel()
    noise.add(ThermalRelaxation(t1=10.0, t2=15.0), targets=0)

    def target_must_not_be_built(*_args, **_kwargs):
        raise AssertionError("target was built before capability classification")

    monkeypatch.setattr(
        "fatqat.emulator.atom_2level.backend._Atom2LevelTarget",
        target_must_not_be_built,
    )
    with pytest.raises(BackendValidationError, match="not supported"):
        Atom2LevelEmulator(model, arrangement=arrangement, noise=noise)


def test_constructor_classifies_attached_noise_exactly_once(model, monkeypatch):
    arrangement = fq.AtomArrangement.rectangular(1, 2, 2.0)
    noise = fq.NoiseModel()
    noise.add(AmplitudeDamping(rate=0.2), targets=0)
    calls = 0
    classify = Atom2LevelEmulator._classify_noise

    def counted_classify(self, attached_noise):
        nonlocal calls
        calls += 1
        return classify(self, attached_noise)

    monkeypatch.setattr(Atom2LevelEmulator, "_classify_noise", counted_classify)

    Atom2LevelEmulator(model, arrangement=arrangement, noise=noise)

    assert calls == 1


def test_custom_global_gate_requires_and_executes_a_whole_arrangement_occurrence(
    model,
):
    arrangement = fq.AtomArrangement.rectangular(1, 2, 2.0)
    backend = Atom2LevelEmulator(
        model,
        arrangement=arrangement,
        gate_implementation_map=_global_gate_map(model),
    )
    program = fq.Program(2)
    program.add(fq.ops.CZ, (0, 1))

    prepared = backend._prepare_program(program)
    result = backend.run(program).result()

    assert prepared.plan[0].control_bindings[0].engine_indices == (0, 1)
    assert result.get_statevector().shape == (4,)
    assert result.metadata["solver"]["solver"] == "sesolve"

    narrow = Atom2LevelEmulator(
        model,
        arrangement=arrangement,
        gate_implementation_map=_global_gate_map(model, fq.ops.X),
    )
    narrow_program = fq.Program(2)
    narrow_program.add(fq.ops.X, 0)
    with pytest.raises(BackendValidationError, match="outside its gate occurrence"):
        narrow.run(narrow_program)


def test_backend_exposes_no_legacy_discovery_method(model):
    backend = _backend(model)

    assert not hasattr(backend, "describe_" + "channel")
    assert backend._target.model is model
    assert model.drive_control() == backend.model.drive_control()
    assert model.detuning_control() == backend.model.detuning_control()


def test_program_binding_requires_exact_binary_arrangement_cardinality(model):
    backend = _backend(model)
    with pytest.raises(BackendValidationError, match="exactly one"):
        backend.run(fq.Program(1))
    with pytest.raises(BackendValidationError, match="dimension-two"):
        backend.run(fq.Program([fq.QuantumRegister(2, dim=3)]))


def test_ordinary_gates_are_unsupported_before_runner_construction(model, monkeypatch):
    backend = _backend(model)
    monkeypatch.setattr(
        backend,
        "_create_runner",
        lambda *_args, **_kwargs: pytest.fail("runner was constructed"),
    )

    for operation, targets in (
        (fq.ops.X, 0),
        (fq.ops.Rearrange((1, 0)), (0, 1)),
        (fq.ops.Refill, 0),
    ):
        program = fq.Program(2)
        program.add(operation, targets)
        with pytest.raises(UnsupportedOperationError, match="not supported"):
            backend.run(program)


def test_barriers_are_structural_noops_even_in_terminal_measurement_suffix(model):
    backend = _backend(model)
    program = fq.Program(2, 2)
    program.add(fq.ops.Barrier, (0, 1))
    program.add(_pulse(amplitude=0.0))
    program.add(fq.ops.Barrier, (0, 1))
    program.measure((0, 1), (0, 1))
    program.add(fq.ops.Barrier, (0, 1))

    result = backend.run(program, shots=3).result()

    assert result.get_counts() == {"00": 3}


def test_reset_condition_and_targeted_global_pulse_are_validation_errors(model):
    backend = _backend(model)

    reset = fq.Program(2)
    reset.add(fq.ops.Reset, 0)
    with pytest.raises(BackendValidationError, match="reset"):
        backend.run(reset)

    conditioned = fq.Program(2, 1)
    conditioned.add(_pulse(), condition=(0, 1))
    with pytest.raises(BackendValidationError, match="conditioned"):
        backend.run(conditioned)

    targeted = fq.Program(2)
    with pytest.raises(ValueError, match="expects 0 target"):
        targeted.add(_pulse(), 0)


def test_only_a_terminal_measurement_suffix_may_follow_pulses(model):
    backend = _backend(model)
    accepted = fq.Program(2, 2)
    accepted.add(_pulse())
    accepted.measure(0, 0)
    accepted.measure(1, 1)
    assert backend.run(accepted, shots=2).result().get_counts()

    rejected = fq.Program(2, 1)
    rejected.measure(0, 0)
    rejected.add(_pulse())
    with pytest.raises(BackendValidationError, match="terminal measurement suffix"):
        backend.run(rejected)


def test_lowered_measurement_uses_the_binary_digit_map(model):
    backend = _backend(model)
    program = fq.Program(2, 1)
    program.measure(0, 0)
    prepared = backend._prepare_program(program)
    assert prepared.plan[0].reported_digit_maps == ((0, 1),)


def test_propagator_accepts_coherent_or_empty_programs_and_rejects_measurement(model):
    backend = _backend(model)
    program = fq.Program(2)
    program.add(_pulse(amplitude=np.pi / 3))

    first = backend.propagator(program)
    second = backend.propagator(program)
    assert first.shape == (4, 4)
    assert first == pytest.approx(second)
    first[0, 0] = 99
    assert second[0, 0] != 99
    assert backend.propagator(fq.Program(2)) == pytest.approx(np.eye(4))

    measured = fq.Program(2, 1)
    measured.measure(0, 0)
    with pytest.raises(BackendValidationError, match="propagator.*measurement"):
        backend.propagator(measured)


def test_run_and_propagator_expose_no_solver_mode_keyword(model):
    backend = _backend(model)
    assert "solver" not in inspect.signature(backend.run).parameters
    assert "solver" not in inspect.signature(backend.propagator).parameters
    with pytest.raises(TypeError):
        backend.run(fq.Program(2), solver="sesolve")
