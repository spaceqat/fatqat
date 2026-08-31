"""Public Atom2LevelEmulator construction and program-boundary contracts."""

import inspect
import json
from math import inf, nan
from pathlib import Path

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat._pulse_values import PulseControl
from fatqat.emulator.atom_2level import (
    Atom2LevelModel,
    Atom2LevelEmulator,
)
from fatqat.emulator._core.backend import _PulseBackend
from fatqat.emulator._core.pulse import PulseDefinition, PulseImplementationMap
from fatqat.errors import BackendValidationError, UnsupportedOperationError
from fatqat.noise import Depolarizing
from fatqat.emulator import SampledWaveform

_FIXTURE = Path(__file__).parent / "fixtures" / "atom_2level_reference.json"


@pytest.fixture(name="model")
def model_fixture():
    return Atom2LevelModel.from_document(
        json.loads(_FIXTURE.read_text(encoding="utf-8"))
    )


def _backend(
    model,
    *,
    site_count=2,
    method="statevector",
    noise=None,
):
    return Atom2LevelEmulator(
        model,
        arrangement=fq.emulator.AtomArrangement.rectangular(1, site_count, 2.0),
        method=method,
        noise=noise,
    )


def _pulse(duration=1.0, **components):
    model = Atom2LevelModel.from_document(
        json.loads(_FIXTURE.read_text(encoding="utf-8"))
    )
    amplitude = components.pop("amplitude", 1.0)
    phase = components.pop("phase", 0.0)
    detuning = components.pop("detuning", None)
    if components:
        raise AssertionError(f"unknown test pulse components: {tuple(components)}")
    controls = [
        PulseControl(
            model.control.drive(),
            SampledWaveform(
                (0.0, duration),
                (amplitude * np.exp(1j * phase),) * 2,
            ),
        )
    ]
    if detuning is not None:
        controls.append(
            PulseControl(
                model.control.detuning(),
                SampledWaveform((0.0, duration), (detuning, detuning)),
            )
        )
    return ops.PulseOperation(duration, tuple(controls))


def test_public_constructor_retains_model_and_arrangement_ownership(model):
    arrangement = fq.emulator.AtomArrangement.rectangular(1, 2, 2.0)
    backend = Atom2LevelEmulator(model, arrangement=arrangement)
    assert backend.model is model
    assert backend.arrangement is arrangement
    assert not hasattr(backend, "interaction_cutoff")
    assert not hasattr(backend, "calibration")
    assert type(backend).__bases__ == (_PulseBackend,)
    assert not backend._gate_implementation_map.supported_operations()
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


def test_arrangement_is_read_only_and_constructor_types_are_validated(model):
    arrangement = fq.emulator.AtomArrangement.rectangular(1, 2, 2.0)
    backend = Atom2LevelEmulator(model, arrangement=arrangement)
    assert backend.arrangement is arrangement
    with pytest.raises(AttributeError):
        backend.arrangement = fq.emulator.AtomArrangement.rectangular(1, 2, 3.0)

    with pytest.raises(BackendValidationError, match="Atom2LevelModel"):
        Atom2LevelEmulator(object(), arrangement=arrangement)
    with pytest.raises(BackendValidationError, match="AtomArrangement"):
        Atom2LevelEmulator(model, arrangement=object())
    with pytest.raises(BackendValidationError, match="gate_implementation_map"):
        Atom2LevelEmulator(
            model,
            arrangement=arrangement,
            gate_implementation_map=object(),
        )


@pytest.mark.parametrize("cutoff", [True, -1, nan, inf, -inf, "2", 1j, object()])
def test_run_rejects_invalid_interaction_cutoffs(model, cutoff):
    backend = _backend(model)
    with pytest.raises(
        BackendValidationError,
        match="interaction_cutoff must be None or a finite nonnegative real number",
    ):
        backend.run(
            fq.Program(2),
            simulation_config={"interaction_cutoff": cutoff},
        )


@pytest.mark.parametrize(
    ("authored", "normalized"),
    [(None, None), (0, 0.0), (2, 2.0), (2.5, 2.5)],
)
def test_run_normalizes_interaction_cutoff_in_result_metadata(
    model, authored, normalized
):
    backend = _backend(model)
    result = backend.run(
        fq.Program(2),
        simulation_config={"interaction_cutoff": authored},
    ).result()

    assert result.metadata["simulation_config"]["interaction_cutoff"] == normalized


def test_one_backend_applies_interaction_cutoff_independently_per_run(model):
    backend = _backend(model, site_count=3, method="unitary")
    program = fq.Program(3)
    program.add(_pulse(duration=0.05, amplitude=0.0, detuning=0.0))

    nearest = backend.run(
        program,
        simulation_config={"interaction_cutoff": 2.0},
    ).result()
    full = backend.run(
        program,
        simulation_config={"interaction_cutoff": None},
    ).result()
    nearest_again = backend.run(
        program,
        simulation_config={"interaction_cutoff": 2.0},
    ).result()

    nearest_unitary = nearest.get_unitary()
    full_unitary = full.get_unitary()
    assert nearest_unitary[5, 5] == pytest.approx(1.0)
    assert full_unitary[5, 5] != pytest.approx(nearest_unitary[5, 5])
    assert nearest_again.get_unitary() == pytest.approx(nearest_unitary)


def test_removed_interaction_policy_keyword_and_property_are_absent(model):
    arrangement = fq.emulator.AtomArrangement.rectangular(1, 2, 2.0)
    with pytest.raises(TypeError, match="interaction_policy"):
        Atom2LevelEmulator(
            model,
            arrangement=arrangement,
            interaction_policy=None,
        )
    assert not hasattr(
        Atom2LevelEmulator(model, arrangement=arrangement), "interaction_policy"
    )


def _global_gate_map(model, operation=ops.CZ):
    implementations = PulseImplementationMap()

    def global_drive(_operation, *, device_operands):
        del device_operands
        return PulseDefinition(
            0.2,
            (
                PulseControl(
                    model.control.drive(),
                    SampledWaveform((0.0, 0.2), (0.0, 0.0)),
                ),
            ),
        )

    implementations.add(operation, global_drive)
    return implementations


def test_gate_map_is_copied_once_and_explicit_empty_map_stays_empty(model):
    arrangement = fq.emulator.AtomArrangement.rectangular(1, 2, 2.0)
    gate_map = _global_gate_map(model)
    backend = Atom2LevelEmulator(
        model,
        arrangement=arrangement,
        gate_implementation_map=gate_map,
    )

    gate_map.remove(ops.CZ)

    assert backend._gate_implementation_map.supports(ops.CZ)
    empty = Atom2LevelEmulator(
        model,
        arrangement=arrangement,
        gate_implementation_map=PulseImplementationMap(),
    )
    assert not empty._gate_implementation_map.supported_operations()


def test_invalid_attached_noise_rejects_before_target_construction(model, monkeypatch):
    arrangement = fq.emulator.AtomArrangement.rectangular(1, 2, 2.0)
    noise = fq.NoiseModel()
    noise.add(Depolarizing(p=0.1), operation=ops.X)

    def target_must_not_be_built(*_args, **_kwargs):
        raise AssertionError("target was built before noise validation")

    monkeypatch.setattr(
        "fatqat.emulator.atom_2level.backend._Atom2LevelTarget",
        target_must_not_be_built,
    )
    with pytest.raises(BackendValidationError, match="not supported"):
        Atom2LevelEmulator(model, arrangement=arrangement, noise=noise)


def test_custom_global_gate_requires_and_executes_a_whole_arrangement_occurrence(
    model,
):
    arrangement = fq.emulator.AtomArrangement.rectangular(1, 2, 2.0)
    backend = Atom2LevelEmulator(
        model,
        arrangement=arrangement,
        gate_implementation_map=_global_gate_map(model),
    )
    program = fq.Program(2)
    program.add(ops.CZ, (0, 1))

    prepared = backend._prepare_program(program)
    result = backend.run(program).result()

    assert prepared.plan[0].control_bindings[0].engine_indices == (0, 1)
    assert result.get_statevector().shape == (4,)

    narrow = Atom2LevelEmulator(
        model,
        arrangement=arrangement,
        gate_implementation_map=_global_gate_map(model, ops.X),
    )
    narrow_program = fq.Program(2)
    narrow_program.add(ops.X, 0)
    with pytest.raises(BackendValidationError, match="outside its gate occurrence"):
        narrow.run(narrow_program)


def test_backend_exposes_no_legacy_discovery_method(model):
    backend = _backend(model)

    assert not hasattr(backend, "describe_" + "channel")
    assert backend._target.model is model
    assert model.control.drive() == backend.model.control.drive()
    assert model.control.detuning() == backend.model.control.detuning()


def test_program_binding_requires_exact_binary_arrangement_site_count(model):
    backend = _backend(model)
    for program_size in (1, 3):
        with pytest.raises(BackendValidationError, match="exactly one"):
            backend.run(fq.Program(program_size))
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
        (ops.X, 0),
        (ops.Pair, (0, 1)),
        (ops.Put, 0),
    ):
        program = fq.Program(2)
        program.add(operation, targets)
        with pytest.raises(UnsupportedOperationError, match="not supported"):
            backend.run(program)


def test_barriers_are_structural_noops_even_in_terminal_measurement_suffix(model):
    backend = _backend(model)
    program = fq.Program(2, 2)
    program.add(ops.Barrier, (0, 1))
    program.add(_pulse(amplitude=0.0))
    program.add(ops.Barrier, (0, 1))
    program.measure((0, 1), (0, 1))
    program.add(ops.Barrier, (0, 1))

    result = backend.run(program, shots=3).result()

    assert result.get_counts() == {"00": 3}


def test_reset_condition_and_targeted_global_pulse_are_validation_errors(model):
    backend = _backend(model)

    reset = fq.Program(2)
    reset.add(ops.Reset, 0)
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


def test_unitary_accepts_coherent_or_empty_programs_and_rejects_measurement(model):
    backend = _backend(model, method="unitary")
    program = fq.Program(2)
    program.add(_pulse(amplitude=np.pi / 3))

    first = backend.run(program).result().get_unitary()
    second = backend.run(program).result().get_unitary()
    assert first.shape == (4, 4)
    assert first == pytest.approx(second)
    first[0, 0] = 99
    assert second[0, 0] != 99
    assert backend.run(fq.Program(2)).result().get_unitary() == pytest.approx(np.eye(4))

    measured = fq.Program(2, 1)
    measured.measure(0, 0)
    with pytest.raises(BackendValidationError, match="unitary.*measurement"):
        backend.run(measured)


def test_run_exposes_no_solver_mode_keyword(model):
    backend = _backend(model)
    assert "solver" not in inspect.signature(backend.run).parameters
    assert not hasattr(backend, "propagator")
    with pytest.raises(TypeError):
        backend.run(fq.Program(2), solver="sesolve")
