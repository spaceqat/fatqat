"""Batched seeded two-level trajectory execution contracts."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qutip import Qobj, basis

import fatqat as fq
import fatqat.operations as ops
from fatqat._pulse_values import PulseControl
from fatqat.emulator._core.engine import _ShotContext
from fatqat.emulator._core.scheduling import schedule_pulse_run
from fatqat.emulator.atom_2level import (
    Atom2LevelModel,
    Atom2LevelEmulator,
)
from fatqat.emulator.atom_2level.qutip_adapter import _Atom2LevelQutipAdapter
from fatqat.errors import BackendValidationError
from fatqat.noise import AmplitudeDamping
from fatqat.emulator import SampledWaveform

_FIXTURE = Path(__file__).parent / "fixtures" / "atom_2level_reference.json"


@pytest.fixture(name="model")
def model_fixture():
    return Atom2LevelModel.from_document(
        json.loads(_FIXTURE.read_text(encoding="utf-8"))
    )


def _backend(model, rate=0.3, *, method="statevector"):
    noise = fq.NoiseModel()
    noise.add(AmplitudeDamping(rate=rate), targets=0)
    return Atom2LevelEmulator(
        model,
        arrangement=fq.emulator.AtomArrangement.rectangular(1, 1, 2.0),
        method=method,
        noise=noise,
    )


def _program(*, measured=True, amplitude=1.0, duration=0.6):
    model = Atom2LevelModel.from_document(
        json.loads(_FIXTURE.read_text(encoding="utf-8"))
    )
    program = fq.Program(1, 1 if measured else 0)
    program.add(
        ops.PulseOperation(
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
        program.measure(0, 0)
    return program


@pytest.mark.parametrize("failure", ["none", "count", "nonket", "reordered"])
def test_retained_trajectory_diagnostics_are_explicit(model, monkeypatch, failure):
    backend = _backend(model)
    program = _program()
    prepared = backend._prepare_program(program)
    adapter = _Atom2LevelQutipAdapter(
        backend._target,
        engine_allocation=prepared.engine_allocation,
        background_noise=prepared.background_noise,
        execution_mode="trajectory",
    )
    run = schedule_pulse_run((prepared.plan[0],), boundary_time=0.0)
    seeds = (11, 22)

    def fake_mcsolve(*_args, **_kwargs):
        states = [basis(2, 0), basis(2, 1)]
        returned_seeds = [SimpleNamespace(entropy=11), SimpleNamespace(entropy=22)]
        if failure == "none":
            states = None
        elif failure == "count":
            states = states[:1]
        elif failure == "nonket":
            states[1] = Qobj(np.eye(2))
        else:
            returned_seeds.reverse()
        return SimpleNamespace(runs_final_states=states, seeds=returned_seeds)

    monkeypatch.setattr(
        "fatqat.emulator.atom_2level.qutip_adapter.mcsolve", fake_mcsolve
    )
    messages = {
        "none": "unavailable",
        "count": "wrong number",
        "nonket": "must all be kets",
        "reordered": "seed order",
    }
    with pytest.raises(BackendValidationError, match=messages[failure]):
        adapter.run_trajectory_batch(run, ntraj=2, seeds=seeds)


def test_real_trajectory_runs_are_reproducible_and_converge_to_mesolve(model):
    backend = _backend(model, rate=0.4)
    program = _program(amplitude=np.pi, duration=1.0)
    first = backend.run(
        program, shots=400, simulation_config={"seed": 20260807}
    ).result()
    second = backend.run(
        program, shots=400, simulation_config={"seed": 20260807}
    ).result()

    ensemble_program = _program(measured=False, amplitude=np.pi, duration=1.0)
    density = (
        _backend(model, rate=0.4, method="density_matrix")
        .run(ensemble_program)
        .result()
        .get_density_matrix()
    )
    expected_excited = float(np.real(density[1, 1]))
    observed_excited = first.get_counts().get("1", 0) / 400

    assert first.get_counts() == second.get_counts()
    # With 400 Bernoulli samples the worst-case standard deviation is 0.025;
    # 0.08 is over three sigma and remains stable for the fixed seed.
    assert observed_excited == pytest.approx(expected_excited, abs=0.08)


def test_one_shot_regional_trajectory_continues_and_is_reproducible(model):
    backend = _backend(model)
    prepared = backend._prepare_program(
        _program(measured=False, amplitude=0.8, duration=0.4)
    )

    def execute():
        adapter = _Atom2LevelQutipAdapter(
            backend._target,
            engine_allocation=prepared.engine_allocation,
            background_noise=prepared.background_noise,
            execution_mode="trajectory",
        )
        context = _ShotContext(
            adapter.initial_state(),
            [],
            np.random.default_rng(71),
        )
        for boundary_time in (0.0, 0.4):
            run = schedule_pulse_run(
                prepared.plan,
                boundary_time=boundary_time,
            )
            adapter.evolve(run, context, (True,))
            context.time = run.end_time
        return adapter.finish_shot(context).final_state

    first = execute()
    second = execute()

    assert first == pytest.approx(second)
    assert np.linalg.norm(first) == pytest.approx(1.0)
