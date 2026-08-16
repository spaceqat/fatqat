"""Batched seeded two-level trajectory execution contracts."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qutip import Qobj, basis

import fatqat as fq
from fatqat._pulse_values import PulseControl
from fatqat.emulator._core.engine import PulseEngine
from fatqat.emulator._core.scheduling import schedule_pulse_run
from fatqat.emulator.atom_2level import (
    Atom2LevelModel,
    Atom2LevelEmulator,
)
from fatqat.emulator.atom_2level.qutip_adapter import _Atom2LevelQutipAdapter
from fatqat.errors import BackendValidationError
from fatqat.noise import AmplitudeDamping
from fatqat.waveforms import SampledWaveform

_FIXTURE = Path(__file__).parent / "fixtures" / "atom_2level_reference.json"


@pytest.fixture(name="model")
def model_fixture():
    return Atom2LevelModel(json.loads(_FIXTURE.read_text(encoding="utf-8")))


def _backend(model, rate=0.3):
    noise = fq.NoiseModel()
    noise.add(AmplitudeDamping(rate=rate), targets=0)
    return Atom2LevelEmulator(
        model,
        arrangement=fq.AtomArrangement.rectangular(1, 1, 2.0),
        noise=noise,
    )


def _program(*, measured=True, amplitude=1.0, duration=0.6):
    model = Atom2LevelModel(json.loads(_FIXTURE.read_text(encoding="utf-8")))
    program = fq.Program(1, 1 if measured else 0)
    program.add(
        fq.ops.PulseOperation(
            duration,
            (
                PulseControl(
                    model.drive_control(),
                    SampledWaveform((0.0, duration), (amplitude, amplitude)),
                ),
            ),
        )
    )
    if measured:
        program.measure(0, 0)
    return program


def test_one_mcsolve_call_receives_exact_shot_count_seed_order_and_options(
    model, monkeypatch
):
    backend = _backend(model)
    captured = {}

    class SolverResult:
        def __init__(self, final_states, seeds):
            self.runs_final_states = final_states
            self.seeds = [SimpleNamespace(entropy=seed) for seed in seeds]

        @property
        def states(self):
            raise AssertionError("adapter read averaged/intermediate states")

        @property
        def average_final_state(self):
            raise AssertionError("adapter read an averaged final state")

    def fake_mcsolve(
        hamiltonian,
        initial,
        tlist,
        *,
        c_ops,
        ntraj,
        seeds,
        options,
    ):
        captured.update(
            {
                "hamiltonian": hamiltonian,
                "initial": initial,
                "tlist": tlist,
                "c_ops": c_ops,
                "ntraj": ntraj,
                "seeds": tuple(seeds),
                "options": options,
            }
        )
        states = tuple(basis(2, index % 2) for index in range(ntraj))
        return SolverResult(states, seeds)

    monkeypatch.setattr(
        "fatqat.emulator.atom_2level.qutip_adapter.mcsolve", fake_mcsolve
    )
    result = backend.run(_program(), shots=5, simulation_config={"seed": 1234}).result()

    assert captured["ntraj"] == 5
    assert len(captured["seeds"]) == 5
    assert len(set(captured["seeds"])) == 5
    assert captured["options"]["store_final_state"] is True
    assert captured["options"]["keep_runs_results"] is True
    assert captured["options"]["progress_bar"] is False
    assert sum(result.get_counts().values()) == 5
    assert result.metadata["solver"]["solver"] == "mcsolve"


def test_counts_off_trajectory_executes_exactly_one_retained_run(model, monkeypatch):
    backend = _backend(model)
    calls = []

    def fake_mcsolve(*_args, ntraj, seeds, **_kwargs):
        calls.append((ntraj, tuple(seeds)))
        return SimpleNamespace(
            runs_final_states=[basis(2, 0)],
            seeds=[SimpleNamespace(entropy=seeds[0])],
        )

    monkeypatch.setattr(
        "fatqat.emulator.atom_2level.qutip_adapter.mcsolve", fake_mcsolve
    )
    result = backend.run(
        _program(),
        shots=20,
        result_config={"counts": False, "final_state": False},
    ).result()

    assert calls == [(1, calls[0][1])]
    assert len(calls[0][1]) == 1
    assert result.available_data == frozenset()


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


def test_engine_preserves_returned_trajectory_order_and_counts_only_memory(
    model, monkeypatch
):
    backend = _backend(model)
    captured = []
    original = PulseEngine.run_terminal_trajectory_batch

    def fake_mcsolve(*_args, ntraj, seeds, **_kwargs):
        return SimpleNamespace(
            runs_final_states=[basis(2, index % 2) for index in range(ntraj)],
            seeds=[SimpleNamespace(entropy=seed) for seed in seeds],
        )

    def record(self, *args, **kwargs):
        outcomes = original(self, *args, **kwargs)
        captured.extend(outcomes)
        return outcomes

    monkeypatch.setattr(
        "fatqat.emulator.atom_2level.qutip_adapter.mcsolve", fake_mcsolve
    )
    monkeypatch.setattr(PulseEngine, "run_terminal_trajectory_batch", record)
    result = backend.run(_program(), shots=4).result()

    assert result.get_counts_as_tuples() == {(0,): 2, (1,): 2}
    assert [outcome.classical_digits for outcome in captured] == [
        (0,),
        (1,),
        (0,),
        (1,),
    ]
    assert all(outcome.final_state is None for outcome in captured)


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
    density = backend.run(ensemble_program).result().get_density_matrix()
    expected_excited = float(np.real(density[1, 1]))
    observed_excited = first.get_counts().get("1", 0) / 400

    assert first.get_counts() == second.get_counts()
    # With 400 Bernoulli samples the worst-case standard deviation is 0.025;
    # 0.08 is over three sigma and remains stable for the fixed seed.
    assert observed_excited == pytest.approx(expected_excited, abs=0.08)


def test_one_shot_trajectory_final_state_is_reproducible(model):
    backend = _backend(model)
    kwargs = {
        "shots": 1,
        "simulation_config": {"seed": 71},
        "result_config": {"counts": True, "final_state": True},
    }
    first = backend.run(_program(), **kwargs).result()
    second = backend.run(_program(), **kwargs).result()

    assert first.get_counts() == second.get_counts()
    assert first.get_statevector() == pytest.approx(second.get_statevector())
