"""Small public contract shared by both pulse-emulator families."""

from __future__ import annotations

import numpy as np
import pytest

import fatqat as fq
from fatqat.emulator._core.engine import PulseEngine
from fatqat.emulator._core.outcome import _PulseShotOutcome
from fatqat.errors import BackendExecutionError, BackendValidationError
from fatqat.job import Job


@pytest.fixture(name="family_backend", params=("superconducting", "atom"))
def family_backend_fixture(request, model, atom_3level_model):
    """Build one backend with the smallest valid two-resource program shape."""
    if request.param == "superconducting":
        return fq.emulator.TransmonEmulator(model)
    return fq.emulator.Atom3LevelEmulator(
        atom_3level_model,
        arrangement=fq.emulator.AtomArrangement.rectangular(1, 2, 2.0),
    )


def _program():
    program = fq.Program(2)
    program.add(fq.ops.RX(0.1), 0)
    return program


def test_both_families_return_an_eager_job_with_public_metadata(family_backend):
    job = family_backend.run(_program(), shots=17)

    assert isinstance(job, Job)
    assert job.result().metadata["backend_name"] in {
        "TransmonEmulator",
        "Atom3LevelEmulator",
    }
    assert job.result().metadata["shots"] == 17


def test_both_families_reject_unknown_simulation_controls(family_backend):
    with pytest.raises(
        BackendValidationError, match="does not support simulation_config"
    ):
        family_backend.run(_program(), simulation_config={"unknown": 1})


def test_both_families_return_a_square_empty_propagator(family_backend):
    propagator = family_backend.propagator(fq.Program(2))

    assert propagator.ndim == 2
    assert propagator.shape[0] == propagator.shape[1]
    assert np.allclose(propagator, np.eye(propagator.shape[0]))


def test_counts_only_execution_does_not_retain_shot_state_arrays(
    family_backend, monkeypatch
):
    original = PulseEngine.run
    captured = []

    def record(self, *args, **kwargs):
        outcomes = original(self, *args, **kwargs)
        captured.extend(outcomes)
        return outcomes

    monkeypatch.setattr(PulseEngine, "run", record)
    program = fq.Program(2, 1)
    program.measure(0, 0)

    result = family_backend.run(program, shots=2).result()

    assert result.get_counts()
    assert result.metadata["shots"] == 2
    assert captured
    assert all(outcome.final_state is None for outcome in captured)


def test_mixed_outcome_kinds_fail_with_a_diagnostic_cause(family_backend, monkeypatch):
    def mixed(*_args, **_kwargs):
        return (
            _PulseShotOutcome(None, "density_matrix", ()),
            _PulseShotOutcome(None, "statevector", ()),
        )

    monkeypatch.setattr(PulseEngine, "run", mixed)
    job = family_backend.run(
        fq.Program(2),
        shots=2,
        result_config={"counts": True, "final_state": False},
    )

    with pytest.raises(BackendExecutionError) as exc:
        job.result()
    assert isinstance(exc.value.__cause__, BackendExecutionError)
    assert "mixed final-state kinds" in str(exc.value.__cause__)


def test_missing_requested_final_state_fails_with_a_diagnostic_cause(
    family_backend, monkeypatch
):
    monkeypatch.setattr(
        PulseEngine,
        "run",
        lambda *_args, **_kwargs: (_PulseShotOutcome(None, "density_matrix", ()),),
    )
    job = family_backend.run(
        fq.Program(2), result_config={"counts": False, "final_state": True}
    )

    with pytest.raises(BackendExecutionError) as exc:
        job.result()
    assert isinstance(exc.value.__cause__, BackendExecutionError)
    assert "omitted a requested final state" in str(exc.value.__cause__)


def test_any_missing_requested_final_state_fails_for_multi_shot_results(
    family_backend, monkeypatch
):
    monkeypatch.setattr(
        PulseEngine,
        "run",
        lambda *_args, **_kwargs: (
            _PulseShotOutcome(None, "density_matrix", ()),
            _PulseShotOutcome(
                np.eye(1),
                "density_matrix",
                (),
            ),
        ),
    )
    job = family_backend.run(
        fq.Program(2),
        shots=2,
        result_config={"counts": True, "final_state": True},
    )

    with pytest.raises(BackendExecutionError) as exc:
        job.result()
    assert isinstance(exc.value.__cause__, BackendExecutionError)
    assert "omitted a requested final state" in str(exc.value.__cause__)
