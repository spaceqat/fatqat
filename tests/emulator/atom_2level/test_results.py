"""Ideal two-level result defaults, terminal sampling, and metadata."""

import json
from pathlib import Path

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat._pulse_values import PulseControl
from fatqat.emulator._core.engine import PulseEngine
from fatqat.emulator.atom_2level import (
    Atom2LevelModel,
    Atom2LevelEmulator,
)
from fatqat.errors import (
    BackendValidationError,
    ResultFieldUnavailableError,
    UnsupportedOperationError,
)
from fatqat.noise import ReadoutConfusion
from fatqat.emulator import SampledWaveform

_FIXTURE = Path(__file__).parent / "fixtures" / "atom_2level_reference.json"


@pytest.fixture(name="backend")
def backend_fixture():
    model = Atom2LevelModel.from_document(
        json.loads(_FIXTURE.read_text(encoding="utf-8"))
    )
    return Atom2LevelEmulator(
        model,
        arrangement=fq.emulator.AtomArrangement.rectangular(1, 2, 2.0),
    )


def _pulse_program(*, measured=False, amplitude=np.pi / 2):
    model = Atom2LevelModel.from_document(
        json.loads(_FIXTURE.read_text(encoding="utf-8"))
    )
    program = fq.Program(2, 2 if measured else 0)
    program.add(
        ops.PulseOperation(
            1.0,
            (
                PulseControl(
                    model.control.drive(),
                    SampledWaveform((0.0, 1.0), (amplitude, amplitude)),
                ),
            ),
        )
    )
    if measured:
        program.measure((0, 1), (0, 1))
    return program


def _single_site_case(*, noise=None):
    model = Atom2LevelModel.from_document(
        json.loads(_FIXTURE.read_text(encoding="utf-8"))
    )
    backend = Atom2LevelEmulator(
        model,
        arrangement=fq.emulator.AtomArrangement.chain(1, spacing=2.0),
        noise=noise,
    )
    program = fq.Program(1)
    program.add(
        ops.PulseOperation(
            1.0,
            (
                PulseControl(
                    model.control.drive(),
                    SampledWaveform(
                        (0.0, 1.0),
                        (1j * np.pi / 2, 1j * np.pi / 2),
                    ),
                ),
            ),
        )
    )
    return backend, program


def test_unmeasured_ideal_run_defaults_to_a_pure_statevector(backend):
    result = backend.run(_pulse_program()).result()

    assert result.available_data == frozenset({"statevector"})
    assert result.get_statevector().shape == (4,)
    assert np.linalg.norm(result.get_statevector()) == pytest.approx(1.0)
    assert result.metadata["backend_name"] == "Atom2LevelEmulator"
    assert result.metadata["runtime"] == "qutip"
    assert result.metadata["runtime_details"] == {
        "solver": "sesolve",
        "solver_options": {
            "nsteps": 100000,
            "max_step": 0.5,
        },
    }


def test_terminal_measurement_defaults_to_counts_without_final_state(backend):
    result = backend.run(_pulse_program(measured=True), shots=30).result()

    assert result.available_data == frozenset({"counts"})
    assert sum(result.get_counts().values()) == 30
    with pytest.raises(ResultFieldUnavailableError):
        result.get_statevector()


def test_explicit_counts_without_measurement_keep_all_zero_classical_key(backend):
    program = fq.Program(2)
    result = backend.run(
        program,
        shots=4,
        result_config={"counts": True, "final_state": False},
    ).result()
    assert result.get_counts() == {"": 4}


def test_requested_posterior_state_uses_shared_statevector_shot_validation(backend):
    with pytest.raises(BackendValidationError) as error:
        backend.run(
            _pulse_program(measured=True),
            shots=2,
            result_config={"final_state": True},
        )
    assert str(error.value) == (
        "statevector with physical measurement sampling is only supported "
        "for shots == 1"
    )


def test_terminal_many_shot_counts_are_seeded_and_counts_only_retains_no_arrays(
    backend, monkeypatch
):
    captured = []
    original = PulseEngine.run

    def record(self, *args, **kwargs):
        outcomes = original(self, *args, **kwargs)
        captured.extend(outcomes)
        return outcomes

    monkeypatch.setattr(PulseEngine, "run", record)
    program = _pulse_program(measured=True)
    first = backend.run(
        program, shots=40, simulation_config={"seed": 20260807}
    ).result()
    second = backend.run(
        program, shots=40, simulation_config={"seed": 20260807}
    ).result()

    assert first.get_counts() == second.get_counts()
    assert sum(first.get_counts().values()) == 40
    assert captured
    assert all(outcome.final_state is None for outcome in captured)
    assert all(set(key) <= {"0", "1"} for key in first.get_counts())


def test_one_shot_requested_posterior_is_an_independent_collapsed_ket(backend):
    result = backend.run(
        _pulse_program(measured=True),
        shots=1,
        simulation_config={"seed": 11},
        result_config={"counts": True, "final_state": True},
    ).result()
    state = result.get_statevector()
    counts = result.get_counts()

    assert state.shape == (4,)
    assert np.count_nonzero(np.abs(state) > 1e-10) == 1
    assert np.linalg.norm(state) == pytest.approx(1.0)
    assert counts == {format(int(np.argmax(np.abs(state))), "02b"): 1}


def test_terminal_density_matrix_measurement_samples_and_collapses():
    model = Atom2LevelModel.from_document(
        json.loads(_FIXTURE.read_text(encoding="utf-8"))
    )
    backend = Atom2LevelEmulator(
        model,
        arrangement=fq.emulator.AtomArrangement.rectangular(1, 2, 2.0),
        method="density_matrix",
    )
    program = _pulse_program(measured=True)

    sampled = backend.run(
        program,
        shots=30,
        simulation_config={"seed": 20260830},
    ).result()
    posterior = backend.run(
        program,
        shots=1,
        simulation_config={"seed": 11},
        result_config={"counts": True, "final_state": True},
    ).result()

    assert sum(sampled.get_counts().values()) == 30
    assert all(set(key) <= {"0", "1"} for key in sampled.get_counts())
    state = posterior.get_density_matrix()
    assert state.shape == (4, 4)
    assert np.trace(state) == pytest.approx(1.0)
    assert np.trace(state @ state) == pytest.approx(1.0)
    assert posterior.get_counts() == {
        format(int(np.argmax(np.real(np.diag(state)))), "02b"): 1
    }


def test_estimator_exact_x_and_sampled_diagonal_projector_execution():
    backend, program = _single_site_case()
    estimator = fq.Estimator(backend)

    exact_x = estimator.run(program, fq.Observable([("X", 1.0)])).result()
    diagonal = fq.Observable.from_sparse(
        [("Z", (0,), 1.0), ("ONE", (0,), 0.5)],
        num_qubits=1,
    )
    sampled = estimator.run(
        program,
        diagonal,
        shots=1_000,
        simulation_config={"seed": 19},
    ).result()

    assert exact_x.get_expectation() == pytest.approx(1.0, abs=1e-5)
    assert exact_x.get_standard_error() == 0.0
    assert sampled.get_expectation() == pytest.approx(
        0.25, abs=6 * sampled.get_standard_error()
    )
    assert sampled.metadata == {
        "backend_name": "Atom2LevelEmulator",
        "method": "statevector",
        "runtime": "qutip",
        "runtime_details": {"solver": "sesolve"},
        "shots": 1_000,
    }


def test_estimator_pulse_basis_and_exact_readout_capabilities_raise_synchronously():
    backend, program = _single_site_case()

    with pytest.raises(UnsupportedOperationError, match="X or Y"):
        fq.Estimator(backend).run(
            program,
            fq.Observable([("X", 1.0)]),
            shots=16,
        )

    noise = fq.NoiseModel()
    noise.add(ReadoutConfusion(np.array([[0.0, 1.0], [1.0, 0.0]])))
    noisy_backend, noisy_program = _single_site_case(noise=noise)
    with pytest.raises(UnsupportedOperationError, match="readout confusion"):
        fq.Estimator(noisy_backend).run(
            noisy_program,
            fq.Observable([("Z", 1.0)]),
        )


@pytest.mark.parametrize("method", ("statevector", "density_matrix"))
@pytest.mark.parametrize("shots", (0, 8))
def test_estimator_rejects_pulse_reset_for_all_state_routes(method, shots):
    model = Atom2LevelModel.from_document(
        json.loads(_FIXTURE.read_text(encoding="utf-8"))
    )
    backend = Atom2LevelEmulator(
        model,
        arrangement=fq.emulator.AtomArrangement.chain(1, spacing=2.0),
        method=method,
    )
    program = fq.Program(1)
    program.add(ops.Reset, 0)

    with pytest.raises(BackendValidationError, match="does not support reset"):
        fq.Estimator(backend).run(
            program,
            fq.Observable([("Z", 1.0)]),
            shots=shots,
        )
