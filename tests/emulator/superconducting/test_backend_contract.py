"""Pulse backend shell contracts before physical execution is wired."""

from dataclasses import fields

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.emulator import PulseControl, PulseDefinition
from fatqat.emulator import SampledWaveform
from fatqat.emulator.superconducting.backend import TransmonEmulator
from fatqat.emulator.superconducting.model import TransmonModel
from fatqat.emulator._core.backend import _PulseBackend
from fatqat.emulator._core import planning
from fatqat.emulator._core.config import _EmulatorConfig
from fatqat.emulator.superconducting.realization import (
    default_transmon_gate_implementation_map,
)
from fatqat.errors import BackendExecutionError, BackendValidationError
from fatqat.noise import NoiseModel, PhaseDamping, TransitionRelaxation
from fatqat.registers import QuantumRegister
from fatqat.resource_layout import ResourceLayout


@pytest.fixture(name="make_backend")
def make_backend_fixture(model, calibration):
    """Build a backend on the shared model with an optional noise model."""
    del calibration

    def build(noise=None, *, method="density_matrix"):
        return TransmonEmulator(model, method=method, noise=noise)

    return build


def test_constructor_map_type_and_model_ownership_are_exact(model):
    with pytest.raises(BackendValidationError, match="PulseImplementationMap"):
        TransmonEmulator(model, gate_implementation_map=object())
    backend = TransmonEmulator(model)
    assert backend.model is model
    assert type(backend).__bases__ == (_PulseBackend,)
    assert not any(
        name in type(backend).__dict__
        for name in ("run", "propagator", "_prepare_program", "_execute")
    )
    for removed in (
        "_resolve_resource_layout",
        "_direct_control_target_indices",
        "_lower_gate_noise",
        "_engine_index_to_model_ordinal",
        "_background_noise",
        "_physical_dimension",
        "_create_runner_from_bindings",
    ):
        assert not hasattr(backend, removed)
    assert not hasattr(backend, "calibration")
    with pytest.raises(AttributeError):
        backend.model = model


def test_emulator_config_carries_only_the_controls_pulse_execution_honors():
    # The emulator reads exactly two settings. Anything else would be a knob
    # the pulse path silently ignores.
    assert {field.name for field in fields(_EmulatorConfig)} == {
        "seed",
        "schedule_mode",
    }
    assert _EmulatorConfig().schedule_mode == "ASAP"
    assert _EmulatorConfig().seed is None


@pytest.mark.parametrize(
    ("knob", "value"),
    [
        ("shot_parallelism", "threads"),
        ("kernel_parallelism", "threads"),
        ("max_workers", 2),
        ("fusion", True),
    ],
)
def test_matrix_execution_knobs_are_rejected_rather_than_silently_ignored(
    backend, knob, value
):
    # These belong to the matrix engine. The emulator has no engine to steer
    # with them, so accepting one would promise tuning that never happens.
    program = fq.Program(1)
    program.add(ops.RZ(0.2), 0)
    with pytest.raises(BackendValidationError, match=knob):
        backend.run(program, simulation_config={knob: value})


def test_run_directly_validates_config_and_executes_ideal_program(backend):
    program = fq.Program(1)
    program.add(ops.RZ(0.2), 0)
    with pytest.raises(BackendValidationError, match="result_config key"):
        backend.run(program, result_config={"density_matrix": True})

    with pytest.raises(BackendValidationError, match="schedule_mode"):
        backend.run(program, simulation_config={"schedule_mode": "SIDEWAYS"})

    result = backend.run(
        program, result_config={"counts": False, "final_state": True}
    ).result()
    assert result.get_density_matrix().shape == (9, 9)
    assert result.metadata["simulation_config"]["schedule_mode"] == "ASAP"


def test_result_metadata_keeps_common_runtime_facts(backend):
    program = fq.Program(1)
    program.add(ops.RX(0.1), 0)
    statevector_backend = TransmonEmulator(backend.model, method="statevector")
    result = statevector_backend.run(program, result_config={"counts": False}).result()
    assert result.metadata["backend_name"] == "TransmonEmulator"
    assert result.metadata["runtime"] == "qutip"
    assert result.metadata["runtime_details"] == {
        "solver": "sesolve",
        "solver_options": {
            "nsteps": 100000,
            "max_step": 0.078125,
        },
    }
    assert result.metadata["simulation_config"]["schedule_mode"] == "ASAP"
    assert result.metadata["result_config"] == {
        "counts": False,
        "final_state": True,
    }
    assert "solver" not in result.metadata


def test_runtime_details_report_every_solver_used_across_dynamic_regions(
    make_backend,
):
    noise = NoiseModel()
    noise.add(PhaseDamping(rate=0.001), operation=ops.RX)
    backend = make_backend(noise, method="statevector")

    program = fq.Program(2, 1)
    program.add(ops.RX(0.1), 0)
    program.measure(0, 0)
    program.add(ops.RY(0.1), 0)

    result = backend.run(
        program,
        shots=1,
        simulation_config={"seed": 1},
        result_config={"counts": True, "final_state": True},
    ).result()

    assert result.metadata["runtime_details"] == {
        "solver": ("mcsolve", "sesolve"),
        "solver_options": {
            "nsteps": 100000,
            "max_step": 0.078125,
        },
    }


def test_unitary_result_applies_the_terminal_frame(backend):
    angle = 0.2
    program = fq.Program(1)
    program.add(ops.RZ(angle), 0)

    unitary_backend = TransmonEmulator(backend.model, method="unitary")
    complete = unitary_backend.run(program).result().get_unitary()
    expected_frame = np.diag(np.exp(1j * angle * np.arange(3)))

    assert np.allclose(complete, np.kron(expected_frame, np.eye(3)))


def test_unitary_rejects_noncoherent_program_features_and_noise(backend, make_backend):
    unitary_backend = TransmonEmulator(backend.model, method="unitary")
    measured = fq.Program(1, 1)
    measured.measure(0, 0)
    with pytest.raises(BackendValidationError, match="measurement"):
        unitary_backend.run(measured)

    reset = fq.Program(1)
    reset.add(ops.Reset, 0)
    with pytest.raises(BackendValidationError, match="reset"):
        unitary_backend.run(reset)

    conditioned = fq.Program(1, 1)
    conditioned.add(ops.RX(0.2), 0, condition=(0, 1))
    with pytest.raises(BackendValidationError, match="conditioned"):
        unitary_backend.run(conditioned)

    noise = NoiseModel()
    noise.add(PhaseDamping(rate=0.001), targets="q0")
    noisy_backend = make_backend(noise, method="unitary")
    driven = fq.Program(1)
    driven.add(ops.RX(0.2), 0)
    with pytest.raises(BackendValidationError, match="dissipative"):
        noisy_backend.run(driven)


def test_unitary_ignores_inactive_noise_and_applies_the_terminal_frame(make_backend):
    noise = NoiseModel()
    noise.add(PhaseDamping(rate=0.001), targets="q0")
    backend = make_backend(noise, method="unitary")
    program = fq.Program(1)
    program.add(ops.RZ(0.2), 0)

    expected = np.kron(np.diag(np.exp(0.2j * np.arange(3))), np.eye(3))
    assert np.allclose(backend.run(program).result().get_unitary(), expected)


def test_unitary_empty_program_is_full_model_identity(backend):
    empty = fq.Program(0)
    unitary_backend = TransmonEmulator(backend.model, method="unitary")
    assert np.allclose(unitary_backend.run(empty).result().get_unitary(), np.eye(9))


def test_final_state_measurement_constraint_and_reset_only_determinism_validate_before_execution(
    backend,
):
    measured = fq.Program(1, 1)
    measured.measure(0, 0)
    with pytest.raises(BackendValidationError) as exc:
        backend.run(measured, shots=2, result_config={"final_state": True})
    assert str(exc.value) == (
        "density_matrix with physical measurement sampling is only supported "
        "for shots == 1"
    )

    with pytest.raises(BackendValidationError) as exc:
        backend.run(measured, shots=1.5)
    assert str(exc.value) == (
        "shots must be an int when requested results depend on it"
    )

    reset_only = fq.Program(1)
    reset_only.add(ops.Reset, 0)
    result = backend.run(
        reset_only, shots=0, result_config={"final_state": True}
    ).result()
    assert result.get_density_matrix().shape == (9, 9)


def test_invalid_shots_follow_preparation_but_precede_runner(make_backend, monkeypatch):
    noise = NoiseModel()
    noise.add(PhaseDamping(rate=0.01), targets="q0")
    backend = make_backend(noise)
    program = fq.Program(1, 1)
    program.measure(0, 0)

    def runner_must_not_be_built(*_args, **_kwargs):
        raise AssertionError("runner was built before shot validation")

    monkeypatch.setattr(backend, "_create_runner", runner_must_not_be_built)
    with pytest.raises(BackendValidationError) as exc:
        backend.run(program, shots=1.5)
    assert str(exc.value) == (
        "shots must be an int when requested results depend on it"
    )


def test_layout_binds_model_ids_while_engine_indices_stay_private(backend):
    program = fq.Program(2)
    program.add(ops.iSwap, (0, 1))
    prepared = backend._prepare_program(program)
    layout = prepared.resource_layout
    allocation = prepared.engine_allocation
    plan = prepared.plan

    assert layout.device_labels_for(
        (program.quantum_registers[0][0], program.quantum_registers[0][1])
    ) == (
        "q0",
        "q1",
    )
    assert (
        allocation.engine_index(layout.device_label(program.quantum_registers[0][1]))
        == 1
    )
    assert (
        plan[0].resource_claims[0]
        == backend._target.bind_gate_operands(("q0",)).claims[0]
    )
    assert plan[0].target_indices == (0, 1)


def test_common_preparation_owns_target_and_lindblad_binding_once(model, monkeypatch):
    noise = NoiseModel()
    noise.add(PhaseDamping(rate=0.02), operation=ops.RX)
    relaxation = TransitionRelaxation(
        rate=0.01,
        coefficients={(1, 0): 1, (2, 1): np.sqrt(2)},
    )
    noise.add(relaxation, targets="q0")
    noise.add(relaxation, targets="q1")
    backend = TransmonEmulator(model, method="density_matrix", noise=noise)
    program = fq.Program(1)
    program.add(ops.RX(0.3), 0)
    target_binding_calls = 0
    lindblad_targets = []
    bind_program = backend._target.bind_program
    bind_lindblad = planning.bind_lindblad_operators

    def counted_bind_program(source):
        nonlocal target_binding_calls
        target_binding_calls += 1
        return bind_program(source)

    def counted_bind_lindblad(local_operators, *, engine_indices):
        lindblad_targets.append(tuple(engine_indices))
        return bind_lindblad(
            local_operators,
            engine_indices=engine_indices,
        )

    monkeypatch.setattr(backend._target, "bind_program", counted_bind_program)
    monkeypatch.setattr(
        planning,
        "bind_lindblad_operators",
        counted_bind_lindblad,
    )

    prepared = backend._prepare_program(program)

    assert target_binding_calls == 1
    assert prepared.engine_allocation.device_operands == ("q0", "q1")
    assert lindblad_targets == [(0,), (0,), (1,)]


def test_sparse_layout_keeps_unaddressed_transmon_in_full_engine_model(
    model_document,
):
    model_document["system"]["subsystems"].append("q2")
    model_document["parameters"]["subsystems"]["q2"] = {
        "frequency": 5.35,
        "anharmonicity": -0.25,
    }
    model = TransmonModel.from_document(model_document)
    noise = NoiseModel()
    noise.add(PhaseDamping(rate=0.02), targets="q1")
    backend = TransmonEmulator(model, method="density_matrix", noise=noise)
    program = fq.Program(2, 1)
    program.add(ops.RX(np.pi), 0)
    program.measure(0, 0)
    q0, q1 = program.quantum_registers[0][0], program.quantum_registers[0][1]
    layout = ResourceLayout({q0: "q2", q1: "q0"})

    prepared = backend._prepare_program(program, layout)
    result = backend.run(
        program,
        resource_layout=layout,
        shots=1,
        simulation_config={"seed": 17},
        result_config={"counts": True, "final_state": True},
    ).result()

    assert prepared.engine_allocation.device_operands == ("q0", "q1", "q2")
    assert prepared.plan[0].target_indices == (2,)
    assert tuple(term.engine_indices for term in prepared.background_noise) == ((1,),)
    assert result.metadata["state_axes"] == [
        {"device_operand": "q0", "register_ref": q1},
        {"device_operand": "q1", "register_ref": None},
        {"device_operand": "q2", "register_ref": q0},
    ]
    assert all("engine_index" not in axis for axis in result.metadata["state_axes"])
    assert result.get_counts() == {"1": 1}
    density = result.get_density_matrix()
    q2_excited_indices = (1, 2)
    assert sum(np.real(density[index, index]) for index in q2_excited_indices) == (
        pytest.approx(1.0)
    )


def test_capacity_and_non_qubit_programs_fail_before_execution(backend):
    oversized = fq.Program(3)
    with pytest.raises(BackendValidationError, match="requires 3"):
        backend.run(oversized)

    qutrit = QuantumRegister(1, name="q", dim=3)
    non_qubit = fq.Program([qutrit])
    with pytest.raises(BackendValidationError, match="dimension-two"):
        backend.run(non_qubit)


def test_private_execution_failures_are_sanitized_but_keep_the_original_cause(
    backend, monkeypatch
):
    """The public message stays stable; the real failure stays reachable.

    Sanitizing keeps solver internals out of the message a caller sees, but
    discarding the exception entirely left no way to diagnose a failure. The
    original is chained as `__cause__`, so the message contract and
    debuggability hold at the same time.
    """
    program = fq.Program(1)

    def fail(*args, **kwargs):
        raise RuntimeError("private qutip solver detail")

    monkeypatch.setattr(backend, "_execute", fail)
    job = backend.run(program)
    assert job.status == "ERROR"
    with pytest.raises(
        BackendExecutionError, match="Pulse backend execution failed"
    ) as excinfo:
        job.result()

    assert "qutip" not in str(excinfo.value)
    cause = excinfo.value.__cause__
    assert isinstance(cause, RuntimeError)
    assert str(cause) == "private qutip solver detail"


def test_unrealizable_envelope_is_rejected_before_execution(model, calibration):
    """A complex detuning envelope is one user error with one message.

    Preparation asks the bound target to reject this before constructing the
    `PulseBlock`, so the validation error is raised directly by `run()`.
    """

    def complex_detuning_rx(operation, *, device_operands):
        del operation
        (subsystem_id,) = device_operands
        return PulseDefinition(
            10.0,
            (
                PulseControl(
                    model.control.detuning(subsystem_id),
                    SampledWaveform((0.0, 10.0), (0.1 + 0.2j, 0.1 + 0.2j)),
                ),
            ),
        )

    implementations = default_transmon_gate_implementation_map(
        model=model, calibration=calibration
    )
    implementations.remove(ops.RX)
    implementations.add(ops.RX, complex_detuning_rx)
    backend = TransmonEmulator(model, gate_implementation_map=implementations)
    program = fq.Program(1)
    program.add(ops.RX(0.3), 0)

    with pytest.raises(BackendValidationError, match="detuning.*must be real"):
        backend.run(program)


def test_sc_model_is_read_only_and_retained_by_identity(model):
    backend = TransmonEmulator(model)
    assert backend.model is model
    with pytest.raises(AttributeError):
        backend.model = model
