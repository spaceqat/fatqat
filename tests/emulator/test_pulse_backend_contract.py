"""Pulse backend shell contracts before physical execution is wired."""

import json
from pathlib import Path

import numpy as np
import pytest

import fatqat as fq
from fatqat.backends.backend_utils import _LoweringContext
from fatqat.emulator.backend import PulseBackend
from fatqat.emulator.engine_contract import PulseSimulationConfig
from fatqat.emulator.superconducting import (
    load_calibration_spec,
    load_physics_model,
)
from fatqat.errors import BackendExecutionError, BackendValidationError
from fatqat.noise import NoiseModel, PhaseDamping
from fatqat.registers import QuantumRegister

_FIXTURES = Path(__file__).parent / "fixtures"


def _backend(noise=None):
    model = load_physics_model(
        json.loads((_FIXTURES / "sc_transmon_exchange.json").read_text())
    )
    calibration = load_calibration_spec(
        json.loads((_FIXTURES / "sc_transmon_exchange_calibration.json").read_text()),
        model,
    )
    return PulseBackend(model, calibration, noise=noise)


def test_auto_configuration_normalizes_to_serial_and_worker_restrictions_raise():
    assert PulseSimulationConfig().parallel_mode == "serial"
    assert PulseSimulationConfig(max_workers=1).parallel_mode == "serial"
    with pytest.raises(BackendValidationError, match="only parallel_mode"):
        PulseSimulationConfig(parallel_mode="multiprocessing")
    with pytest.raises(BackendValidationError, match="max_workers"):
        PulseSimulationConfig(max_workers=2)


def test_run_directly_validates_config_and_executes_ideal_program():
    backend = _backend()
    program = fq.Program(1)
    program.add(fq.ops.RZ(0.2), 0)
    with pytest.raises(BackendValidationError, match="result_config key"):
        backend.run(program, result_config={"density_matrix": True})
    with pytest.raises(BackendValidationError, match="only parallel_mode"):
        backend.run(program, simulation_config={"parallel_mode": "loky"})

    with pytest.raises(BackendValidationError, match="schedule_mode"):
        backend.run(program, simulation_config={"schedule_mode": "SIDEWAYS"})

    result = backend.run(
        program, result_config={"counts": False, "final_state": True}
    ).result()
    assert result.get_density_matrix().shape == (9, 9)
    assert result.metadata["solver"]["frame_convention"].endswith("(Delta_i = 0)")
    assert result.metadata["simulation_config"]["schedule_mode"] == "ASAP"


def test_propagator_applies_the_terminal_frame_by_default():
    backend = _backend()
    angle = 0.2
    program = fq.Program(1)
    program.add(fq.ops.RZ(angle), 0)

    dynamical = backend.propagator(program, apply_final_frame=False)
    complete = backend.propagator(program)
    expected_frame = np.diag(np.exp(1j * angle * np.arange(3)))

    assert np.allclose(dynamical, np.eye(9))
    assert np.allclose(complete, np.kron(expected_frame, np.eye(3)))


def test_propagator_rejects_noncoherent_program_features_and_noise():
    backend = _backend()

    measured = fq.Program(1, 1)
    measured.measure(0, 0)
    with pytest.raises(BackendValidationError, match="measurement"):
        backend.propagator(measured)

    reset = fq.Program(1)
    reset.add(fq.ops.Reset, 0)
    with pytest.raises(BackendValidationError, match="reset"):
        backend.propagator(reset)

    conditioned = fq.Program(1, 1)
    conditioned.add(fq.ops.RX(0.2), 0, condition=(0, 1))
    with pytest.raises(BackendValidationError, match="conditioned"):
        backend.propagator(conditioned)

    noise = NoiseModel()
    noise.add_channel(PhaseDamping(rate=0.001), targets="q0")
    noisy_backend = _backend(noise)
    driven = fq.Program(1)
    driven.add(fq.ops.RX(0.2), 0)
    with pytest.raises(BackendValidationError, match="dissipative"):
        noisy_backend.propagator(driven)


def test_propagator_allows_noise_when_frame_only_plan_has_zero_duration():
    noise = NoiseModel()
    noise.add_channel(PhaseDamping(rate=0.001), targets="q0")
    backend = _backend(noise)
    program = fq.Program(1)
    program.add(fq.ops.RZ(0.2), 0)

    expected = np.kron(np.diag(np.exp(0.2j * np.arange(3))), np.eye(3))
    assert np.allclose(backend.propagator(program), expected)


def test_propagator_validates_its_options_and_empty_program_is_identity(monkeypatch):
    backend = _backend()
    empty = fq.Program(0)

    from fatqat.emulator import qutip_adapter

    def fail_if_runner_is_built(*_args, **_kwargs):
        pytest.fail("an empty propagator constructed a QuTiP runner")

    monkeypatch.setattr(qutip_adapter, "SCQutipAdapter", fail_if_runner_is_built)
    assert np.allclose(backend.propagator(empty), np.eye(9))
    with pytest.raises(BackendValidationError, match="apply_final_frame"):
        backend.propagator(empty, apply_final_frame=1)
    with pytest.raises(BackendValidationError, match="schedule_mode"):
        backend.propagator(empty, schedule_mode="SIDEWAYS")


@pytest.mark.parametrize(
    "options",
    ({"apply_final_frame": 1}, {"schedule_mode": "SIDEWAYS"}),
)
def test_propagator_validates_options_before_lowering(monkeypatch, options):
    backend = _backend()

    def fail_if_lowered(_program):
        pytest.fail("invalid propagator options reached lowering")

    monkeypatch.setattr(backend, "_prepare_program", fail_if_lowered)
    with pytest.raises(BackendValidationError):
        backend.propagator(fq.Program(0), **options)


def test_final_state_measurement_constraint_and_reset_only_determinism_validate_before_execution():
    backend = _backend()
    measured = fq.Program(1, 1)
    measured.measure(0, 0)
    with pytest.raises(BackendValidationError, match="shots == 1"):
        backend.run(measured, shots=2, result_config={"final_state": True})

    reset_only = fq.Program(1)
    reset_only.add(fq.ops.Reset, 0)
    result = backend.run(
        reset_only, shots=0, result_config={"final_state": True}
    ).result()
    assert result.get_density_matrix().shape == (9, 9)


def test_layout_binds_model_ids_while_engine_indices_stay_private():
    backend = _backend()
    program = fq.Program(2)
    program.add(fq.ops.iSwap, (0, 1))
    layout = backend._resolve_resource_layout(program)
    allocation = backend._allocate_engine_indices(program)
    context = _LoweringContext(
        resource_layout=layout, engine_index_allocation=allocation
    )
    plan, _ = backend._lower_program(program, context=context)

    assert layout.device_labels_for(
        (program.quantum_registers[0][0], program.quantum_registers[0][1])
    ) == (
        "q0",
        "q1",
    )
    assert allocation.subsystem_index(program.quantum_registers[0][1]) == 1
    assert plan[0].resource_claims[0] == backend.model.resource("q0")
    assert plan[0].target_indices == (0, 1)


def test_capacity_and_non_qubit_programs_fail_before_execution():
    backend = _backend()
    oversized = fq.Program(3)
    with pytest.raises(BackendValidationError, match="requires 3"):
        backend.run(oversized)

    qutrit = QuantumRegister(1, name="q", dim=3)
    non_qubit = fq.Program([qutrit])
    with pytest.raises(BackendValidationError, match="dimension-two"):
        backend.run(non_qubit)


def test_private_execution_failures_are_sanitized_on_the_returned_job(monkeypatch):
    backend = _backend()
    program = fq.Program(1)

    def fail(*args, **kwargs):
        raise RuntimeError("private qutip solver detail")

    monkeypatch.setattr(backend, "_execute", fail)
    job = backend.run(program)
    assert job.status == "ERROR"
    with pytest.raises(BackendExecutionError, match="Pulse backend execution failed"):
        job.result()
