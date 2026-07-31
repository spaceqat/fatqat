"""Pulse backend shell contracts before physical execution is wired."""

import json
from pathlib import Path

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
from fatqat.registers import QuantumRegister

_FIXTURES = Path(__file__).parent / "fixtures"


def _backend():
    model = load_physics_model(
        json.loads((_FIXTURES / "sc_transmon_exchange.json").read_text())
    )
    calibration = load_calibration_spec(
        json.loads((_FIXTURES / "sc_transmon_exchange_calibration.json").read_text()),
        model,
    )
    return PulseBackend(model, calibration)


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

    with pytest.raises(BackendValidationError, match="placement_mode"):
        backend.run(program, simulation_config={"placement_mode": "SIDEWAYS"})

    result = backend.run(
        program, result_config={"counts": False, "final_state": True}
    ).result()
    assert result.get_density_matrix().shape == (9, 9)
    assert result.metadata["solver"]["frame_convention"].endswith("(Delta_i = 0)")
    assert result.metadata["simulation_config"]["placement_mode"] == "ASAP"


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
