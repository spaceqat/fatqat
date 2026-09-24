"""Tests for the private QEC17 gate-level ASAP scheduler."""

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops

from fatqat._backends.view_normalization import _break_grouped_operations
from fatqat.errors import BackendValidationError
from fatqat.noise import AmplitudeDamping, NoiseModel
from fatqat.operations import Measurement, Operation
from fatqat.program import Program, _AppliedOperation
from fatqat.resource_layout import ResourceLayout
from fatqat.simulator import fake_superconducting as sc
from fatqat.simulator._qec17_scheduling import (
    _QEC17_TICK_DURATION_SECONDS,
    _schedule_qec17_asap,
)


def _duration_ticks(operation: Operation) -> int:
    if operation is ops.CZ:
        return 2
    if type(operation) is ops.RZ:
        return 0
    return 1


def _signature(
    instructions: tuple[_AppliedOperation | Measurement, ...],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(
        (
            "Measurement" if isinstance(step, Measurement) else step.operation.name,
            tuple(target.index for target in step.targets),
        )
        for step in instructions
    )


def test_asap_aligns_multi_qubit_gate_without_mutating_program():
    program = Program(2)
    program.add(ops.X, 0)
    program.add(ops.Y, 0)
    program.add(ops.H, 1)
    program.add(ops.CZ, (0, 1))
    original = program._instructions

    scheduled = _schedule_qec17_asap(
        _break_grouped_operations(original),
        program.quantum_registers[0],
        _duration_ticks,
    )

    assert _signature(scheduled) == (
        ("X", (0,)),
        ("Y", (0,)),
        ("H", (1,)),
        ("I", (1,)),
        ("CZ", (0, 1)),
    )
    assert program._instructions is original
    assert _signature(program._instructions) == (
        ("X", (0,)),
        ("Y", (0,)),
        ("H", (1,)),
        ("CZ", (0, 1)),
    )


def test_barrier_synchronizes_only_targets_and_final_sync_covers_all_qubits():
    program = Program(3)
    qubits = program.quantum_registers[0]
    program.add(ops.X, 0)
    program.add(ops.Barrier, (qubits[0], qubits[1]))
    program.add(ops.Y, 1)

    scheduled = _schedule_qec17_asap(
        _break_grouped_operations(program._instructions),
        qubits,
        _duration_ticks,
    )

    assert _signature(scheduled) == (
        ("X", (0,)),
        ("I", (1,)),
        ("Y", (1,)),
        ("I", (0,)),
        ("I", (2,)),
        ("I", (2,)),
    )
    assert all(
        not (
            isinstance(step, _AppliedOperation)
            and isinstance(step.operation, type(ops.Barrier))
        )
        for step in scheduled
    )


def test_zero_duration_operation_does_not_advance_its_qubit():
    program = Program(2)
    program.add(ops.RZ(0.25), 0)
    program.add(ops.X, 1)

    scheduled = _schedule_qec17_asap(
        program._instructions,
        program.quantum_registers[0],
        _duration_ticks,
    )

    assert _signature(scheduled) == (
        ("RZ", (0,)),
        ("X", (1,)),
        ("I", (0,)),
    )
    assert _QEC17_TICK_DURATION_SECONDS == 20e-9


def test_user_authored_identity_already_fills_the_idle_tick():
    program = Program(2)
    program.add(ops.X, 0)
    program.add(ops.I, 1)

    scheduled = _schedule_qec17_asap(
        program._instructions,
        program.quantum_registers[0],
        _duration_ticks,
    )

    assert _signature(scheduled) == (("X", (0,)), ("I", (1,)))


def test_measurement_is_a_full_width_scheduling_boundary():
    program = Program(2, 1)
    program.add(ops.X, 0)
    program.measure(1, 0)
    program.add(ops.Y, 1)

    scheduled = _schedule_qec17_asap(
        program._instructions,
        program.quantum_registers[0],
        _duration_ticks,
    )

    assert _signature(scheduled) == (
        ("X", (0,)),
        ("I", (1,)),
        ("Measurement", (1,)),
        ("Y", (1,)),
        ("I", (0,)),
    )


def test_reset_is_a_full_width_scheduling_boundary():
    program = Program(2)
    program.add(ops.X, 0)
    program.add(ops.Reset, 1)
    program.add(ops.Y, 1)

    scheduled = _schedule_qec17_asap(
        program._instructions,
        program.quantum_registers[0],
        _duration_ticks,
    )

    assert _signature(scheduled) == (
        ("X", (0,)),
        ("I", (1,)),
        ("Reset", (1,)),
        ("Y", (1,)),
        ("I", (0,)),
    )


def test_condition_is_rejected_instead_of_silently_losing_idle_noise():
    program = Program(2, 1)
    program.add(ops.X, 0)
    program.add(ops.Y, 1, condition=(0, 1))
    with pytest.raises(BackendValidationError, match="classical conditions"):
        _schedule_qec17_asap(
            program._instructions, program.quantum_registers[0], _duration_ticks
        )


def test_barrier_condition_is_ignored_without_disabling_scheduling():
    program = Program(2, 1)
    program.add(ops.X, 0)
    program.add(ops.Barrier, (0, 1), condition=(0, 1))
    program.add(ops.Y, 1)

    scheduled = _schedule_qec17_asap(
        program._instructions,
        program.quantum_registers[0],
        _duration_ticks,
    )

    assert _signature(scheduled) == (
        ("X", (0,)),
        ("I", (1,)),
        ("Y", (1,)),
        ("I", (0,)),
    )


def test_unknown_duration_is_rejected_instead_of_losing_idle_noise():
    program = Program(2)
    program.add(ops.X, 0)
    program.add(ops.SX, 1)
    with pytest.raises(BackendValidationError, match="no duration for SX"):
        _schedule_qec17_asap(
            program._instructions,
            program.quantum_registers[0],
            lambda operation: None if operation is ops.SX else 1,
        )


@pytest.mark.parametrize("duration", [-1, 0.5, True])
def test_rejects_invalid_operation_duration(duration):
    program = Program(1)
    program.add(ops.X, 0)

    with pytest.raises((TypeError, ValueError), match="non-negative integer"):
        _schedule_qec17_asap(
            program._instructions,
            program.quantum_registers[0],
            lambda _operation: duration,
        )


def test_qec17_native_duration_table_uses_20_ns_ticks():
    assert sc._qec17_operation_duration_ticks(ops.X) == 1
    assert sc._qec17_operation_duration_ticks(type(ops.X)()) == 1
    assert sc._qec17_operation_duration_ticks(type(ops.I)()) == 1
    assert sc._qec17_operation_duration_ticks(ops.SU2(np.eye(2))) == 1
    assert sc._qec17_operation_duration_ticks(ops.RZ(0.25)) == 0
    assert sc._qec17_operation_duration_ticks(ops.Z) == 0
    assert sc._qec17_operation_duration_ticks(ops.CZ) == 2
    assert sc._qec17_operation_duration_ticks(type(ops.CZ)()) == 2
    assert sc._qec17_operation_duration_ticks(ops.SX) is None


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
def test_noisy_backend_applies_implicit_idle_without_mutating_program(runtime):

    noise = NoiseModel()
    noise.add(AmplitudeDamping(p=1.0), operation=ops.I)
    backend = sc.SCQubitQEC17Simulator(method="DM", runtime=runtime, noise=noise)
    program = Program(2)
    program.add(ops.X, 0)
    original = program._instructions

    # Public basis order is q0,q1. Start in |01>; X runs on q0 while q1
    # receives the automatically scheduled idle damping, yielding |10>.
    initial_state = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
    result = backend.run(
        program,
        shots=0,
        initial_state=initial_state,
        result_config={"counts": False, "final_state": True},
    ).result()

    np.testing.assert_allclose(
        result.get_density_matrix(),
        np.diag([0.0, 0.0, 1.0, 0.0]),
    )
    assert program._instructions is original
    assert len(program._instructions) == 1


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
def test_implicit_idle_uses_logical_ref_for_physical_noise_selector(runtime):

    noise = NoiseModel()
    noise.add(AmplitudeDamping(p=1.0), operation=ops.I, targets=2)
    backend = sc.SCQubitQEC17Simulator(method="DM", runtime=runtime, noise=noise)
    program = Program(2)
    qubits = program.quantum_registers[0]
    program.add(ops.X, qubits[0])
    layout = ResourceLayout({qubits[0]: 13, qubits[1]: 2})

    initial_state = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
    result = backend.run(
        program,
        shots=0,
        initial_state=initial_state,
        resource_layout=layout,
        result_config={"counts": False, "final_state": True},
    ).result()

    np.testing.assert_allclose(
        result.get_density_matrix(),
        np.diag([0.0, 0.0, 1.0, 0.0]),
    )


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
def test_barrier_changes_only_qec17_idle_schedule(runtime):

    noise = NoiseModel()
    noise.add(AmplitudeDamping(p=1.0), operation=ops.I)
    backend = sc.SCQubitQEC17Simulator(method="DM", runtime=runtime, noise=noise)

    without_barrier = Program(2)
    without_barrier.add(ops.X, 0)
    without_barrier.add(ops.X, 1)

    with_barrier = Program(2)
    with_barrier.add(ops.X, 0)
    with_barrier.add(ops.Barrier, (0, 1))
    with_barrier.add(ops.X, 1)

    options = {
        "shots": 0,
        "result_config": {"counts": False, "final_state": True},
    }
    parallel = backend.run(without_barrier, **options).result()
    fenced = backend.run(with_barrier, **options).result()

    np.testing.assert_allclose(
        parallel.get_density_matrix(),
        np.diag([0.0, 0.0, 0.0, 1.0]),
    )
    np.testing.assert_allclose(
        fenced.get_density_matrix(),
        np.diag([0.0, 1.0, 0.0, 0.0]),
    )


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
def test_qec17_parameter_sweep_reuses_scheduled_idle_structure(runtime):

    theta = fq.Parameter("theta")
    program = Program(2)
    program.add(ops.RZ(theta), 0)
    program.add(ops.X, 1)
    values = np.array([0.2, 0.9])

    noise = NoiseModel()
    noise.add(AmplitudeDamping(p=0.25), operation=ops.I, targets=0)
    backend = sc.SCQubitQEC17Simulator(method="DM", runtime=runtime, noise=noise)
    initial_state = np.array([1.0, 0.0, 1.0, 0.0], dtype=complex) / np.sqrt(2)
    options = {
        "shots": 0,
        "initial_state": initial_state,
        "result_config": {"counts": False, "final_state": True},
    }

    swept = backend.run_sweep(program, {theta: values}, **options).result()
    bound = [
        backend.run(
            program.assign_parameters({theta: value}),
            **options,
        ).result()
        for value in values
    ]

    for sweep_result, bound_result in zip(swept, bound, strict=True):
        np.testing.assert_allclose(
            sweep_result.get_density_matrix(),
            bound_result.get_density_matrix(),
        )
