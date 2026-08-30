"""Computational-subspace fidelity checks for superconducting pulse recipes."""

from math import pi

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.simulator import Simulator
from fatqat.implementation import default_matrix_implementation_map

_PARALLEL_ROTATIONS = (
    (ops.RX(0.4), (0,)),
    (ops.RY(-0.35), (1,)),
)
_H0_CZ = (
    (ops.RZ(pi), (0,)),
    (ops.RY(pi / 2), (0,)),
    (ops.CZ, (0, 1)),
)
_HH_CZ = (
    (ops.RZ(pi), (0,)),
    (ops.RY(pi / 2), (0,)),
    (ops.RZ(pi), (1,)),
    (ops.RY(pi / 2), (1,)),
    (ops.CZ, (0, 1)),
)
_MIXED_CZ_ISWAP = (
    (ops.RZ(pi), (0,)),
    (ops.RY(pi / 2), (0,)),
    (ops.RY(0.7), (1,)),
    (ops.CZ, (0, 1)),
    (ops.iSwap, (0, 1)),
    (ops.RY(0.4), (0,)),
    (ops.RZ(-0.3), (1,)),
    (ops.RX(0.2), (1,)),
)
# This is the composed-sequence guard for the virtual-frame sign convention:
# its non-pi RZ updates precede phase-sensitive drives on the same subsystem.
# The H-based sequences above use RZ(pi), where exp(+i*pi) == exp(-i*pi), and
# therefore cannot distinguish the two frame-binding signs.
_MULTIPLE_FRAMES = (
    (ops.RZ(0.2), (0,)),
    (ops.RX(0.3), (0,)),
    (ops.RZ(-0.4), (0,)),
    (ops.RY(0.5), (0,)),
    (ops.RX(-0.25), (1,)),
)


def program_from_operations(operations):
    program = fq.Program(2)
    for operation, targets in operations:
        program.add(operation, targets)
    return program


def embed_local_unitary(local, targets, n_qubits):
    """Embed a target-ordered local matrix in canonical little-endian order."""
    target_count = len(targets)
    result = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for input_index in range(2**n_qubits):
        input_bits = [(input_index >> qubit) & 1 for qubit in range(n_qubits)]
        local_input = sum(
            input_bits[target] << (target_count - 1 - ordinal)
            for ordinal, target in enumerate(targets)
        )
        for local_output in range(2**target_count):
            output_bits = input_bits.copy()
            for ordinal, target in enumerate(targets):
                output_bits[target] = (local_output >> (target_count - 1 - ordinal)) & 1
            output_index = sum(bit << qubit for qubit, bit in enumerate(output_bits))
            result[output_index, input_index] += local[local_output, local_input]
    return result


def matrix_program_unitary(program):
    """Compose public matrix rules in FATQAT's canonical state order."""
    implementation_map = default_matrix_implementation_map()
    references = tuple(
        register[index]
        for register in program.quantum_registers
        for index in range(register.size)
    )
    result = np.eye(2 ** len(references), dtype=complex)
    for applied in program._instructions:
        rule = implementation_map.implementation_for(applied.operation)
        assert rule is not None
        local = rule(applied.operation, targets=applied.targets)
        targets = tuple(references.index(target) for target in applied.targets)
        result = embed_local_unitary(local, targets, len(references)) @ result
    return result


def computational_subspace_unitary(backend, program, schedule_mode="ASAP"):
    """Project the full qutrit propagator into canonical qubit order."""
    n_qubits = sum(register.size for register in program.quantum_registers)
    physical_dimension = len(backend.model.basis_order)
    indices = tuple(
        sum(
            ((basis_index >> qubit) & 1) * physical_dimension**qubit
            for qubit in range(n_qubits)
        )
        for basis_index in range(2**n_qubits)
    )
    unitary_backend = fq.emulator.TransmonEmulator(backend.model, method="unitary")
    propagator = (
        unitary_backend.run(program, simulation_config={"schedule_mode": schedule_mode})
        .result()
        .get_unitary()
    )
    return propagator[np.ix_(indices, indices)]


def process_fidelity_and_leakage(actual, ideal):
    """Return phase-invariant process fidelity and mean input leakage."""
    dimension = ideal.shape[0]
    overlap = np.trace(ideal.conj().T @ actual)
    process_fidelity = abs(overlap) ** 2 / dimension**2
    survival = np.trace(actual.conj().T @ actual).real / dimension
    return float(process_fidelity), float(1.0 - survival)


def pulse_ground_state(backend, program):
    """Return propagated |0...0> in FATQAT's canonical state order."""
    return computational_subspace_unitary(backend, program)[:, 0]


def test_virtual_frame_sequence_matches_analytic_unitary(backend):
    z_angle = pi / 2
    x_angle = 0.3
    program = program_from_operations(
        ((ops.RZ(z_angle), (0,)), (ops.RX(x_angle), (0,)))
    )
    actual = computational_subspace_unitary(backend, program)

    ideal_rz = np.diag((np.exp(-0.5j * z_angle), np.exp(0.5j * z_angle)))
    cosine = np.cos(x_angle / 2)
    sine = np.sin(x_angle / 2)
    ideal_rx = np.array(((cosine, -1j * sine), (-1j * sine, cosine)), dtype=complex)
    ideal = embed_local_unitary(ideal_rx @ ideal_rz, (0,), 2)

    phase_invariant_error = 1 - abs(np.trace(ideal.conj().T @ actual)) / 4
    assert phase_invariant_error < 1e-5


@pytest.mark.parametrize(
    ("operation", "targets", "minimum_fidelity"),
    (
        (ops.RX(0.37), (0,), 0.9999),
        (ops.RX(0.37), (1,), 0.9999),
        (ops.RY(-0.41), (0,), 0.9999),
        (ops.RY(-0.41), (1,), 0.9999),
        (ops.RZ(0.53), (0,), 0.999999),
        (ops.RZ(0.53), (1,), 0.999999),
        (ops.iSwap, (0, 1), 0.985),
        (ops.iSwap, (1, 0), 0.985),
        (ops.CZ, (0, 1), 0.99),
        (ops.CZ, (1, 0), 0.99),
    ),
    ids=(
        "rx-q0",
        "rx-q1",
        "ry-q0",
        "ry-q1",
        "rz-q0",
        "rz-q1",
        "iswap-01",
        "iswap-10",
        "cz-01",
        "cz-10",
    ),
)
def test_native_gate_process_fidelity_and_leakage(
    backend, operation, targets, minimum_fidelity
):
    program = program_from_operations(((operation, targets),))
    actual = computational_subspace_unitary(backend, program)
    ideal = matrix_program_unitary(program)

    process_fidelity, leakage = process_fidelity_and_leakage(actual, ideal)

    assert process_fidelity > minimum_fidelity
    assert -1e-8 < leakage < 1e-5


@pytest.mark.parametrize("schedule_mode", ("ASAP", "ALAP"))
@pytest.mark.parametrize(
    ("operations", "minimum_fidelity"),
    (
        (_PARALLEL_ROTATIONS, 0.9999),
        (_H0_CZ, 0.998),
        (_HH_CZ, 0.989),
        (_MIXED_CZ_ISWAP, 0.991),
        (_MULTIPLE_FRAMES, 0.9998),
    ),
    ids=("parallel-rotations", "h-cz", "hh-cz", "mixed-cz-iswap", "frames"),
)
def test_composed_process_fidelity_in_both_schedule_modes(
    backend, operations, minimum_fidelity, schedule_mode
):
    program = program_from_operations(operations)
    actual = computational_subspace_unitary(backend, program, schedule_mode)
    ideal = matrix_program_unitary(program)

    process_fidelity, leakage = process_fidelity_and_leakage(actual, ideal)

    assert process_fidelity > minimum_fidelity
    assert -1e-8 < leakage < 1e-5


@pytest.mark.parametrize(
    ("operations", "minimum_fidelity"),
    (
        (_HH_CZ, 0.985),
        (_MIXED_CZ_ISWAP, 0.995),
    ),
    ids=("hh-cz", "mixed-cz-iswap"),
)
def test_composed_ground_state_matches_matrix_simulator(
    backend, operations, minimum_fidelity
):
    """Cross-check the pulse stack against the matrix simulator end to end.

    `test_composed_process_fidelity_in_both_schedule_modes` already bounds
    every composed program's full unitary, which mathematically implies this
    state fidelity. Two representative programs cover the matrix-simulator
    ordering contract without repeating every expensive pulse solve.
    """
    program = program_from_operations(operations)
    pulse_state = pulse_ground_state(backend, program)
    simulator_state = np.asarray(
        Simulator("SV")
        .run(program, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )

    state_fidelity = abs(np.vdot(simulator_state, pulse_state)) ** 2

    assert state_fidelity > minimum_fidelity


def test_run_uses_the_correct_virtual_frame_binding(backend):
    program = fq.Program(1)
    program.add(ops.RZ(pi / 2), 0)
    program.add(ops.RX(0.3), 0)
    pulse_density_matrix = (
        backend.run(program, result_config={"counts": False, "final_state": True})
        .result()
        .get_density_matrix()
    )
    # run() reports the rotating-frame density matrix without composing its
    # terminal frame. RZ(pi/2) therefore binds the later RX as RY(-0.3).
    rotating_frame_state = np.array((np.cos(0.15), -np.sin(0.15)))
    expected_density_matrix = np.outer(
        rotating_frame_state, rotating_frame_state.conj()
    )

    assert np.allclose(
        pulse_density_matrix[np.ix_((0, 1), (0, 1))],
        expected_density_matrix,
        atol=2e-3,
    )
