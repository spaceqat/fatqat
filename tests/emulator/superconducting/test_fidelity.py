"""Computational-subspace fidelity checks for superconducting pulse recipes."""

from math import pi

import numpy as np
import pytest

import fatqat as fq
from fatqat.simulator import Simulator
from fatqat.implementation import default_matrix_implementation_map

_PARALLEL_ROTATIONS = (
    (fq.ops.RX(0.4), (0,)),
    (fq.ops.RY(-0.35), (1,)),
)
_H0_CZ = (
    (fq.ops.RZ(pi), (0,)),
    (fq.ops.RY(pi / 2), (0,)),
    (fq.ops.CZ, (0, 1)),
)
_HH_CZ = (
    (fq.ops.RZ(pi), (0,)),
    (fq.ops.RY(pi / 2), (0,)),
    (fq.ops.RZ(pi), (1,)),
    (fq.ops.RY(pi / 2), (1,)),
    (fq.ops.CZ, (0, 1)),
)
_MIXED_CZ_ISWAP = (
    (fq.ops.RZ(pi), (0,)),
    (fq.ops.RY(pi / 2), (0,)),
    (fq.ops.RY(0.7), (1,)),
    (fq.ops.CZ, (0, 1)),
    (fq.ops.iSwap, (0, 1)),
    (fq.ops.RY(0.4), (0,)),
    (fq.ops.RZ(-0.3), (1,)),
    (fq.ops.RX(0.2), (1,)),
)
# This is the composed-sequence guard for the virtual-frame sign convention:
# its non-pi RZ updates precede phase-sensitive drives on the same subsystem.
# The H-based sequences above use RZ(pi), where exp(+i*pi) == exp(-i*pi), and
# therefore cannot distinguish the two frame-binding signs.
_MULTIPLE_FRAMES = (
    (fq.ops.RZ(0.2), (0,)),
    (fq.ops.RX(0.3), (0,)),
    (fq.ops.RZ(-0.4), (0,)),
    (fq.ops.RY(0.5), (0,)),
    (fq.ops.RX(-0.25), (1,)),
)


def program_from_operations(operations):
    program = fq.Program(2)
    for operation, targets in operations:
        program.add(operation, targets)
    return program


def embed_local_unitary(local, targets, n_qubits):
    """Embed a target-ordered local matrix in first-subsystem-major order."""
    target_count = len(targets)
    result = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for input_index in range(2**n_qubits):
        input_bits = [
            (input_index >> (n_qubits - 1 - qubit)) & 1 for qubit in range(n_qubits)
        ]
        local_input = sum(
            input_bits[target] << (target_count - 1 - ordinal)
            for ordinal, target in enumerate(targets)
        )
        for local_output in range(2**target_count):
            output_bits = input_bits.copy()
            for ordinal, target in enumerate(targets):
                output_bits[target] = (local_output >> (target_count - 1 - ordinal)) & 1
            output_index = sum(
                bit << (n_qubits - 1 - qubit) for qubit, bit in enumerate(output_bits)
            )
            result[output_index, input_index] += local[local_output, local_input]
    return result


def matrix_program_unitary(program):
    """Compose the public matrix rules in the pulse model's tensor order."""
    implementation_map = default_matrix_implementation_map()
    references = tuple(
        register[index]
        for register in program.quantum_registers
        for index in range(register.size)
    )
    result = np.eye(2 ** len(references), dtype=complex)
    for applied in program.operations:
        rule = implementation_map.implementation_for(applied.operation)
        assert rule is not None
        local = rule(applied.operation, targets=applied.targets)
        targets = tuple(references.index(target) for target in applied.targets)
        result = embed_local_unitary(local, targets, len(references)) @ result
    return result


def computational_subspace_unitary(backend, program, schedule_mode="ASAP"):
    """Project the full qutrit propagator into the ordered qubit subspace."""
    n_qubits = sum(register.size for register in program.quantum_registers)
    physical_dimension = backend.model.physical_dimension
    indices = tuple(
        sum(
            ((basis_index >> (n_qubits - 1 - qubit)) & 1)
            * physical_dimension ** (n_qubits - 1 - qubit)
            for qubit in range(n_qubits)
        )
        for basis_index in range(2**n_qubits)
    )
    propagator = backend.propagator(program, schedule_mode=schedule_mode)
    return propagator[np.ix_(indices, indices)]


def process_fidelity_and_leakage(actual, ideal):
    """Return phase-invariant process fidelity and mean input leakage."""
    dimension = ideal.shape[0]
    overlap = np.trace(ideal.conj().T @ actual)
    process_fidelity = abs(overlap) ** 2 / dimension**2
    survival = np.trace(actual.conj().T @ actual).real / dimension
    return float(process_fidelity), float(1.0 - survival)


def pulse_ground_state_in_simulator_order(backend, program):
    """Return propagated |0...0> in the matrix simulator's little-endian order."""
    tensor_order_state = computational_subspace_unitary(backend, program)[:, 0]
    n_qubits = sum(register.size for register in program.quantum_registers)
    tensor_indices = tuple(
        sum(
            ((basis_index >> qubit) & 1) << (n_qubits - 1 - qubit)
            for qubit in range(n_qubits)
        )
        for basis_index in range(2**n_qubits)
    )
    return tensor_order_state[list(tensor_indices)]


def test_virtual_frame_sequence_matches_analytic_unitary(backend):
    z_angle = pi / 2
    x_angle = 0.3
    program = program_from_operations(
        ((fq.ops.RZ(z_angle), (0,)), (fq.ops.RX(x_angle), (0,)))
    )
    actual = computational_subspace_unitary(backend, program)

    ideal_rz = np.diag((np.exp(-0.5j * z_angle), np.exp(0.5j * z_angle)))
    cosine = np.cos(x_angle / 2)
    sine = np.sin(x_angle / 2)
    ideal_rx = np.array(((cosine, -1j * sine), (-1j * sine, cosine)), dtype=complex)
    ideal = np.kron(ideal_rx @ ideal_rz, np.eye(2))

    phase_invariant_error = 1 - abs(np.trace(ideal.conj().T @ actual)) / 4
    assert phase_invariant_error < 1e-5


def test_intermediate_frame_rotates_later_drive_with_virtual_z_sign(backend):
    framed_rx = fq.Program(1)
    framed_rx.add(fq.ops.RZ(pi / 2), 0)
    framed_rx.add(fq.ops.RX(0.3), 0)
    ry = fq.Program(1)
    ry.add(fq.ops.RY(-0.3), 0)

    assert np.allclose(
        backend.propagator(framed_rx, apply_final_frame=False),
        backend.propagator(ry, apply_final_frame=False),
        atol=2e-7,
    )


@pytest.mark.parametrize(
    ("operation", "targets", "minimum_fidelity"),
    (
        (fq.ops.RX(0.37), (0,), 0.9999),
        (fq.ops.RX(0.37), (1,), 0.9999),
        (fq.ops.RY(-0.41), (0,), 0.9999),
        (fq.ops.RY(-0.41), (1,), 0.9999),
        (fq.ops.RZ(0.53), (0,), 0.999999),
        (fq.ops.RZ(0.53), (1,), 0.999999),
        (fq.ops.iSwap, (0, 1), 0.985),
        (fq.ops.iSwap, (1, 0), 0.985),
        (fq.ops.CZ, (0, 1), 0.99),
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
    every composed program's full propagator, which mathematically implies
    this state fidelity. The value kept here is the *other execution path*:
    that comparison goes through `propagator()`, while this one goes through
    the matrix simulator and the ordering conversion between the two backends.
    Two representative programs cover that path; repeating all five only
    re-ran the solver.
    """
    program = program_from_operations(operations)
    pulse_state = pulse_ground_state_in_simulator_order(backend, program)
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
    program.add(fq.ops.RZ(pi / 2), 0)
    program.add(fq.ops.RX(0.3), 0)
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
        pulse_density_matrix[np.ix_((0, 3), (0, 3))],
        expected_density_matrix,
        atol=2e-3,
    )
