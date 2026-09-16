"""LQCloud gate semantics through the general simulator's public results."""

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops


def _half_turn_reference(azimuth):
    return (ops.RZ(-azimuth), ops.RX(np.pi / 2), ops.RZ(azimuth))


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
@pytest.mark.parametrize(
    "gate,reference,phase",
    [
        (ops.HY, (ops.H, ops.Y), 1),
        (ops.MX, (ops.X,), -1),
        (ops.MY, (ops.Y,), -1),
        (ops.MZ, (ops.Z,), -1),
        (ops.XHalf, (ops.RX(np.pi / 2),), 1),
        (ops.MXHalf, (ops.RX(-np.pi / 2),), 1),
        (ops.YHalf, (ops.RY(np.pi / 2),), 1),
        (ops.MYHalf, (ops.RY(-np.pi / 2),), 1),
        (ops.XYHalf, _half_turn_reference(np.pi / 4), 1),
        (ops.MXYHalf, _half_turn_reference(3 * np.pi / 4), 1),
        (ops.MXMYHalf, _half_turn_reference(-3 * np.pi / 4), 1),
        (ops.XMYHalf, _half_turn_reference(-np.pi / 4), 1),
    ],
    ids=lambda value: value.name if isinstance(value, ops.Operation) else None,
)
def test_native_gate_matches_existing_rotations_with_exact_phase(
    runtime, gate, reference, phase
):
    if runtime == "numba":
        pytest.importorskip("numba")
    backend = fq.simulator.Simulator(method="unitary", runtime=runtime)
    program = fq.Program(1)
    program.add(gate, 0)
    expected_program = fq.Program(1)
    for operation in reference:
        expected_program.add(operation, 0)

    actual = backend.run(program, shots=0).result().get_unitary()
    expected = backend.run(expected_program, shots=0).result().get_unitary()

    np.testing.assert_allclose(actual, phase * expected, atol=1e-14)


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
def test_su2_preserves_the_supplied_matrix_on_a_register_view(runtime):
    if runtime == "numba":
        pytest.importorskip("numba")
    matrix = np.exp(0.37j) * np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
    qubits = fq.QuantumRegister(2)
    program = fq.Program([qubits])
    program.add(ops.SU2(matrix), qubits.all())

    actual = (
        fq.simulator.Simulator(method="unitary", runtime=runtime)
        .run(program, shots=0)
        .result()
        .get_unitary()
    )

    np.testing.assert_allclose(actual, np.kron(matrix, matrix), atol=1e-14)
