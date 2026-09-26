import math

import numpy as np
import pytest

import fatqat as fq
from fatqat.compiler import CompileContext, UnsupportedFeatureError
from fatqat.compiler.algorithms import topological_order
from fatqat.compiler.dialects import MEASURE, SC_INSTRUCTION_RULES, SCProgram
from fatqat.compiler.passes import normalize_sc, normalize_sc_program, snapshot_program


def _normalize(program):
    return normalize_sc.run(snapshot_program(program), CompileContext())


def _statevector(program: fq.Program) -> np.ndarray:
    return (
        fq.simulator.Simulator("SV")
        .run(program, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )


def _assert_same_up_to_global_phase(actual: np.ndarray, expected: np.ndarray) -> None:
    pivot = int(np.argmax(np.abs(expected)))
    phase = actual[pivot] / expected[pivot]
    assert np.allclose(actual, phase / abs(phase) * expected)


@pytest.mark.parametrize(
    "operation",
    (
        fq.operations.U(0.37, -0.41, 0.73),
        fq.operations.U1(0.73),
        fq.operations.U2(-0.41, 0.73),
        fq.operations.U3(0.37, -0.41, 0.73),
    ),
)
def test_sc_normalization_lowers_qasm_u_family_without_changing_semantics(operation):
    qubits = fq.QuantumRegister(1)
    source = fq.Program([qubits])
    source.add(fq.operations.H, qubits[0])
    source.add(operation, qubits[0])

    normalized = _normalize(source)
    lowered = fq.Program([qubits])
    for node_id in topological_order(normalized):
        node = normalized.nodes[node_id]
        lowered.add(node.instruction, node.qubits)

    assert all(
        type(node.instruction) in SC_INSTRUCTION_RULES for node in normalized.nodes
    )
    assert any("logical.1" in node.origin_ids for node in normalized.nodes)
    _assert_same_up_to_global_phase(_statevector(lowered), _statevector(source))


def test_direct_sc_operations_and_semantic_swap_are_preserved():
    program = fq.Program(2, 1)
    program.add(fq.operations.RX(0.2), 0)
    program.add(fq.operations.RZ(0.3), 1)
    program.add(fq.operations.CZ, (0, 1))
    program.add(fq.operations.Swap, (0, 1))
    program.measure(0, 0)

    sc = _normalize(program)

    assert isinstance(sc, SCProgram)
    assert [node.instruction for node in sc.nodes] == [
        fq.operations.RX(0.2),
        fq.operations.RZ(0.3),
        fq.operations.CZ,
        fq.operations.Swap,
        MEASURE,
    ]
    assert sc.nodes[3].origin_ids == ("logical.3",)


def test_sc_normalization_preserves_frontend_register_refs():
    register = fq.QuantumRegister(2, name="data")
    program = fq.Program([register])
    program.add(fq.operations.CX, (register[0], register[1]))
    logical = snapshot_program(program)
    sc = normalize_sc_program(logical)

    assert sc.qubits == logical.qubits == (register[0], register[1])
    assert tuple(wire.qubit for wire in sc.wires) == logical.qubits


def test_reset_is_scalarized_to_one_sc_node_per_qubit():
    program = fq.Program(2)
    program.add(fq.operations.Reset, (0, 1))

    sc = _normalize(program)

    assert [node.instruction for node in sc.nodes] == [
        fq.operations.Reset,
        fq.operations.Reset,
    ]
    assert [node.qubits for node in sc.nodes] == [
        (program.quantum_registers[0][0],),
        (program.quantum_registers[0][1],),
    ]
    assert all(node.origin_ids == ("logical.0",) for node in sc.nodes)


def test_bell_circuit_decomposes_deterministically_to_closed_sc_set():
    program = fq.Program(2)
    program.add(fq.operations.H, 0)
    program.add(fq.operations.CX, (0, 1))

    first = _normalize(program)
    second = _normalize(program)

    assert first == second
    assert all(type(node.instruction) in SC_INSTRUCTION_RULES for node in first.nodes)
    assert {origin for node in first.nodes for origin in node.origin_ids} == {
        "logical.0",
        "logical.1",
    }


def test_zero_rotations_are_removed_in_the_builder():
    program = fq.Program(1)
    program.add(fq.operations.RX(0.0), 0)
    program.add(fq.operations.RZ(2 * math.pi), 0)

    sc = _normalize(program)

    assert sc.nodes == ()
    assert sc.wires[0].nodes == ()


def test_adjacent_same_axis_rotations_merge_origins_once():
    program = fq.Program(1)
    program.add(fq.operations.RX(0.25), 0)
    program.add(fq.operations.RX(0.75), 0)

    sc = _normalize(program)

    assert len(sc.nodes) == 1
    assert sc.nodes[0].instruction == fq.operations.RX(1.0)
    assert sc.nodes[0].origin_ids == ("logical.0", "logical.1")
    assert sc.nodes[0].operation_id.startswith("sc.merge.")


def test_adjacent_cz_gates_cancel_only_when_adjacent_on_both_wires():
    program = fq.Program(2)
    program.add(fq.operations.CZ, (0, 1))
    program.add(fq.operations.CZ, (0, 1))

    sc = _normalize(program)

    assert sc.nodes == ()
    assert all(wire.nodes == () for wire in sc.wires)


def test_unsupported_logical_operation_fails_loudly():
    program = fq.Program(2)
    program.add(fq.operations.Fourier, 0)

    with pytest.raises(UnsupportedFeatureError, match="Fourier"):
        _normalize(program)


@pytest.mark.parametrize(
    "gate",
    [
        fq.operations.CX,
        fq.operations.Swap,
        fq.operations.CCX,
        fq.operations.iSwap,
        fq.operations.CY,
        fq.operations.CS,
        fq.operations.CPhase(0.37),
        fq.operations.CPhase(-1.2),
        fq.operations.CSwap,
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("qasm", [False, True])
def test_composite_gates_preserve_entire_unitary(gate, reverse, qasm):
    source = fq.Program(gate.num_subsystems)
    operands = tuple(range(gate.num_subsystems))
    source.add(gate, operands[::-1] if reverse else operands)
    normalized_source = fq.qasm.from_qasm(fq.qasm.to_qasm(source)) if qasm else source
    sc = _normalize(normalized_source)
    lowered = fq.Program(normalized_source.quantum_registers)
    for node_id in topological_order(sc):
        node = sc.nodes[node_id]
        lowered.add(node.instruction, node.qubits)
    simulator = fq.simulator.Simulator("unitary", runtime="numpy")
    expected = simulator.run(source).result().get_unitary()
    actual = simulator.run(lowered).result().get_unitary()
    phase = np.vdot(expected, actual) / expected.shape[0]
    assert np.isclose(abs(phase), 1)
    assert np.allclose(actual, phase * expected, atol=1e-10)
    if not qasm:
        assert all(node.origin_ids == ("logical.0",) for node in sc.nodes)
