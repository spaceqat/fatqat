import math

import pytest

import fatqat as fq
from fatqat.compiler import CompileContext, UnsupportedFeatureError
from fatqat.compiler.dialects import MEASURE, SC_INSTRUCTION_RULES, SCProgram
from fatqat.compiler.passes import normalize_sc, normalize_sc_program, snapshot_program


def _normalize(program):
    return normalize_sc.run(snapshot_program(program), CompileContext())


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
    program.add(fq.operations.iSwap, (0, 1))

    with pytest.raises(UnsupportedFeatureError, match="iSwap"):
        _normalize(program)
