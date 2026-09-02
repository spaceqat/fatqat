import math

import numpy as np
import pytest

import fatqat as fq
from fatqat.compiler import CompileContext, UnsupportedFeatureError
from fatqat.compiler.dialects import (
    LogicalGate,
    LogicalMeasure,
    LogicalProgram,
    NAGate,
    NAMeasure,
)
from fatqat.compiler.passes import normalize_na, normalize_na_program, snapshot_program


def _normalize(program: fq.Program):
    return normalize_na.run(snapshot_program(program), CompileContext())


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


def test_normalize_na_uses_native_ry_and_decomposes_cx():
    atoms = fq.QuantumRegister(2, name="atoms")
    program = fq.Program([atoms])
    ry = fq.operations.RY(0.25)
    program.add(ry, atoms[0])
    program.add(fq.operations.CX, (atoms[0], atoms[1]))

    logical = snapshot_program(program)
    result = normalize_na.run(logical, CompileContext())

    gates = [item for item in result.instructions if type(item) is NAGate]
    assert [type(item.operation) for item in gates] == [
        fq.operations.RY,
        fq.operations.RZ,
        fq.operations.RY,
        type(fq.operations.CZ),
        fq.operations.RZ,
        fq.operations.RY,
    ]
    assert gates[0].operation is ry
    assert gates[1].operation == fq.operations.RZ(math.pi)
    assert gates[2].operation == fq.operations.RY(math.pi / 2)
    assert gates[3].atoms == (atoms[0], atoms[1])
    assert result.atoms == logical.qubits
    assert result.atoms[0] is logical.qubits[0]


def test_normalize_na_lowers_the_exact_closed_gate_set_without_merging():
    atoms = fq.QuantumRegister(2, name="atoms")
    program = fq.Program([atoms])
    program.add(fq.operations.I, atoms[0])
    program.add(fq.operations.RX(0.1), atoms[0])
    program.add(fq.operations.RY(0.2), atoms[0])
    program.add(fq.operations.RZ(0.3), atoms[0])
    program.add(fq.operations.CZ, (atoms[0], atoms[1]))
    program.add(fq.operations.H, atoms[0])
    program.add(fq.operations.X, atoms[0])
    program.add(fq.operations.Y, atoms[0])
    program.add(fq.operations.Z, atoms[0])
    program.add(fq.operations.S, atoms[0])
    program.add(fq.operations.Sdg, atoms[0])
    program.add(fq.operations.T, atoms[0])
    program.add(fq.operations.Tdg, atoms[0])
    program.add(fq.operations.Phase(0.4), atoms[0])
    program.add(fq.operations.RZ(0.5), atoms[0])
    program.add(fq.operations.RZ(-0.5), atoms[0])
    program.add(fq.operations.CZ, (atoms[0], atoms[1]))
    program.add(fq.operations.CZ, (atoms[0], atoms[1]))

    logical = snapshot_program(program)
    result = normalize_na.run(logical, CompileContext())

    gates = [item for item in result.instructions if type(item) is NAGate]
    assert [item.operation for item in gates] == [
        fq.operations.RX(0.1),
        fq.operations.RY(0.2),
        fq.operations.RZ(0.3),
        fq.operations.CZ,
        fq.operations.RZ(math.pi),
        fq.operations.RY(math.pi / 2),
        fq.operations.RX(math.pi),
        fq.operations.RY(math.pi),
        fq.operations.RZ(math.pi),
        fq.operations.RZ(math.pi / 2),
        fq.operations.RZ(-math.pi / 2),
        fq.operations.RZ(math.pi / 4),
        fq.operations.RZ(-math.pi / 4),
        fq.operations.RZ(0.4),
        fq.operations.RZ(0.5),
        fq.operations.RZ(-0.5),
        fq.operations.CZ,
        fq.operations.CZ,
    ]
    assert all(
        type(item.operation)
        in (
            fq.operations.RX,
            fq.operations.RY,
            fq.operations.RZ,
            type(fq.operations.CZ),
        )
        for item in gates
    )
    assert [item.operation_id for item in gates] == [
        f"na.{index}" for index in range(18)
    ]
    assert [item.origin_ids for item in gates] == [
        ("logical.1",),
        ("logical.2",),
        ("logical.3",),
        ("logical.4",),
        ("logical.5",),
        ("logical.5",),
        ("logical.6",),
        ("logical.7",),
        ("logical.8",),
        ("logical.9",),
        ("logical.10",),
        ("logical.11",),
        ("logical.12",),
        ("logical.13",),
        ("logical.14",),
        ("logical.15",),
        ("logical.16",),
        ("logical.17",),
    ]


def test_normalize_na_expands_swap_through_three_cx_with_source_provenance():
    atoms = fq.QuantumRegister(2, name="atoms")
    program = fq.Program([atoms])
    program.add(fq.operations.Swap, (atoms[0], atoms[1]))

    first = _normalize(program)
    second = _normalize(program)

    assert first == second
    assert [type(item.operation) for item in first.instructions] == [
        fq.operations.RZ,
        fq.operations.RY,
        type(fq.operations.CZ),
        fq.operations.RZ,
        fq.operations.RY,
    ] * 3
    first_atom, second_atom = first.atoms
    assert [item.atoms for item in first.instructions] == [
        (second_atom,),
        (second_atom,),
        (first_atom, second_atom),
        (second_atom,),
        (second_atom,),
        (first_atom,),
        (first_atom,),
        (second_atom, first_atom),
        (first_atom,),
        (first_atom,),
        (second_atom,),
        (second_atom,),
        (first_atom, second_atom),
        (second_atom,),
        (second_atom,),
    ]
    assert all(item.origin_ids == ("logical.0",) for item in first.instructions)
    assert [item.operation_id for item in first.instructions] == [
        f"na.{index}" for index in range(15)
    ]


def test_normalize_na_preserves_terminal_measurements_and_their_refs():
    atoms = fq.QuantumRegister(2, name="atoms")
    bits = fq.ClassicalRegister(2, name="bits")
    program = fq.Program([atoms], [bits])
    program.add(fq.operations.X, atoms[0])
    program.measure((atoms[0], atoms[1]), (bits[0], bits[1]))

    logical = snapshot_program(program)
    result = normalize_na.run(logical, CompileContext())

    assert [type(item) for item in result.instructions] == [
        NAGate,
        NAMeasure,
        NAMeasure,
    ]
    source_first_measure, source_second_measure = logical.instructions[1:]
    first_measure, second_measure = result.instructions[1:]
    assert first_measure == NAMeasure(
        "na.1",
        ("logical.1",),
        source_first_measure.qubit,
        source_first_measure.clbit,
    )
    assert second_measure == NAMeasure(
        "na.2",
        ("logical.2",),
        source_second_measure.qubit,
        source_second_measure.clbit,
    )
    assert first_measure.atom is source_first_measure.qubit
    assert first_measure.clbit is source_first_measure.clbit


def test_normalize_na_rejects_reset_and_nonterminal_measurement():
    atom = fq.QuantumRegister(1, name="atom")[0]
    bit = fq.ClassicalRegister(1, name="bit")[0]
    reset = LogicalProgram(
        (atom,), (), (LogicalGate("logical.0", fq.operations.Reset, (atom,)),)
    )
    dynamic = LogicalProgram(
        (atom,),
        (bit,),
        (
            LogicalMeasure("logical.0", atom, bit),
            LogicalGate("logical.1", fq.operations.X, (atom,)),
        ),
    )

    with pytest.raises(UnsupportedFeatureError, match="Reset"):
        normalize_na_program(reset)
    with pytest.raises(UnsupportedFeatureError, match="terminal"):
        normalize_na_program(dynamic)


def test_normalized_bell_operations_match_logical_ideal_semantics():
    atoms = fq.QuantumRegister(2, name="atoms")
    logical_program = fq.Program([atoms])
    logical_program.add(fq.operations.H, atoms[0])
    logical_program.add(fq.operations.CX, (atoms[0], atoms[1]))

    normalized = _normalize(logical_program)
    native_program = fq.Program([atoms])
    for instruction in normalized.instructions:
        if type(instruction) is NAGate:
            native_program.add(instruction.operation, instruction.atoms)

    _assert_same_up_to_global_phase(
        _statevector(native_program), _statevector(logical_program)
    )


def test_normalized_swap_matches_logical_semantics_for_asymmetric_input():
    atoms = fq.QuantumRegister(2, name="atoms")
    logical_program = fq.Program([atoms])
    logical_program.add(fq.operations.X, atoms[0])
    logical_program.add(fq.operations.Swap, (atoms[0], atoms[1]))

    normalized = _normalize(logical_program)
    native_program = fq.Program([atoms])
    for instruction in normalized.instructions:
        if type(instruction) is NAGate:
            native_program.add(instruction.operation, instruction.atoms)

    _assert_same_up_to_global_phase(
        _statevector(native_program), _statevector(logical_program)
    )
