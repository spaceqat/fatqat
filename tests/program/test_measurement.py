"""Tests Program.measure insertion, ordering, and validation."""

import pytest

from fatqat.program import Program
from fatqat.operations import Measurement
import fatqat.operations as ops
from fatqat.registers import GridRegister, QuantumRegister, ClassicalRegister


def test_measure_appends_measurement():
    p = Program(2, 2)
    result = p.measure(0, 0)
    assert result is None
    assert len(p._instructions) == 1
    m = p._instructions[0]
    assert isinstance(m, Measurement)
    assert m.targets == (p.quantum_registers[0][0],)
    assert m.outputs == (p.classical_registers[0][0],)


def test_instructions_preserve_order_and_type_mix():
    p = Program(2, 2)
    p.add(ops.H, 0)
    p.add(ops.CZ, (0, 1))
    p.measure(0, 0)
    p.measure(1, 1)
    assert len(p._instructions) == 4
    assert p._instructions[0].operation.name == "H"
    assert isinstance(p._instructions[2], Measurement)


def test_measure_rejects_quantum_ref_as_output():
    p = Program(2, 2)
    with pytest.raises(TypeError):
        p.measure(0, p.quantum_registers[0][1])  # quantum ref as classical slot


def test_measure_int_output_rejects_multiple_classical_registers():
    p = Program(
        [QuantumRegister(2, name="q")],
        [ClassicalRegister(1, name="a"), ClassicalRegister(1, name="b")],
    )

    with pytest.raises(TypeError, match="explicit RegisterRef"):
        p.measure(0, 0)


def test_measure_explicit_output_ref_works_with_multiple_classical_registers():
    p = Program(
        [QuantumRegister(2, name="q")],
        [ClassicalRegister(1, name="a"), ClassicalRegister(1, name="b")],
    )

    p.measure(1, p.classical_registers[1][0])

    assert p._instructions[0].targets == (p.quantum_registers[0][1],)
    assert p._instructions[0].outputs == (p.classical_registers[1][0],)


def test_measure_rejects_metadata_argument():
    p = Program(1, 1)

    with pytest.raises(TypeError):
        p.measure(0, 0, metadata={"k": 1})


def test_measure_stores_single_as_one_tuples():
    p = Program(2, 2)
    p.measure(0, 1)

    m = p._instructions[0]
    assert isinstance(m, Measurement)
    assert m.targets == (p.quantum_registers[0][0],)
    assert m.outputs == (p.classical_registers[0][1],)


def test_measure_accepts_grouped_operands():
    p = Program(3, 3)
    p.measure((0, 2), (1, 0))

    m = p._instructions[0]
    assert isinstance(m, Measurement)
    assert m.targets == (p.quantum_registers[0][0], p.quantum_registers[0][2])
    assert m.outputs == (p.classical_registers[0][1], p.classical_registers[0][0])


def test_measure_rejects_mismatched_group_sizes():
    p = Program(3, 2)
    with pytest.raises(ValueError, match="same number"):
        p.measure((0, 1, 2), (0, 1))


def test_measure_rejects_empty_group():
    p = Program(1, 1)
    with pytest.raises(ValueError, match="at least one"):
        p.measure((), ())


def test_measure_all_appends_one_grouped_instruction_in_flat_order():
    p = Program(
        [QuantumRegister(2, name="qa"), QuantumRegister(1, name="qb")],
        [ClassicalRegister(1, name="ca"), ClassicalRegister(2, name="cb")],
    )

    result = p.measure_all()

    assert result is None
    assert len(p._instructions) == 1
    m = p._instructions[0]
    assert isinstance(m, Measurement)
    assert m.targets == (
        p.quantum_registers[0][0],
        p.quantum_registers[0][1],
        p.quantum_registers[1][0],
    )
    assert m.outputs == (
        p.classical_registers[0][0],
        p.classical_registers[1][0],
        p.classical_registers[1][1],
    )


def test_measure_all_uses_first_n_classical_slots_with_extra_classical_storage():
    p = Program(2, 4)

    result = p.measure_all()

    assert result is None

    instructions = p.instructions
    assert len(instructions) == 1

    measurement = instructions[0]
    assert isinstance(measurement, Measurement)

    assert measurement.targets == (
        p.quantum_registers[0][0],
        p.quantum_registers[0][1],
    )

    assert measurement.outputs == (
        p.classical_registers[0][0],
        p.classical_registers[0][1],
    )


def test_measure_all_accepts_explicit_output_destinations():
    p = Program(2, 4)

    result = p.measure_all(outputs=(2, 3))

    assert result is None

    instructions = p.instructions
    assert len(instructions) == 1

    measurement = instructions[0]
    assert isinstance(measurement, Measurement)

    assert measurement.targets == (
        p.quantum_registers[0][0],
        p.quantum_registers[0][1],
    )

    assert measurement.outputs == (
        p.classical_registers[0][2],
        p.classical_registers[0][3],
    )


def test_measure_all_repeated_measurements_use_distinct_outputs():
    p = Program(2, 4)

    p.measure_all(outputs=(0, 1))
    p.measure_all(outputs=(2, 3))

    instructions = p.instructions
    assert len(instructions) == 2

    first_measurement = instructions[0]
    second_measurement = instructions[1]

    assert isinstance(first_measurement, Measurement)
    assert isinstance(second_measurement, Measurement)

    expected_targets = (
        p.quantum_registers[0][0],
        p.quantum_registers[0][1],
    )

    assert first_measurement.targets == expected_targets
    assert second_measurement.targets == expected_targets

    assert first_measurement.outputs == (
        p.classical_registers[0][0],
        p.classical_registers[0][1],
    )

    assert second_measurement.outputs == (
        p.classical_registers[0][2],
        p.classical_registers[0][3],
    )


def test_measure_all_accepts_explicit_outputs_across_registers():
    p = Program(
        [
            QuantumRegister(2, name="qa"),
            QuantumRegister(1, name="qb"),
        ],
        [
            ClassicalRegister(1, name="ca"),
            ClassicalRegister(2, name="cb"),
        ],
    )

    outputs = (
        p.classical_registers[1][0],
        p.classical_registers[0][0],
        p.classical_registers[1][1],
    )

    result = p.measure_all(outputs=outputs)

    assert result is None

    instructions = p.instructions
    assert len(instructions) == 1

    measurement = instructions[0]
    assert isinstance(measurement, Measurement)

    assert measurement.targets == (
        p.quantum_registers[0][0],
        p.quantum_registers[0][1],
        p.quantum_registers[1][0],
    )

    assert measurement.outputs == outputs


def test_measure_all_rejects_mismatched_resource_counts():
    p = Program(2, 1)
    with pytest.raises(ValueError, match="same number"):
        p.measure_all()


def test_measure_all_rejects_empty_program():
    p = Program(0, 0)
    with pytest.raises(ValueError, match="at least one"):
        p.measure_all()


def test_measure_dim_mismatch_raises():
    qt = QuantumRegister(1, dim=3)
    cb = ClassicalRegister(1, dim=2)
    program = Program([qt], [cb])
    with pytest.raises(ValueError):
        program.measure(qt[0], cb[0])


def test_measure_matching_dims_ok():
    qt = QuantumRegister(1, dim=3)
    ct = ClassicalRegister(1, dim=3)
    program = Program([qt], [ct])
    program.measure(qt[0], ct[0])  # no raise


def test_measure_rejects_register_view_target():
    qubits = GridRegister(2, 2, name="qubits")
    cr = ClassicalRegister(4)
    p = Program([qubits], [cr])
    with pytest.raises(TypeError):
        p.measure(qubits.row(0), 0)


def test_measure_all_dim_mismatch_raises():
    qt = QuantumRegister(1, dim=3)
    qb = QuantumRegister(1, dim=2)
    cb = ClassicalRegister(1, dim=2)
    ct = ClassicalRegister(1, dim=3)
    # quantum order (3, 2) vs classical order (2, 3): flat pairing mismatches.
    program = Program([qt, qb], [cb, ct])
    with pytest.raises(ValueError):
        program.measure_all()


def test_measure_all_failure_preserves_existing_instructions():
    p = Program(2, 1)
    p.add(ops.H, 0)

    before = p.instructions

    with pytest.raises(ValueError, match="same number"):
        p.measure_all()

    assert p.instructions == before
    assert len(p.instructions) == 1


def test_measure_all_preserves_mixed_dimension_validation():
    p = Program(
        [
            QuantumRegister(1, name="qutrit", dim=3),
            QuantumRegister(1, name="qubit", dim=2),
        ],
        [
            ClassicalRegister(1, name="dit", dim=3),
            ClassicalRegister(1, name="bit", dim=2),
        ],
    )

    result = p.measure_all()

    assert result is None

    measurement = p.instructions[0]
    assert isinstance(measurement, Measurement)

    assert measurement.targets == (
        p.quantum_registers[0][0],
        p.quantum_registers[1][0],
    )

    assert measurement.outputs == (
        p.classical_registers[0][0],
        p.classical_registers[1][0],
    )


def test_measure_all_rejects_mixed_dimension_mismatch():
    p = Program(
        [
            QuantumRegister(1, name="qutrit", dim=3),
            QuantumRegister(1, name="qubit", dim=2),
        ],
        [
            ClassicalRegister(1, name="bit", dim=2),
            ClassicalRegister(1, name="dit", dim=3),
        ],
    )

    with pytest.raises(ValueError):
        p.measure_all()

    assert p.instructions == ()


def test_measure_all_rejects_integer_outputs_with_multiple_registers():
    p = Program(
        [QuantumRegister(2, name="q")],
        [
            ClassicalRegister(2, name="ca"),
            ClassicalRegister(2, name="cb"),
        ],
    )

    with pytest.raises(TypeError, match="explicit RegisterRef"):
        p.measure_all(outputs=(0, 1))

    assert p.instructions == ()


def test_measure_all_rejects_explicit_output_count_mismatch():
    p = Program(2, 4)

    with pytest.raises(ValueError, match="same number"):
        p.measure_all(outputs=(0,))

    assert p.instructions == ()
