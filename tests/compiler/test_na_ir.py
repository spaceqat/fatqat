import math

import pytest

import fatqat as fq
from fatqat.compiler import ValidationError
from fatqat.compiler.dialects import NAGate, NAMeasure, NAProgram, verify_na_program


def test_na_program_uses_stable_program_refs():
    atoms = fq.QuantumRegister(2, name="atoms")
    bits = fq.ClassicalRegister(2, name="c")
    atom0, atom1 = atoms[0], atoms[1]
    bit0, bit1 = bits[0], bits[1]
    program = NAProgram(
        atoms=(atom0, atom1),
        clbits=(bit0, bit1),
        instructions=(
            NAGate("na.0", ("logical.0",), fq.operations.RY(0.5), (atom0,)),
            NAGate("na.1", ("logical.1",), fq.operations.CZ, (atom0, atom1)),
            NAMeasure("na.2", ("logical.2",), atom0, bit0),
        ),
    )

    verify_na_program(program)

    assert program.atoms[0] is atom0


def test_na_validator_rejects_operation_outside_closed_whitelist():
    atom = fq.QuantumRegister(1, name="atoms")[0]
    program = NAProgram(
        atoms=(atom,),
        clbits=(),
        instructions=(NAGate("na.0", ("logical.0",), fq.operations.X, (atom,)),),
    )

    with pytest.raises(ValidationError, match="unsupported NA gate"):
        verify_na_program(program)


def test_na_validator_rejects_repeated_atom_operand():
    atom = fq.QuantumRegister(1, name="atoms")[0]
    program = NAProgram(
        atoms=(atom,),
        clbits=(),
        instructions=(NAGate("na.0", ("logical.0",), fq.operations.CZ, (atom, atom)),),
    )

    with pytest.raises(ValidationError, match="repeats an atom operand"):
        verify_na_program(program)


def test_na_validator_rejects_nan_rotation_angle():
    atom = fq.QuantumRegister(1, name="atoms")[0]
    program = NAProgram(
        atoms=(atom,),
        clbits=(),
        instructions=(
            NAGate("na.0", ("logical.0",), fq.operations.RY(math.nan), (atom,)),
        ),
    )

    with pytest.raises(ValidationError, match="finite real"):
        verify_na_program(program)


def test_na_validator_rejects_undeclared_atom():
    atoms = fq.QuantumRegister(2, name="atoms")
    program = NAProgram(
        atoms=(atoms[0],),
        clbits=(),
        instructions=(
            NAGate("na.0", ("logical.0",), fq.operations.RZ(0.5), (atoms[1],)),
        ),
    )

    with pytest.raises(ValidationError, match="undeclared atom"):
        verify_na_program(program)


def test_na_validator_rejects_duplicate_operation_id():
    atom = fq.QuantumRegister(1, name="atoms")[0]
    program = NAProgram(
        atoms=(atom,),
        clbits=(),
        instructions=(
            NAGate("na.0", ("logical.0",), fq.operations.RX(0.5), (atom,)),
            NAGate("na.0", ("logical.1",), fq.operations.RZ(0.5), (atom,)),
        ),
    )

    with pytest.raises(ValidationError, match="duplicate NA operation ID"):
        verify_na_program(program)


def test_na_validator_requires_measurements_to_be_terminal():
    atom = fq.QuantumRegister(1, name="atoms")[0]
    clbit = fq.ClassicalRegister(1, name="c")[0]
    program = NAProgram(
        atoms=(atom,),
        clbits=(clbit,),
        instructions=(
            NAMeasure("na.0", ("logical.0",), atom, clbit),
            NAGate("na.1", ("logical.1",), fq.operations.RZ(0.5), (atom,)),
        ),
    )

    with pytest.raises(ValidationError, match="terminal"):
        verify_na_program(program)
