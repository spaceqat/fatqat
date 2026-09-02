import pytest

import fatqat as fq
from fatqat.compiler import ValidationError
from fatqat.compiler.dialects import (
    GoogleProgram,
    IBMProgram,
    NativeGate,
    NativeMeasure,
    NativeReset,
    verify_google_program,
    verify_ibm_program,
)


def _qref():
    return fq.QuantumRegister(1, name="q")[0]


def test_ibm_and_google_programs_have_distinct_ir_ids():
    assert IBMProgram.IR_ID == "sc.ibm.hardware.v1"
    assert GoogleProgram.IR_ID == "sc.google.hardware.v1"


def test_route_generated_gate_has_generated_by_instead_of_origins():
    gate = NativeGate("native.0", fq.operations.CZ, (0, 1), (), "route.swap.0")

    assert gate.origin_ids == ()
    assert gate.generated_by == "route.swap.0"


def test_native_measure_keeps_original_classical_ref():
    clbit = fq.ClassicalRegister(1, name="c")[0]
    measure = NativeMeasure("native.measure.0", 3, clbit, ("logical.2",))

    assert measure.site == 3
    assert measure.clbit is clbit


def test_ibm_program_accepts_only_ibm_basis_operations():
    qubit = _qref()
    valid = IBMProgram(
        (NativeGate("native.0", fq.operations.SX, (0,), ("logical.0",)),),
        ((qubit, 0),),
        ((qubit, 0),),
    )
    invalid = IBMProgram(
        (NativeGate("native.0", fq.operations.RX(0.2), (0,), ("logical.0",)),),
        ((qubit, 0),),
        ((qubit, 0),),
    )

    verify_ibm_program(valid)
    with pytest.raises(ValidationError, match="IBM native operation"):
        verify_ibm_program(invalid)


def test_google_program_accepts_only_google_basis_operations():
    qubit = _qref()
    valid = GoogleProgram(
        (NativeGate("native.0", fq.operations.RX(0.2), (0,), ("logical.0",)),),
        ((qubit, 0),),
        ((qubit, 0),),
    )
    invalid = GoogleProgram(
        (NativeGate("native.0", fq.operations.SX, (0,), ("logical.0",)),),
        ((qubit, 0),),
        ((qubit, 0),),
    )

    verify_google_program(valid)
    with pytest.raises(ValidationError, match="Google native operation"):
        verify_google_program(invalid)


def test_native_program_accepts_measure_and_reset_with_semantic_origins():
    qubit = _qref()
    clbit = fq.ClassicalRegister(1, name="c")[0]
    program = IBMProgram(
        (
            NativeReset("native.reset.0", 0, ("logical.0",)),
            NativeMeasure("native.measure.0", 0, clbit, ("logical.1",)),
        ),
        ((qubit, 0),),
        ((qubit, 0),),
    )

    verify_ibm_program(program)
