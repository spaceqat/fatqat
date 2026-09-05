import pytest

import fatqat as fq
from fatqat.compiler import ValidationError
from fatqat.compiler.dialects import (
    NativeGate,
    NativeMeasure,
    NativeReset,
    SCNativeProgram,
    verify_sc_native_program,
)
from fatqat.compiler.dialects.sc_native import (
    _RotationNativeProgram,
    _verify_rotation_native_program,
)


def _qref():
    return fq.QuantumRegister(1, name="q")[0]


def test_canonical_and_rotation_programs_have_distinct_ir_ids():
    assert SCNativeProgram.IR_ID == "sc.native.v1"
    assert _RotationNativeProgram.IR_ID == "sc.rotation.native.v1"


def test_route_generated_gate_has_generated_by_instead_of_origins():
    gate = NativeGate("native.0", fq.operations.CZ, (0, 1), (), "route.swap.0")

    assert gate.origin_ids == ()
    assert gate.generated_by == "route.swap.0"


def test_native_measure_keeps_original_classical_ref():
    clbit = fq.ClassicalRegister(1, name="c")[0]
    measure = NativeMeasure("native.measure.0", 3, clbit, ("logical.2",))

    assert measure.site == 3
    assert measure.clbit is clbit


def test_sc_native_program_accepts_only_canonical_basis_operations():
    qubit = _qref()
    valid = SCNativeProgram(
        (NativeGate("native.0", fq.operations.SX, (0,), ("logical.0",)),),
        ((qubit, 0),),
        ((qubit, 0),),
    )
    invalid = SCNativeProgram(
        (NativeGate("native.0", fq.operations.RX(0.2), (0,), ("logical.0",)),),
        ((qubit, 0),),
        ((qubit, 0),),
    )

    verify_sc_native_program(valid)
    with pytest.raises(ValidationError, match="SC native operation"):
        verify_sc_native_program(invalid)


def test_rotation_program_accepts_only_rotation_basis_operations():
    qubit = _qref()
    valid = _RotationNativeProgram(
        (NativeGate("native.0", fq.operations.RX(0.2), (0,), ("logical.0",)),),
        ((qubit, 0),),
        ((qubit, 0),),
    )
    invalid = _RotationNativeProgram(
        (NativeGate("native.0", fq.operations.SX, (0,), ("logical.0",)),),
        ((qubit, 0),),
        ((qubit, 0),),
    )

    _verify_rotation_native_program(valid)
    with pytest.raises(ValidationError, match="rotation native operation"):
        _verify_rotation_native_program(invalid)


def test_native_program_accepts_measure_and_reset_with_semantic_origins():
    qubit = _qref()
    clbit = fq.ClassicalRegister(1, name="c")[0]
    program = SCNativeProgram(
        (
            NativeReset("native.reset.0", 0, ("logical.0",)),
            NativeMeasure("native.measure.0", 0, clbit, ("logical.1",)),
        ),
        ((qubit, 0),),
        ((qubit, 0),),
    )

    verify_sc_native_program(program)
