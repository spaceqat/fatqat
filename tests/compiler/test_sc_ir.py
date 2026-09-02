import math

import pytest

import fatqat as fq
from fatqat.compiler import ValidationError
from fatqat.compiler.dialects import (
    MEASURE,
    SC_INSTRUCTION_RULES,
    SCNode,
    SCProgram,
    SCWire,
    verify_sc_program,
)


def _q(index):
    return fq.QuantumRegister(index + 1, name="q")[index]


def _c(index):
    return fq.ClassicalRegister(index + 1, name="c")[index]


def test_semantic_swap_is_a_legal_sc_node():
    q0, q1 = _q(0), _q(1)
    c0 = _c(0)
    program = SCProgram(
        qubits=(q0, q1),
        clbits=(c0,),
        nodes=(
            SCNode("sc.0", ("logical.0",), fq.operations.RX(0.5), (q0,)),
            SCNode("sc.1", ("logical.1",), fq.operations.Swap, (q0, q1)),
            SCNode("sc.2", ("logical.2",), MEASURE, (q0,), (c0,)),
        ),
        wires=(SCWire(q0, (0, 1, 2)), SCWire(q1, (1,))),
    )

    verify_sc_program(program)


def test_sc_instruction_contract_is_read_only():
    with pytest.raises(TypeError):
        SC_INSTRUCTION_RULES[object] = (1, 0)


def test_sc_validator_rejects_operation_outside_closed_whitelist():
    q0 = _q(0)
    program = SCProgram(
        (q0,),
        (),
        (SCNode("sc.0", ("logical.0",), fq.operations.H, (q0,)),),
        (SCWire(q0, (0,)),),
    )

    with pytest.raises(ValidationError, match="unsupported SC instruction"):
        verify_sc_program(program)


@pytest.mark.parametrize("theta", [math.nan, math.inf, -math.inf])
def test_sc_validator_rejects_non_finite_real_rotation_angles(theta):
    q0 = _q(0)
    program = SCProgram(
        (q0,),
        (),
        (SCNode("sc.0", ("logical.0",), fq.operations.RX(theta), (q0,)),),
        (SCWire(q0, (0,)),),
    )

    with pytest.raises(ValidationError, match="finite real"):
        verify_sc_program(program)


def test_rotation_operation_rejects_boolean_angles_before_sc_validation():
    with pytest.raises(TypeError, match="must be a real number"):
        fq.operations.RX(True)


def test_sc_validator_rejects_wrong_resource_kind_and_operand_shape():
    q0 = _q(0)
    c0 = _c(0)
    wrong_kind = SCProgram(
        (c0,),
        (),
        (SCNode("sc.0", ("logical.0",), fq.operations.RX(0.1), (c0,)),),
        (SCWire(c0, (0,)),),
    )
    wrong_arity = SCProgram(
        (q0,),
        (),
        (SCNode("sc.0", ("logical.0",), fq.operations.CZ, (q0,)),),
        (SCWire(q0, (0,)),),
    )

    with pytest.raises(ValidationError, match="wrong kind"):
        verify_sc_program(wrong_kind)
    with pytest.raises(ValidationError, match="operand shape"):
        verify_sc_program(wrong_arity)


def test_sc_validator_requires_exact_wire_coverage():
    q0, q1 = _q(0), _q(1)
    program = SCProgram(
        (q0, q1),
        (),
        (SCNode("sc.0", ("logical.0",), fq.operations.CZ, (q0, q1)),),
        (SCWire(q0, (0,)), SCWire(q1, ())),
    )

    with pytest.raises(ValidationError, match="wire coverage"):
        verify_sc_program(program)


def test_sc_validator_rejects_cycle_formed_across_wires():
    q0, q1 = _q(0), _q(1)
    nodes = (
        SCNode("sc.0", ("logical.0",), fq.operations.CZ, (q0, q1)),
        SCNode("sc.1", ("logical.1",), fq.operations.Swap, (q0, q1)),
    )
    program = SCProgram(
        (q0, q1),
        (),
        nodes,
        (SCWire(q0, (0, 1)), SCWire(q1, (1, 0))),
    )

    with pytest.raises(ValidationError, match="cycle"):
        verify_sc_program(program)


def test_sc_validator_requires_measurement_to_be_terminal():
    q0 = _q(0)
    c0 = _c(0)
    program = SCProgram(
        (q0,),
        (c0,),
        (
            SCNode("sc.0", ("logical.0",), MEASURE, (q0,), (c0,)),
            SCNode("sc.1", ("logical.1",), fq.operations.RZ(0.5), (q0,)),
        ),
        (SCWire(q0, (0, 1)),),
    )

    with pytest.raises(ValidationError, match="terminal"):
        verify_sc_program(program)


def test_sc_validator_rejects_duplicate_or_empty_origin_ids():
    q0 = _q(0)
    duplicate = SCProgram(
        (q0,),
        (),
        (SCNode("sc.0", ("logical.0", "logical.0"), fq.operations.RX(0.1), (q0,)),),
        (SCWire(q0, (0,)),),
    )
    empty = SCProgram(
        (q0,),
        (),
        (SCNode("sc.0", (), fq.operations.RX(0.1), (q0,)),),
        (SCWire(q0, (0,)),),
    )

    with pytest.raises(ValidationError, match="origin IDs"):
        verify_sc_program(duplicate)
    with pytest.raises(ValidationError, match="origin IDs"):
        verify_sc_program(empty)
