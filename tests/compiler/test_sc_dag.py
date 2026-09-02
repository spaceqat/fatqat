import pytest

import fatqat as fq
from fatqat.compiler.algorithms import (
    DAGCursor,
    build_dag_index,
    topological_order,
)
from fatqat.compiler.dialects import SCNode, SCProgram, SCWire


def _q(index):
    return fq.QuantumRegister(index + 1, name="q")[index]


def test_dag_index_deduplicates_dependency_repeated_by_two_wires():
    q0, q1 = _q(0), _q(1)
    program = SCProgram(
        (q0, q1),
        (),
        (
            SCNode("sc.0", ("logical.0",), fq.operations.CZ, (q0, q1)),
            SCNode("sc.1", ("logical.1",), fq.operations.Swap, (q0, q1)),
        ),
        (SCWire(q0, (0, 1)), SCWire(q1, (0, 1))),
    )

    index = build_dag_index(program)

    assert index.predecessors == ((), (0,))
    assert index.successors == ((1,), ())


def test_cursor_consumes_ready_nodes_and_unlocks_successors():
    q0, q1 = _q(0), _q(1)
    program = SCProgram(
        (q0, q1),
        (),
        (
            SCNode("sc.0", ("logical.0",), fq.operations.RX(0.1), (q0,)),
            SCNode("sc.1", ("logical.1",), fq.operations.RZ(0.2), (q1,)),
            SCNode("sc.2", ("logical.2",), fq.operations.CZ, (q0, q1)),
        ),
        (SCWire(q0, (0, 2)), SCWire(q1, (1, 2))),
    )
    cursor = DAGCursor(build_dag_index(program))

    assert cursor.ready == (0, 1)
    cursor.consume(1)
    assert cursor.ready == (0,)
    cursor.consume(0)
    assert cursor.ready == (2,)
    cursor.consume(2)
    assert cursor.ready == ()
    assert cursor.complete


def test_cursor_rejects_consuming_a_blocked_or_consumed_node():
    q0 = _q(0)
    program = SCProgram(
        (q0,),
        (),
        (
            SCNode("sc.0", ("logical.0",), fq.operations.RX(0.1), (q0,)),
            SCNode("sc.1", ("logical.1",), fq.operations.RZ(0.2), (q0,)),
        ),
        (SCWire(q0, (0, 1)),),
    )
    cursor = DAGCursor(build_dag_index(program))

    with pytest.raises(ValueError, match="not ready"):
        cursor.consume(1)
    cursor.consume(0)
    with pytest.raises(ValueError, match="not ready"):
        cursor.consume(0)


def test_reversed_index_swaps_predecessors_and_successors():
    q0 = _q(0)
    program = SCProgram(
        (q0,),
        (),
        (
            SCNode("sc.0", ("logical.0",), fq.operations.RX(0.1), (q0,)),
            SCNode("sc.1", ("logical.1",), fq.operations.RZ(0.2), (q0,)),
        ),
        (SCWire(q0, (0, 1)),),
    )
    index = build_dag_index(program)

    assert index.reversed().predecessors == index.successors
    assert index.reversed().successors == index.predecessors


def test_topological_order_uses_qubit_then_operation_id_tie_breaking():
    q0, q1 = _q(0), _q(1)
    program = SCProgram(
        (q0, q1),
        (),
        (
            SCNode("sc.z", ("logical.0",), fq.operations.RZ(0.2), (q1,)),
            SCNode("sc.a", ("logical.1",), fq.operations.RX(0.1), (q0,)),
            SCNode("sc.last", ("logical.2",), fq.operations.CZ, (q0, q1)),
        ),
        (SCWire(q0, (1, 2)), SCWire(q1, (0, 2))),
    )

    assert topological_order(program) == (1, 0, 2)
