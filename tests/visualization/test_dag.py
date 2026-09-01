"""Tests for the private instruction dependency DAG."""

from dataclasses import FrozenInstanceError

import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.visualization._dag import _build_instruction_dag


def _reasons(dag, source, target):
    return {
        edge.reason
        for edge in dag.edges
        if edge.source == source and edge.target == target
    }


def test_empty_program_has_empty_dag():
    dag = _build_instruction_dag(fq.Program(0))

    assert dag.nodes == ()
    assert dag.edges == ()
    assert dict(dag.position_to_node) == {}


def test_disjoint_operations_have_no_dependency_edge():
    program = fq.Program(2)
    program.add(ops.H, 0)
    program.add(ops.X, 1)
    program.add(ops.Z, 0)

    dag = _build_instruction_dag(program)

    assert not hasattr(dag, "layers")
    assert not hasattr(dag.nodes[0], "layer")
    assert dag.position_to_node[2] is dag.nodes[2]
    assert _reasons(dag, 0, 2) == {"quantum"}


def test_measurement_to_condition_has_classical_dependency():
    program = fq.Program(2, 1)
    program.measure(0, 0)
    program.add(ops.X, 1, condition=(0, 1))

    dag = _build_instruction_dag(program)

    assert _reasons(dag, 0, 1) == {"classical"}
    assert dag.nodes[0].node_type == "measurement"
    assert dag.nodes[0].outputs == (program.classical_registers[0][0],)
    assert dag.nodes[1].condition == ((program.classical_registers[0][0], 1),)


def test_barrier_dependencies_are_explicit():
    program = fq.Program(2)
    program.add(ops.H, 0)
    program.add(ops.X, 1)
    program.add(ops.Barrier, (0, 1))
    program.add(ops.Z, 0)

    dag = _build_instruction_dag(program)

    assert _reasons(dag, 0, 2) == {"barrier", "quantum"}
    assert _reasons(dag, 1, 2) == {"barrier", "quantum"}
    assert "barrier" in _reasons(dag, 2, 3)


def test_hardware_connectivity_instructions_are_not_generic_dag_nodes():
    program = fq.Program(2)
    program.add(ops.Put, (0, 1))
    program.add(ops.Pair, (0, 1))
    program.add(ops.CZ, (0, 1))
    program.add(ops.Unpair, (0, 1))

    dag = _build_instruction_dag(program)

    assert [node.operation_name for node in dag.nodes] == ["CZ"]
    assert dag.nodes[0].node_id == 2
    assert dag.position_to_node[2] is dag.nodes[0]
    assert dag.node_by_id[2] is dag.nodes[0]


def test_hardware_only_program_has_an_empty_generic_dag():
    program = fq.Program(3)
    program.add(ops.Pair, (0, 1))
    program.add(ops.Pair, (1, 2))

    dag = _build_instruction_dag(program)
    assert dag.nodes == ()
    assert dict(dag.position_to_node) == {}


def test_unpaired_two_target_gate_is_not_a_dag_violation():
    program = fq.Program(2)
    program.add(ops.CZ, (0, 1))

    dag = _build_instruction_dag(program)

    assert dag.nodes[0].node_type == "operation"
    assert not any(edge.reason == "connectivity" for edge in dag.edges)


def test_dag_view_models_are_not_mutable_or_publicly_exported():
    dag = _build_instruction_dag(fq.Program(1))

    with pytest.raises(FrozenInstanceError):
        dag.nodes = ()

    assert not hasattr(fq.visualization, "_InstructionDAG")
    assert "_InstructionDAG" not in fq.visualization.__all__
