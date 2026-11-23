"""Tests for the source instruction dependency graph."""

import fatqat as fq
import fatqat.operations as ops
from fatqat.emulator import ControlChannel, PulseControl, SampledWaveform


def _reasons(dag, source, target):
    return {
        edge.reason
        for edge in dag.edges
        if edge.source == source and edge.target == target
    }


def test_empty_program_has_empty_dag():
    dag = fq.Program(0).dag()

    assert dag.nodes == ()
    assert dag.edges == ()
    assert dict(dag.position_to_node) == {}


def test_disjoint_operations_have_no_dependency_edge():
    program = fq.Program(2)
    program.add(ops.H, 0)
    program.add(ops.X, 1)
    program.add(ops.Z, 0)

    dag = program.dag()

    assert not hasattr(dag, "layers")
    assert not hasattr(dag.nodes[0], "layer")
    assert dag.position_to_node[2] is dag.nodes[2]
    assert _reasons(dag, 0, 2) == {"quantum"}


def test_measurement_to_condition_has_classical_dependency():
    program = fq.Program(4, 1)
    program.add(ops.X, 3, condition=(0, 0))
    program.measure(0, 0)
    program.add(ops.Put, 1, condition=(0, 1))
    program.add(ops.X, 3, condition=(0, 1))
    program.measure(2, 0)

    dag = program.dag()

    assert {
        (edge.source, edge.target) for edge in dag.edges if edge.reason == "classical"
    } == {(0, 1), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)}
    assert dag.nodes[1].node_type == "measurement"
    assert dag.nodes[1].outputs == (program.classical_registers[0][0],)
    assert dag.nodes[2].condition == ((program.classical_registers[0][0], 1),)


def test_barrier_dependencies_are_explicit():
    program = fq.Program(2)
    program.add(ops.H, 0)
    program.add(ops.X, 1)
    program.add(ops.Barrier, (0, 1))
    program.add(ops.Z, 0)

    dag = program.dag()

    assert _reasons(dag, 0, 2) == {"barrier", "quantum"}
    assert _reasons(dag, 1, 2) == {"barrier", "quantum"}
    assert "barrier" in _reasons(dag, 2, 3)


def test_hardware_instructions_preserve_order_on_shared_targets():
    program = fq.Program(3)
    program.add(ops.Put, (0, 1))
    program.add(ops.H, 2)
    program.add(ops.Pair, (0, 1))
    program.add(ops.CZ, (0, 1))
    program.add(ops.Unpair, (0, 1))

    dag = program.dag()

    assert [node.operation_name for node in dag.nodes] == [
        "Put",
        "H",
        "Pair",
        "CZ",
        "Unpair",
    ]
    assert [node.node_type for node in dag.nodes] == [
        "put",
        "operation",
        "pair",
        "operation",
        "unpair",
    ]
    assert dag.node_ids == (0, 1, 2, 3, 4)
    assert {(edge.source, edge.target) for edge in dag.edges} == {
        (0, 2),
        (2, 3),
        (3, 4),
    }


def test_direct_pulse_preserves_order_across_disjoint_targets():
    program = fq.Program(3)
    program.add(ops.H, 0)
    program.add(ops.H, 1)
    pulse = ops.PulseOperation(
        1.0,
        (PulseControl(ControlChannel(), SampledWaveform((0.0, 1.0), (1.0, 1.0))),),
    )
    program.add(pulse)
    program.add(ops.X, 2)
    program.add(pulse)

    dag = program.dag()
    assert [node.name for node in dag.nodes] == [
        "H",
        "H",
        "PulseOperation",
        "X",
        "PulseOperation",
    ]
    assert dag.nodes[2].targets == ()
    assert set(dag.predecessors(2)) == {0, 1}
    assert dag.predecessors(3) == (2,)
    assert dag.predecessors(4) == (3,)


def test_unpaired_two_target_gate_is_not_a_dag_violation():
    program = fq.Program(2)
    program.add(ops.CZ, (0, 1))

    dag = program.dag()

    assert dag.nodes[0].node_type == "operation"
    assert not any(edge.reason == "connectivity" for edge in dag.edges)
