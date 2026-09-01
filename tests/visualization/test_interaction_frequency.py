"""Tests for the hardware-independent logical interaction graph."""

import matplotlib
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.visualization._dag import _build_interaction_frequency_graph


def test_interaction_frequency_counts_logical_two_qubit_operations():
    program = fq.Program(3)
    program.add(ops.CX, (0, 1))
    program.add(ops.CZ, (0, 1))
    program.add(ops.CX, (1, 2))

    graph = _build_interaction_frequency_graph(program)

    assert len(graph.nodes) == 3
    assert (
        graph.edge_between(
            program.quantum_registers[0][0], program.quantum_registers[0][1]
        ).count
        == 2
    )
    assert (
        graph.edge_between(
            program.quantum_registers[0][1], program.quantum_registers[0][2]
        ).count
        == 1
    )


def test_interaction_frequency_excludes_hardware_directives():
    program = fq.Program(2)
    program.add(ops.Put, (0, 1))
    program.add(ops.Pair, (0, 1))
    program.add(ops.CZ, (0, 1))
    program.add(ops.Unpair, (0, 1))

    graph = _build_interaction_frequency_graph(program)

    assert len(graph.edges) == 1
    assert graph.edges[0].count == 1


def test_interaction_frequency_view_returns_figure():
    program = fq.Program(5, 1)
    program.add(ops.H, 0)
    program.add(ops.H, 3)
    program.add(ops.CX, (0, 1))
    program.add(ops.CZ, (1, 2))
    program.add(ops.Swap, (2, 3))
    program.add(ops.CX, (0, 1))
    program.add(ops.CX, (1, 3))
    program.add(ops.CZ, (3, 4))
    program.measure(4, 0)

    figure = program.draw(view="interaction_frequency")

    assert isinstance(figure, matplotlib.figure.Figure)
    axis = figure.axes[0]
    assert {text.get_text() for text in axis.texts} >= {
        "0",
        "1",
        "2",
        "3",
        "4",
    }
    assert all(
        text.get_text() not in {"q[0]", "q[1]", "q[2]", "q[3]", "q[4]"}
        for text in axis.texts
    )
    assert len(axis.collections) == 1
    assert len(axis.collections[0].get_offsets()) == 5
    assert sorted(line.get_linewidth() for line in axis.lines) == pytest.approx(
        [4.0, 4.0, 4.0, 4.0, 6.5]
    )


def test_interaction_frequency_view_rejects_non_matplotlib_renderer():
    with pytest.raises(ValueError, match="only supports the matplotlib"):
        fq.Program(2).draw(renderer="text", view="interaction_frequency")
