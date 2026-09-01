"""Tests for the hardware-independent logical interaction graph."""

import matplotlib
import pytest
from cycler import cycler
from matplotlib.colors import to_hex

import fatqat as fq
import fatqat.operations as ops
from fatqat._program_graph import _build_interaction_frequency_graph


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


def test_interaction_frequency_draw_returns_figure():
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

    figure = program.interaction_frequency().draw()

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


def test_interaction_frequency_draw_rejects_non_matplotlib_renderer():
    with pytest.raises(ValueError, match="only supports the matplotlib"):
        fq.Program(2).interaction_frequency().draw(renderer="text")


def test_interaction_frequency_draw_inherits_matplotlib_colors():
    program = fq.Program(2)
    program.add(ops.CZ, (0, 1))

    with matplotlib.rc_context(
        {
            "axes.facecolor": "#172554",
            "axes.prop_cycle": cycler(color=["#22d3ee", "#f472b6"]),
            "text.color": "#f8fafc",
        }
    ):
        figure = program.interaction_frequency().draw()

    axis = figure.axes[0]
    edge_label = next(text for text in axis.texts if text.get_bbox_patch() is not None)
    nodes = axis.collections[0]

    assert to_hex(axis.lines[0].get_color()) == "#22d3ee"
    assert to_hex(nodes.get_facecolors()[0]) == "#f472b6"
    assert to_hex(nodes.get_edgecolors()[0]) == "#f8fafc"
    assert to_hex(edge_label.get_bbox_patch().get_facecolor()) == "#172554"


def test_program_draw_rejects_analysis_view_argument():
    with pytest.raises(TypeError, match="only draws circuits"):
        fq.Program(2).draw(view="interaction_frequency")
