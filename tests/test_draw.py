"""Translation of a Program into a QuTiP-QIP circuit for drawing.

Asserts on the translated ``QubitCircuit`` structure (gate names, controls,
targets, classical controls) rather than on rendered images.

Both supported QuTiP-QIP APIs are asserted against through `_elements`, which
normalizes the two ways a built circuit exposes its contents. Assertions avoid
QuTiP's own gate spellings where the versions differ (released ``SNOT`` is
master's ``H``) and check the fatqat-controlled facts instead: wire routing,
custom-box labels, and the classical-control wiring.
"""

from dataclasses import dataclass
from typing import ClassVar

import matplotlib
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.draw import to_qubit_circuit
from fatqat.emulator import ControlChannel, PulseControl, SampledWaveform
from fatqat.errors import UnsupportedOperationError
from fatqat.operations import PulseOperation


def _elements(circuit):
    """Return one ``(name, targets, controls, classical_controls)`` per element.

    The released API exposes ``circuit.gates`` holding Gate/Measurement objects;
    master replaces it with ``circuit.instructions`` wrapping an ``operation``.
    Both are normalized here so the assertions below read the same either way.
    """
    if hasattr(circuit, "instructions"):
        # master: instructions wrap the operation; classical controls are cbits
        items = [(item.operation.name, item) for item in circuit.instructions]
    else:
        items = [(getattr(item, "name", "M"), item) for item in circuit.gates]

    def field(item, *names):
        for name in names:
            value = getattr(item, name, None)
            if value:
                return list(value)
        return []

    return [
        (
            name,
            field(item, "targets", "qubits"),
            field(item, "controls"),
            field(item, "classical_controls", "cbits"),
        )
        for name, item in items
    ]


@dataclass(frozen=True)
class _Custom(ops.Operation):
    """A user-defined single-qubit gate with no native QuTiP equivalent."""

    name: ClassVar[str] = "MyG"
    num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class _CustomCY(ops.Operation):
    """A custom two-wire box whose label collides with a QuTiP gate name."""

    name: ClassVar[str] = "CY"
    num_subsystems: ClassVar[int] = 2


def test_gates_translate_to_qutip_equivalents():
    # Native names, a parametric angle, and the controls-first convention
    # (CX -> control 0 / target 1; CCX -> TOFFOLI with controls 0,1).
    program = fq.Program(3)
    program.add(ops.H, 0)
    program.add(ops.RZ(0.7), 1)
    program.add(ops.CX, (0, 1))
    program.add(ops.CCX, (0, 1, 2))
    elements = _elements(to_qubit_circuit(program))

    assert len(elements) == 4
    assert elements[0][1] == [0]  # H on wire 0
    assert elements[1][1] == [1]  # RZ on wire 1
    # controls-first convention: CX -> control 0 / target 1; CCX -> controls 0,1
    assert (elements[2][2], elements[2][1]) == ([0], [1])
    assert (elements[3][2], elements[3][1]) == ([0, 1], [2])


@pytest.mark.parametrize("gate", (ops.CY, ops.CS))
def test_controlled_y_and_s_translate_with_control_first(gate):
    program = fq.Program(2)
    program.add(gate, (0, 1))

    ((_, targets, controls, _),) = _elements(to_qubit_circuit(program))

    assert controls == [0]
    assert targets == [1]


def test_custom_gate_draws_as_a_box_with_its_name():
    program = fq.Program(1)
    program.add(_Custom(), 0)
    ((name, targets, controls, _),) = _elements(to_qubit_circuit(program))

    assert name == "MyG"  # the box carries the operation's own name
    assert targets == [0]
    assert controls == []  # unknown gate: plain box, no control dot


def test_custom_gate_name_collision_stays_a_plain_box():
    program = fq.Program(2)
    program.add(_CustomCY(), (0, 1))

    ((name, targets, controls, _),) = _elements(to_qubit_circuit(program))

    assert name == "CY"
    assert targets == [0, 1]
    assert controls == []


def test_measurement_reset_barrier_and_condition_translate():
    program = fq.Program(2, 2)
    program.measure(0, 0)
    program.add(ops.X, 1, condition=(0, 1))
    program.add(ops.Reset, 0)
    program.add(ops.Barrier, (0, 1))
    elements = _elements(to_qubit_circuit(program))

    assert [name for name, *_ in elements] == ["M", "X", "|0>", "barrier"]
    assert elements[1][3] == [0]  # feedforward draws a classical control wire
    assert elements[2][1] == [0]  # reset box on the reset wire
    assert elements[3][1] == [0, 1]  # barrier spans its operands


def test_barrier_renderers_draw_a_dashed_separator_instead_of_a_box():
    program = fq.Program(2)
    program.add(ops.H, 0)
    program.add(ops.Barrier, (0, 1))
    program.add(ops.CX, (0, 1))

    text = program.draw("text")
    figure = program.draw("matplotlib")
    figure.canvas.draw()

    assert "┊" in text
    non_wire_lines = [
        line for index, line in enumerate(text.splitlines()) if index % 3 != 1
    ]
    assert all("─┊─" not in line for line in non_wire_lines)
    assert "barrier" not in text.lower()
    assert any(line.get_linestyle() == "--" for line in figure.axes[0].lines)
    assert "barrier" not in {label.get_text().lower() for label in figure.axes[0].texts}


def test_qudit_program_draws_as_plain_wires():
    # A diagram does not depict dimension: a qutrit register is one wire per
    # subsystem, and its qudit-only gates fall through to labeled boxes.
    program = fq.Program([fq.QuantumRegister(2, dim=3)])
    program.add(ops.Fourier, 0)
    program.add(ops.Sum, (0, 1))
    circuit = to_qubit_circuit(program)

    # one wire per subsystem (``N`` was renamed ``num_qubits``)
    assert getattr(circuit, "num_qubits", None) or circuit.N == 2
    assert [name for name, *_ in _elements(circuit)] == ["Fourier", "Sum"]


def _bell():
    program = fq.Program(2, 2)
    program.add(ops.H, 0)
    program.add(ops.CX, (0, 1))
    program.measure((0, 1), (0, 1))
    return program


def test_text_renderer_returns_the_terminal_diagram():
    program = fq.Program(1)
    program.add(_Custom(), 0)
    text = program.draw("text")

    # Returned, not printed - so the caller can print, log, or save it.
    assert isinstance(text, str)
    assert "MyG" in text  # a label fatqat controls on both QuTiP versions


def test_hadamard_renderers_use_conventional_h_label():
    program = fq.Program(1)
    program.add(ops.H, 0)

    text = program.draw("text")
    figure = program.draw("matplotlib")
    figure.canvas.draw()

    assert "H" in text
    assert "SNOT" not in text
    assert "H" in {label.get_text() for label in figure.axes[0].texts}


def test_text_renderer_draws_feedforward_condition():
    program = fq.Program(2, 1)
    program.measure(0, 0)
    program.add(ops.X, 1, condition=(0, 0))

    text = program.draw("text")
    classical_line = next(
        line for line in text.splitlines() if line.lstrip().startswith("c0 ")
    )

    assert "X if c0=0" in text
    assert "█" in classical_line


def test_matplotlib_renderer_returns_a_savable_figure():
    figure = _bell().draw()
    assert isinstance(figure, matplotlib.figure.Figure)


def test_matplotlib_renderer_balances_wire_around_outer_gates():
    program = fq.Program(1)
    program.add(ops.H, 0)
    program.add(ops.X, 0)

    figure = program.draw("matplotlib")
    axis = figure.axes[0]
    wire = next(
        line
        for line in axis.lines
        if len(line.get_xdata()) == 2 and line.get_ydata()[0] == line.get_ydata()[1]
    )
    left_gate = min(axis.patches, key=lambda patch: patch.get_x())
    right_gate = max(axis.patches, key=lambda patch: patch.get_x())

    left_idle = left_gate.get_x() - min(wire.get_xdata())
    right_idle = max(wire.get_xdata()) - (right_gate.get_x() + right_gate.get_width())

    assert right_idle == pytest.approx(left_idle)


def test_matplotlib_renderer_draws_feedforward_condition():
    program = fq.Program(2, 1)
    program.measure(0, 0)
    program.add(ops.X, 1, condition=(0, 0))

    figure = program.draw("matplotlib")
    figure.canvas.draw()

    assert "X if c0=0" in {text.get_text() for text in figure.axes[0].texts}


def test_drawing_rejects_pulse_operations_before_qutip_translation():
    program = fq.Program(1)
    program.add(
        PulseOperation(
            1.0,
            (
                PulseControl(
                    ControlChannel(),
                    SampledWaveform((0.0, 1.0), (1.0, 1.0)),
                ),
            ),
        )
    )

    with pytest.raises(UnsupportedOperationError, match="PulseOperation"):
        to_qubit_circuit(program)
