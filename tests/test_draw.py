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
import fatqat.operations as op
from fatqat.emulator import ControlChannel, PulseControl
from fatqat.waveforms import SampledWaveform
from fatqat.draw import draw, to_qubit_circuit
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
class _Custom(op.Operation):
    """A user-defined single-qubit gate with no native QuTiP equivalent."""

    name: ClassVar[str] = "MyG"
    _num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class _CustomCY(op.Operation):
    """A custom two-wire box whose label collides with a QuTiP gate name."""

    name: ClassVar[str] = "CY"
    _num_subsystems: ClassVar[int] = 2


def test_gates_translate_to_qutip_equivalents():
    # Native names, a parametric angle, and the controls-first convention
    # (CX -> control 0 / target 1; CCX -> TOFFOLI with controls 0,1).
    program = fq.Program(3)
    program.add(op.H, 0)
    program.add(op.RZ(0.7), 1)
    program.add(op.CX, (0, 1))
    program.add(op.CCX, (0, 1, 2))
    elements = _elements(to_qubit_circuit(program))

    assert len(elements) == 4
    assert elements[0][1] == [0]  # H on wire 0
    assert elements[1][1] == [1]  # RZ on wire 1
    # controls-first convention: CX -> control 0 / target 1; CCX -> controls 0,1
    assert (elements[2][2], elements[2][1]) == ([0], [1])
    assert (elements[3][2], elements[3][1]) == ([0, 1], [2])


@pytest.mark.parametrize("gate", (op.CY, op.CS))
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
    program.add(op.X, 1, condition=(0, 1))
    program.add(op.Reset, 0)
    program.add(op.Barrier, (0, 1))
    elements = _elements(to_qubit_circuit(program))

    assert [name for name, *_ in elements] == ["M", "X", "|0>", "barrier"]
    assert elements[1][3] == [0]  # feedforward draws a classical control wire
    assert elements[2][1] == [0]  # reset box on the reset wire
    assert elements[3][1] == [0, 1]  # barrier spans its operands


def test_qudit_program_draws_as_plain_wires():
    # A diagram does not depict dimension: a qutrit register is one wire per
    # subsystem, and its qudit-only gates fall through to labeled boxes.
    program = fq.Program([fq.QuantumRegister(2, dim=3)])
    program.add(op.Fourier, 0)
    program.add(op.Sum, (0, 1))
    circuit = to_qubit_circuit(program)

    # one wire per subsystem (``N`` was renamed ``num_qubits``)
    assert getattr(circuit, "num_qubits", None) or circuit.N == 2
    assert [name for name, *_ in _elements(circuit)] == ["Fourier", "Sum"]


def _bell():
    program = fq.Program(2, 2)
    program.add(op.H, 0)
    program.add(op.CX, (0, 1))
    program.measure((0, 1), (0, 1))
    return program


def test_text_renderer_returns_the_terminal_diagram():
    program = fq.Program(1)
    program.add(_Custom(), 0)
    text = draw(program, "text")

    # Returned, not printed - so the caller can print, log, or save it.
    assert isinstance(text, str)
    assert "MyG" in text  # a label fatqat controls on both QuTiP versions


def test_text_renderer_draws_and_orders_feedforward_condition():
    program = fq.Program(2, 1)
    program.measure(0, 0)
    program.add(op.X, 1, condition=(0, 0))

    text = draw(program, "text")
    lines = text.splitlines()
    measurement_line = next(line for line in lines if " M " in line)
    q0_line = next(line for line in lines if line.lstrip().startswith("q0 "))
    c0_line = next(line for line in lines if line.lstrip().startswith("c0 "))
    control_column = c0_line.index("█")

    assert any("X if c0=0" in line for line in lines)
    assert measurement_line.index("M") < control_column
    assert q0_line[control_column] == "│"


def test_matplotlib_renderer_returns_a_savable_figure():
    figure = draw(_bell(), "matplotlib")
    assert isinstance(figure, matplotlib.figure.Figure)


def test_matplotlib_renderer_draws_and_orders_feedforward_condition():
    program = fq.Program(2, 1)
    program.measure(0, 0)
    program.add(op.X, 1, condition=(0, 0))

    figure = draw(program, "matplotlib")
    axis = figure.axes[0]
    label = next(text for text in axis.texts if text.get_text() == "X if c0=0")
    gate_x = label.get_position()[0]
    vertical_lines = [
        line
        for line in axis.lines
        if len(set(line.get_xdata())) == 1 and abs(line.get_xdata()[0] - gate_x) < 1e-9
    ]

    assert any(
        min(line.get_ydata()) == 0.0 and max(line.get_ydata()) >= 1.0
        for line in vertical_lines
    )
    assert (
        min(
            patch.get_x() + patch.get_width() / 2
            for patch in axis.patches
            if hasattr(patch, "get_x")
        )
        < gate_x
    )


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
