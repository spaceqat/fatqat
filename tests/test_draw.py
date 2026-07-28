"""Translation of a Program into a QuTiP-QIP circuit for drawing.

Asserts on the translated ``QubitCircuit`` structure (gate names, controls,
targets, arg values, classical controls) rather than on rendered images.
Self-skips when the optional ``qutip-qip`` dependency is absent, exactly like
the numba- and qiskit-only tests.
"""

from dataclasses import dataclass
from typing import ClassVar

import pytest

import fatqat as fq
import fatqat.operations as op

pytest.importorskip("qutip_qip")

# pylint: disable=wrong-import-position  # needs the importorskip above
from fatqat.draw import draw, to_qubit_circuit

# pylint: enable=wrong-import-position


@dataclass(frozen=True)
class _Custom(op.Operation):
    """A user-defined single-qubit gate with no native QuTiP equivalent."""

    name: ClassVar[str] = "MyG"
    _num_subsystems: ClassVar[int] = 1


def test_gates_translate_to_qutip_equivalents():
    # Native names, a parametric angle, and the controls-first convention
    # (CX -> control 0 / target 1; CCX -> TOFFOLI with controls 0,1).
    program = fq.Program(3)
    program.add(op.H, 0)
    program.add(op.RZ(0.7), 1)
    program.add(op.CX, (0, 1))
    program.add(op.CCX, (0, 1, 2))
    gates = to_qubit_circuit(program).gates

    assert [g.name for g in gates] == ["SNOT", "RZ", "CNOT", "TOFFOLI"]
    assert gates[1].arg_value == 0.7
    assert (gates[2].controls, gates[2].targets) == ([0], [1])
    assert (gates[3].controls, gates[3].targets) == ([0, 1], [2])


def test_custom_gate_draws_as_a_box_with_its_name():
    program = fq.Program(1)
    program.add(_Custom(), 0)
    (gate,) = to_qubit_circuit(program).gates

    assert gate.name == "MyG"
    assert gate.controls is None  # unknown gate: plain box, no control dot


def test_measurement_reset_barrier_and_condition_translate():
    program = fq.Program(2, 2)
    program.measure(0, 0)
    program.add(op.X, 1, condition=(0, 1))
    program.add(op.Reset, 0)
    program.add(op.Barrier, (0, 1))
    elements = to_qubit_circuit(program).gates

    assert [e.name for e in elements] == ["M", "X", "|0>", "barrier"]
    assert elements[0].classical_store == 0
    assert elements[1].classical_controls == [0]  # feedforward wire
    assert elements[3].targets == [0, 1]  # barrier spans its operands


def test_qudit_program_is_rejected():
    program = fq.Program([fq.QuantumRegister(1, dim=3)])
    with pytest.raises(ValueError, match="qubit circuits only"):
        to_qubit_circuit(program)


def _bell():
    program = fq.Program(2, 2)
    program.add(op.H, 0)
    program.add(op.CX, (0, 1))
    program.measure((0, 1), (0, 1))
    return program


def test_text_renderer_returns_the_terminal_diagram():
    text = draw(_bell(), "text")
    assert isinstance(text, str)
    assert "SNOT" in text and "CNOT" in text


def test_matplotlib_renderer_returns_a_savable_figure():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    figure = draw(_bell(), "matplotlib")
    assert isinstance(figure, matplotlib.figure.Figure)
