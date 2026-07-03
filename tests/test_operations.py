"""Tests operation objects, gate metadata, and immutability."""

import pytest

from qnsim import operations as ops
from qnsim.operations import Operation


@pytest.mark.parametrize("gate,name,n_qubits", [
    (ops.I,  "I",  1),
    (ops.H,  "H",  1),
    (ops.S,  "S",  1),
    (ops.Sdg, "Sdg", 1),
    (ops.X,  "X",  1),
    (ops.Y,  "Y",  1),
    (ops.Z,  "Z",  1),
    (ops.T,  "T",  1),
    (ops.Tdg, "Tdg", 1),
    (ops.CX, "CX", 2),
    (ops.CZ, "CZ", 2),
    (ops.Swap, "Swap", 2),
    (ops.CY, "CY", 2),
])
def test_fixed_gate_name_and_arity(gate, name, n_qubits):
    assert gate.name == name
    assert gate.num_qubits == n_qubits


def test_parametric_gate_is_class_storing_theta():
    g = ops.RX(0.2)
    assert isinstance(g, Operation)
    assert g.name == "RX"
    assert g.theta == 0.2
    assert g.num_qubits == 1
    assert ops.RY(0.3).name == "RY"
    assert ops.RZ(0.4).name == "RZ"


def test_gates_distinguished_by_class():
    assert type(ops.X) is not type(ops.H)
    assert isinstance(ops.X, Operation)


def test_phase_gate_is_class_storing_theta():
    g = ops.Phase(0.7)
    assert isinstance(g, Operation)
    assert g.name == "Phase"
    assert g.theta == 0.7
    assert g.num_qubits == 1


def test_operations_are_frozen():
    with pytest.raises(Exception):
        ops.RX(0.1).theta = 9.0
