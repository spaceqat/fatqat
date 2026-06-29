import pytest

from qnsim import operations as ops
from qnsim.operations import Operation


def test_fixed_gate_instances_exist_with_uppercase_names():
    assert ops.H.name == "H"
    assert ops.X.name == "X"
    assert ops.Y.name == "Y"
    assert ops.Z.name == "Z"
    assert ops.T.name == "T"
    assert ops.CX.name == "CX"
    assert ops.CZ.name == "CZ"


def test_num_qubits():
    assert ops.H.num_qubits == 1
    assert ops.X.num_qubits == 1
    assert ops.CX.num_qubits == 2
    assert ops.CZ.num_qubits == 2


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


def test_operations_are_frozen():
    with pytest.raises(Exception):
        ops.RX(0.1).theta = 9.0
