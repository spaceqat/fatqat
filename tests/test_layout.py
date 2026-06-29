import pytest

from qnsim.layout import ResourceLayout
from qnsim.program import Program
from qnsim.registers import QuantumRegister, ClassicalRegister


def test_single_register_layout():
    p = Program(3, 2)
    layout = ResourceLayout.from_program(p)
    assert layout.system_dims == (2, 2, 2)
    assert layout.n_qubits == 3
    assert layout.n_clbits == 2
    assert layout.qubit_index(p.qreg[0][0]) == 0
    assert layout.qubit_index(p.qreg[0][2]) == 2
    assert layout.clbit_index(p.creg[0][1]) == 1


def test_multi_register_flat_concatenation():
    qa = QuantumRegister(2, name="a")
    qb = QuantumRegister(2, name="b")
    p = Program.registers(qreg=[qa, qb])
    layout = ResourceLayout.from_program(p)
    assert layout.qubit_index(qa[0]) == 0
    assert layout.qubit_index(qa[1]) == 1
    assert layout.qubit_index(qb[0]) == 2
    assert layout.qubit_index(qb[1]) == 3
    assert layout.system_dims == (2, 2, 2, 2)


def test_unknown_ref_raises():
    p = Program(1)
    foreign = QuantumRegister(1, name="x")
    layout = ResourceLayout.from_program(p)
    with pytest.raises(KeyError):
        layout.qubit_index(foreign[0])
