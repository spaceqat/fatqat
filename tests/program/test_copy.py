"""Tests Program.copy isolation behavior."""

from fatqat.program import Program
from fatqat import operations as ops


def test_copy_is_independent_for_operations():
    p = Program(2, 2)
    p.add(ops.H, 0)
    q = p.copy()
    q.add(ops.X, 1)
    assert len(p.operations) == 1
    assert len(q.operations) == 2


def test_copy_isolates_metadata():
    p = Program(1, metadata={"src": "orig"})
    q = p.copy()
    q.metadata["src"] = "changed"
    assert p.metadata["src"] == "orig"


def test_copy_preserves_register_tuples():
    p = Program(1)
    q = p.copy()
    assert q.quantum_registers == p.quantum_registers
    assert isinstance(q.quantum_registers, tuple)
    assert q.classical_registers == p.classical_registers
    assert isinstance(q.classical_registers, tuple)
