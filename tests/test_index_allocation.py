"""Tests for the separate private quantum and classical allocations."""

import pytest

from fatqat._index_allocation import _ClassicalAllocation, _EngineAllocation
from fatqat.program import Program
from fatqat.registers import ClassicalRegister


def test_engine_order_is_defined_by_modeled_device_order():
    allocation = _EngineAllocation((9, 3), (2, 3))

    assert allocation.device_operands == (9, 3)
    assert allocation.system_dims == (2, 3)
    assert allocation.n_subsystems == 2
    assert allocation.engine_index(9) == 0
    assert allocation.engine_index(3) == 1
    with pytest.raises(KeyError, match="not part"):
        allocation.engine_index(0)


@pytest.mark.parametrize(
    ("operands", "dims"),
    (
        ((0, 0), (2, 2)),
        ((0,), (0,)),
        ((0,), (-1,)),
        ((0,), (True,)),
        ((0, 1), (2,)),
    ),
)
def test_engine_allocation_rejects_invalid_model(operands, dims):
    with pytest.raises(ValueError):
        _EngineAllocation(operands, dims)


def test_classical_allocation_is_independent_and_declaration_ordered():
    first = ClassicalRegister(2, dim=3, name="first")
    second = ClassicalRegister(1, name="second")
    program = Program([], [first, second])
    allocation = _ClassicalAllocation.from_program(program)

    assert allocation.classical_dims == (3, 3, 2)
    assert allocation.n_clbits == 3
    assert allocation.classical_index(first[1]) == 1
    assert allocation.classical_index(second[0]) == 2


def test_classical_allocation_rejects_foreign_lookalike_register():
    register = ClassicalRegister(1, name="c")
    allocation = _ClassicalAllocation.from_program(Program([], [register]))

    with pytest.raises(KeyError, match="not part"):
        allocation.classical_index(ClassicalRegister(1, name="c")[0])
