"""Kernel identity seam: keyed registration, lowering, and None-key rules."""

import numpy as np

import fatqat as fq
from fatqat._backends.steps import ApplyMatrixStep
from fatqat.simulator import Simulator
from fatqat._backends.steps import BuiltinKernelKey
from fatqat.implementation import (
    MatrixImplementationMap,
    default_matrix_implementation_map,
)


def _lowered_gate_steps(backend, program):
    plan, _ = backend._lower_program(program)
    return [s for s in plan if isinstance(s, ApplyMatrixStep)]


def test_every_default_gate_carries_a_distinct_key():
    default_map = default_matrix_implementation_map()
    keys = set()
    for op_cls in default_map.supported_operations():
        rule = default_map.implementation_for(op_cls)
        key = rule._kernel_key(None, targets=())
        assert key is not None, op_cls.__name__
        keys.add(key)
    # One key per gate family - identity is per-gate, sharing lives in the
    # engine's key-to-kernel table.
    assert len(keys) == len(default_map.supported_operations())
    assert keys == set(BuiltinKernelKey)


def test_lowering_copies_the_key_onto_the_step():
    program = fq.Program(2)
    program.add(fq.ops.H, 0)
    program.add(fq.ops.CX, (0, 1))
    program.add(fq.ops.RZ(0.3), 1)
    steps = _lowered_gate_steps(Simulator(), program)

    assert [s.kernel_key for s in steps] == [
        BuiltinKernelKey.H,
        BuiltinKernelKey.CX,
        BuiltinKernelKey.RZ,
    ]


def test_custom_rule_lowers_with_no_key():
    custom_map = default_matrix_implementation_map()
    # A custom rule that happens to return exactly the X matrix stays
    # None-keyed: identity comes from the selected implementation, never
    # from matrix content.
    custom_map.remove(fq.ops.X)
    custom_map.add(fq.ops.X, np.array([[0, 1], [1, 0]], dtype=complex))
    program = fq.Program(1)
    program.add(fq.ops.X, 0)
    backend = Simulator(implementation_map=custom_map)

    (step,) = _lowered_gate_steps(backend, program)
    assert step.kernel_key is None


def test_device_specific_rule_lowers_with_no_key():
    device_map = MatrixImplementationMap()
    device_map.add(
        fq.ops.CZ,
        np.diag([1, 1, 1, -1]).astype(complex),
        device_operands=(0, 1),
    )
    program = fq.Program(2)
    program.add(fq.ops.CZ, (0, 1))
    backend = Simulator(implementation_map=device_map)

    (step,) = _lowered_gate_steps(backend, program)
    assert step.kernel_key is None


def test_key_survives_pickling_with_the_step():
    import pickle

    step = ApplyMatrixStep(
        matrix=np.eye(2, dtype=complex),
        target_indices=(0,),
        kernel_key=BuiltinKernelKey.I,
    )
    assert pickle.loads(pickle.dumps(step)).kernel_key is BuiltinKernelKey.I
