"""Tests backend resolution of a validated program into an ordered plan."""

import numpy as np

from qnsim import operations as ops
from qnsim.backends import StateVectorBackend, MeasurementStep
from qnsim.implementation import ApplyMatrixStep
from qnsim.program import Program


def _resolve(program):
    backend = StateVectorBackend()
    plan, _facts = backend._lower(program, backend.resolve_layout(program))
    return plan


def test_resolve_preserves_order_and_step_types():
    p = Program(2, 2)
    p.add(ops.H, 0)
    p.add(ops.CZ, (0, 1))
    p.add_measurement(0, 0)
    p.add_measurement(1, 1)
    plan = _resolve(p)
    assert [type(s) for s in plan] == [
        ApplyMatrixStep, ApplyMatrixStep, MeasurementStep, MeasurementStep
    ]


def test_resolve_matrix_step_has_flat_indices_and_matrix():
    p = Program(2)
    p.add(ops.CZ, (0, 1))
    step = _resolve(p)[0]
    assert isinstance(step, ApplyMatrixStep)
    assert step.target_indices == (0, 1)
    assert np.allclose(step.matrix, np.diag([1, 1, 1, -1]))


def test_resolve_measurement_step_has_flat_indices():
    p = Program(1, 1)
    p.add_measurement(0, 0)
    step = _resolve(p)[0]
    assert isinstance(step, MeasurementStep)
    assert (step.measured_indices, step.classical_indices) == ((0,), (0,))
