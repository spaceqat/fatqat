"""Tests backend resolution of a validated program into an ordered plan."""

import numpy as np
import pytest

import fatqat.operations as ops
from fatqat._backends.steps import ApplyMatrixStep, MeasurementStep
from fatqat.simulator import Simulator
from fatqat.errors import BackendValidationError, UnsupportedOperationError
from fatqat.implementation import (
    MatrixImplementationMap,
    default_matrix_implementation_map,
)
from fatqat.program import Program


def _resolve(program):
    backend = Simulator("SV")
    plan, _facts = backend._lower_program(program)
    return plan


def test_resolve_preserves_order_and_step_types():
    p = Program(2, 2)
    p.add(ops.H, 0)
    p.add(ops.CZ, (0, 1))
    p.measure(0, 0)
    p.measure(1, 1)
    plan = _resolve(p)
    assert [type(s) for s in plan] == [
        ApplyMatrixStep,
        ApplyMatrixStep,
        MeasurementStep,
        MeasurementStep,
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
    p.measure(0, 0)
    step = _resolve(p)[0]
    assert isinstance(step, MeasurementStep)
    assert (step.measured_indices, step.classical_indices) == ((0,), (0,))


# --- target-aware resolution -------------------------------------------------


def test_target_aware_map_allows_registered_target_key():
    cz_rule = default_matrix_implementation_map().implementation_for(ops.CZ)
    m = MatrixImplementationMap()
    m.add(ops.CZ, cz_rule, device_operands=(0, 1))
    backend = Simulator("SV", implementation_map=m)

    p = Program(2)
    p.add(ops.CZ, (0, 1))

    plan, _facts = backend._lower_program(p)

    assert len(plan) == 1
    assert isinstance(plan[0], ApplyMatrixStep)
    assert plan[0].target_indices == (0, 1)


def test_target_aware_map_rejects_illegal_target_key():
    cz_rule = default_matrix_implementation_map().implementation_for(ops.CZ)
    m = MatrixImplementationMap()
    m.add(ops.CZ, cz_rule, device_operands=(0, 1))
    backend = Simulator("SV", implementation_map=m)

    p = Program(2)
    p.add(ops.CZ, (1, 0))

    # Same UnsupportedOperationError type as a wholly-unsupported family
    # (see test below); only the message distinguishes "illegal target"
    # from "no rule at all."
    with pytest.raises(UnsupportedOperationError, match="device operands") as excinfo:
        backend.run(p, result_config={"counts": False, "final_state": True})

    assert isinstance(excinfo.value, BackendValidationError)


def test_target_aware_map_unsupported_family_still_raises_unsupported_operation():
    cz_rule = default_matrix_implementation_map().implementation_for(ops.CZ)
    m = MatrixImplementationMap()
    m.add(ops.CZ, cz_rule, device_operands=(0, 1))
    backend = Simulator("SV", implementation_map=m)

    p = Program(1)
    p.add(ops.X, 0)

    with pytest.raises(UnsupportedOperationError):
        backend.run(p, result_config={"counts": False, "final_state": True})


def test_legacy_default_map_still_resolves_any_target_key():
    backend = Simulator("SV")
    p = Program(2)
    p.add(ops.CZ, (1, 0))

    plan, _facts = backend._lower_program(p)

    assert isinstance(plan[0], ApplyMatrixStep)
    assert plan[0].target_indices == (1, 0)
