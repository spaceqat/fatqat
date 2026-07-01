import numpy as np
import pytest

import qnsim as qs
from qnsim import operations as ops
from qnsim.backends import MeasurementStep, ResetStep, StateVectorBackend
from qnsim.implementation import ApplyMatrixStep
from qnsim.program import Program


def _lower(p):
    b = StateVectorBackend()
    return b._lower(p, b.resolve_layout(p))


def test_lower_terminal_measurement_is_not_dynamic():
    p = Program(2, 2)
    p.add(ops.H, 0)
    p.add(ops.CZ, (0, 1))
    p.add_measurement(0, 0)
    p.add_measurement(1, 1)
    _plan, facts = _lower(p)
    assert facts.is_dynamic is False
    assert facts.has_measurement is True
    assert facts.has_reset is False


def test_lower_measure_then_gate_on_disjoint_qubit_is_not_dynamic():
    p = Program(2, 2)
    p.add(ops.H, 0)
    p.add_measurement(0, 0)
    p.add(ops.X, 1)  # different qubit -> still fast path
    p.add_measurement(1, 1)
    _plan, facts = _lower(p)
    assert facts.is_dynamic is False


def test_lower_gate_on_measured_qubit_is_dynamic():
    p = Program(1, 1)
    p.add(ops.H, 0)
    p.add_measurement(0, 0)
    p.add(ops.X, 0)  # gate on already-measured qubit
    _plan, facts = _lower(p)
    assert facts.is_dynamic is True


def test_lower_condition_is_dynamic_and_resolves_indices():
    p = Program(2, 2)
    p.add(ops.X, 1, condition=(0, 1))
    plan, facts = _lower(p)
    assert facts.is_dynamic is True
    gate = plan[0]
    assert isinstance(gate, ApplyMatrixStep)
    assert gate.condition == ((0, 1),)


def test_lower_reset_is_dynamic_and_emits_reset_step():
    p = Program(1)
    p.add(qs.ops.Reset(), 0)
    plan, facts = _lower(p)
    assert facts.is_dynamic is True
    assert facts.has_reset is True
    assert plan == [ResetStep(qubit_index=0)]


def test_lower_unknown_gate_raises():
    class FooGate(ops.Operation):
        name = "FOO"
        _num_qubits = 1

    p = Program(1)
    p.add(FooGate(), 0)
    with pytest.raises(qs.UnsupportedOperationError):
        _lower(p)


def test_reset_and_reuse_counts():
    # Put q0 in |1>, measure -> c0=1, reset q0, measure again -> c1=0.
    p = Program(1, 2)
    p.add(ops.X, 0)
    p.add_measurement(0, 0)
    p.add(qs.ops.Reset(), 0)
    p.add_measurement(0, 1)
    counts = StateVectorBackend().run(p, shots=32, seed=0).result().get_counts()
    assert counts == {"01": 32}  # c1=0 (left), c0=1 (right) -> "01"


def test_dynamic_counts_use_snapshots_not_final_index():
    # After reset the final basis state has q0=|0>, but c0 recorded the pre-reset 1.
    # A from-final-index builder would wrongly read c0=0.
    p = Program(1, 1)
    p.add(ops.X, 0)
    p.add_measurement(0, 0)
    p.add(qs.ops.Reset(), 0)
    counts = StateVectorBackend().run(p, shots=10, seed=0).result().get_counts()
    assert counts == {"1": 10}


def test_condition_only_statevector_default_at_many_shots():
    # Dynamic (condition) but no measurement/reset -> statevector available/default.
    p = Program(2, 2)
    p.add(ops.X, 1, condition=(0, 0))  # applies (slot 0 == 0)
    sv = StateVectorBackend().run(p, shots=8).result().get_statevector()
    expected = np.zeros(4, dtype=complex)
    expected[0b10] = 1.0  # qubit 1 -> |1>
    assert np.allclose(sv, expected)


def test_statevector_with_reset_and_many_shots_rejected():
    p = Program(1)
    p.add(qs.ops.Reset(), 0)
    with pytest.raises(qs.BackendValidationError):
        StateVectorBackend().run(
            p, shots=10, result_config=qs.ResultConfig(statevector=True)
        )


def test_conditional_reset_fires_when_guard_true():
    # q0=|1>, measure -> c0=1; reset conditioned on c0==1 fires; second read is 0.
    p = Program(1, 2)
    p.add(ops.X, 0)
    p.add_measurement(0, 0)
    p.add(qs.ops.Reset(), 0, condition=(0, 1))
    p.add_measurement(0, 1)
    counts = StateVectorBackend().run(p, shots=16, seed=0).result().get_counts()
    assert counts == {"01": 16}  # c1=0, c0=1


def test_conditional_reset_skipped_when_guard_false():
    # Same shape, guard c0==0 is false, so reset is SKIPPED and the second read
    # stays 1. This is the case a dropped reset-condition would silently break.
    p = Program(1, 2)
    p.add(ops.X, 0)
    p.add_measurement(0, 0)
    p.add(qs.ops.Reset(), 0, condition=(0, 0))
    p.add_measurement(0, 1)
    counts = StateVectorBackend().run(p, shots=16, seed=0).result().get_counts()
    assert counts == {"11": 16}  # c1=1, c0=1 -> reset did not fire


def test_condition_only_statevector_ignores_shots_value():
    # Non-stochastic dynamic program: the statevector must be produced regardless
    # of `shots` (including shots == 0), because it does not depend on sampling.
    p = Program(2, 2)
    p.add(ops.X, 1, condition=(0, 0))
    sv = (
        StateVectorBackend()
        .run(p, shots=0, result_config=qs.ResultConfig(statevector=True))
        .result()
        .get_statevector()
    )
    expected = np.zeros(4, dtype=complex)
    expected[0b10] = 1.0
    assert np.allclose(sv, expected)
