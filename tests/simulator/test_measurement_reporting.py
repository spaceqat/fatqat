"""Physical measurement, reported-digit mapping, and classical confusion."""

import numpy as np

from fatqat.simulator import EvolutionState
from fatqat.simulator._engine.state import _LoweredBasisState
from fatqat.simulator._execution_contract import (
    _ResultRequest,
    _RunContext,
)
from fatqat._backends.steps import ApplyMatrixStep, MeasurementStep
from fatqat.implementation.matrices import shift_matrix
from fatqat.simulator._engine.engines.numpy_engine import NumpySVEngine
from fatqat.simulator._engine._execution_policy import _ExecutionPolicy


def test_reported_digit_mapping_precedes_confusion_and_feedforward():
    """A physical qutrit outcome can report a bit without changing its state."""
    always_flip = np.array([[0.0, 1.0], [1.0, 0.0]])
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    engine = NumpySVEngine()
    plan = (
        ApplyMatrixStep(matrix=shift_matrix(3, 2), target_indices=(0,)),
        MeasurementStep(
            measured_indices=(0,),
            classical_indices=(0,),
            # The physical |2> outcome reports 1, then the classical
            # readout confusion flips that report to 0.
            reported_digit_maps=((0, 1, 1),),
            confusions=(always_flip,),
        ),
        ApplyMatrixStep(matrix=x, target_indices=(1,), condition=((0, 0),)),
    )
    context = _RunContext(
        request=_ResultRequest(counts=True, final_state=True),
        system_dims=(3, 2),
        n_clbits=1,
        shots=1,
        seed=9,
    )
    policy = _ExecutionPolicy(
        execution_path="per_shot",
        shot_strategy="serial",
        kernel_strategy="serial",
        worker_limit=1,
        fusion=False,
    )
    payload = engine.prepare_payload(
        plan,
        system_dims=context.system_dims,
        n_clbits=context.n_clbits,
        policy=policy,
    )
    result = engine.execute_payload(
        context,
        payload,
        policy,
        initial_state=EvolutionState(_LoweredBasisState((0, 0))),
    )

    # Feedforward receives the confused reported digit, so its X fires.
    assert result.outcome_keys.tolist() == [[0]]
    # The physical posterior remains |2, 1>, not the reported classical 0.
    assert np.allclose(result.state, [0, 0, 0, 0, 0, 1])
