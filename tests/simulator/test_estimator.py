"""Public Estimator integration for matrix Simulator capabilities."""

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat._index_allocation import _EngineAllocation
from fatqat.errors import UnsupportedOperationError
from fatqat.noise import Loss, ReadoutConfusion
from fatqat.observable import Observable
from fatqat.simulator import AtomArraySimulator, Simulator


class _ReversedAllocationSimulator(Simulator):
    def _allocate_engine_indices(self, program, resource_layout):
        allocation = super()._allocate_engine_indices(program, resource_layout)
        return _EngineAllocation(
            tuple(reversed(allocation.device_operands)),
            tuple(reversed(allocation.system_dims)),
        )


def test_estimator_remaps_logical_factors_through_engine_allocation():
    program = fq.Program(2)
    program.add(ops.X, 0)
    estimator = fq.Estimator(_ReversedAllocationSimulator(runtime="numpy"))

    result = estimator.run(program, Observable([("ZI", 1.0)])).result()

    assert result.get_expectation() == pytest.approx(-1.0)


def test_synthetic_basis_is_ideal_and_readout_confusion_is_applied():
    always_flip = np.array([[0.0, 1.0], [1.0, 0.0]])
    noise = fq.NoiseModel()
    noise.add(fq.noise.Depolarizing(p=1.0), operation=ops.H)
    noise.add(fq.noise.Depolarizing(p=1.0), operation=ops.Sdg)
    noise.add(ReadoutConfusion(always_flip))
    program = fq.Program(1)
    program.add(ops.RY(np.pi / 2), 0)
    estimator = fq.Estimator(Simulator(runtime="numpy", noise=noise))
    observable = Observable([("X", 1.0)])

    sampled = estimator.run(
        program,
        observable,
        shots=64,
        simulation_config={"seed": 4},
    ).result()

    assert sampled.get_expectation() == -1.0
    assert sampled.get_standard_error() == 0.0
    with pytest.raises(UnsupportedOperationError, match="readout confusion"):
        estimator.run(program, observable)


def test_atom_array_carries_terminal_occupancy_into_sampled_tail():
    program = fq.Program(2)
    program.add(ops.Put, 0)
    program.add(ops.RX(np.pi), 0)
    estimator = fq.Estimator(AtomArraySimulator(runtime="numpy"))

    loaded = estimator.run(
        program,
        Observable([("ZI", 1.0)]),
        shots=32,
        simulation_config={"seed": 7},
    ).result()

    assert loaded.get_expectation() == -1.0
    assert loaded.get_standard_error() == 0.0
    with pytest.raises(UnsupportedOperationError, match="unoccupied atom"):
        estimator.run(program, Observable([("IZ", 1.0)]), shots=32)


@pytest.mark.parametrize("shots", [0, 16])
def test_atom_array_identity_cannot_bypass_structural_loss(shots):
    noise = fq.NoiseModel()
    noise.add(Loss(p=0.0), operation=ops.RX)
    program = fq.Program(1)
    program.add(ops.Put, 0)
    program.add(ops.RX(0.1), 0)
    estimator = fq.Estimator(AtomArraySimulator(runtime="numpy", noise=noise))

    with pytest.raises(UnsupportedOperationError, match="carrier loss"):
        estimator.run(program, Observable([("I", 1.0)]), shots=shots)
