"""Public Estimator integration for matrix Simulator capabilities."""

from dataclasses import replace

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat._index_allocation import _EngineAllocation
from fatqat.errors import UnsupportedOperationError
from fatqat.noise import Loss, ReadoutConfusion
from fatqat.observable import NumberOperator, Observable, Projector
from fatqat.simulator import AtomArraySimulator, Simulator
from fatqat.simulator._execution_contract import _TrajectoryCapabilities


class _ReversedAllocationSimulator(Simulator):
    def _allocate_engine_indices(self, program, resource_layout):
        allocation = super()._allocate_engine_indices(program, resource_layout)
        return _EngineAllocation(
            tuple(reversed(allocation.device_operands)),
            tuple(reversed(allocation.system_dims)),
        )


@pytest.mark.parametrize("method", ["statevector", "density_matrix"])
def test_estimator_remaps_logical_factors_through_engine_allocation(method):
    program = fq.Program(2)
    program.add(ops.X, 0)
    estimator = fq.Estimator(
        _ReversedAllocationSimulator(runtime="numpy", method=method)
    )

    result = estimator.run(program, Observable([("ZI", 1.0)])).result()

    assert result.get_expectation() == pytest.approx(-1.0)


def test_classical_support_is_required_only_for_sampled_expectations(monkeypatch):
    backend = Simulator(runtime="numpy")
    monkeypatch.setattr(
        backend._engine,
        "_capabilities",
        replace(backend._engine.capabilities, trajectory=_TrajectoryCapabilities()),
    )
    estimator = fq.Estimator(backend)
    program = fq.Program(1)
    observable = Observable([("Z", 1.0)])

    assert estimator.run(program, observable).result().get_expectation() == 1.0
    with pytest.raises(UnsupportedOperationError, match="classical register"):
        estimator.run(program, observable, shots=8)


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


@pytest.mark.parametrize("shots", [0, 32])
def test_atom_array_expectations_use_evolved_occupancy(shots):
    program = fq.Program(2, 1)
    program.add(ops.Put, 0, condition=(0, 0))
    program.add(ops.RX(np.pi), 0)
    estimator = fq.Estimator(AtomArraySimulator(runtime="numpy"))

    loaded = estimator.run(
        program,
        Observable([("ZI", 1.0)]),
        shots=shots,
        simulation_config={"seed": 7},
    ).result()

    assert loaded.get_expectation() == -1.0
    assert loaded.get_standard_error() == 0.0
    with pytest.raises(UnsupportedOperationError, match="unoccupied atom"):
        estimator.run(program, Observable([("IZ", 1.0)]), shots=shots)


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


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
@pytest.mark.parametrize("method", ["statevector", "density_matrix"])
def test_sampled_observables_start_from_independent_base_states(runtime, method):
    program = fq.Program(2)
    program.add(ops.H, 0)
    program.add(ops.CX, (0, 1))
    estimator = fq.Estimator(Simulator(runtime=runtime, method=method))
    observables = [Observable([(label, 1.0)]) for label in ("XX", "YY", "ZZ", "XX")]

    result = estimator.run(
        program, observables, shots=32, simulation_config={"seed": 7}
    ).result()

    assert result.get_expectation() == pytest.approx([1.0, -1.0, 1.0, 1.0])
    assert result.get_standard_error() == pytest.approx([0.0] * 4)


def _mixed_dimension_case():
    """A qutrit, a qubit, and a qutrit in a superposition of every level."""
    q0 = fq.QuantumRegister(1, dim=3)
    q1 = fq.QuantumRegister(1, dim=2)
    q2 = fq.QuantumRegister(1, dim=3)
    program = fq.Program([q0, q1, q2])
    program.add(ops.SubspaceRX(0.7, (0, 1)), q0[0])
    program.add(ops.SubspaceRY(1.1, (1, 2)), q0[0])
    program.add(ops.H, q1[0])
    program.add(ops.RZ(0.4), q1[0])
    program.add(ops.RX(0.3), q1[0])
    program.add(ops.Fourier, q2[0])
    program.add(ops.SubspaceRZ(0.9, (0, 2)), q2[0])
    observables = [
        Observable.from_sparse([(NumberOperator(), (0,), 1.0)]),
        Observable.from_sparse([([Projector(2), "Z"], (0, 1), 0.5)]),
        Observable.from_sparse([(["X", NumberOperator()], (1, 2), 1.5)]),
        Observable.from_sparse([(["ONE", "Y", Projector(2)], (0, 1, 2), -2.0)]),
        Observable.from_sparse([("Z", (1,), 1.0), ([], (), 0.25)]),
    ]
    return program, observables


def _dense_expectations(state, observables, dims):
    paulis = {
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.diag([1.0, -1.0]).astype(complex),
    }
    values = []
    for observable in observables:
        total = 0.0
        for coefficient, factors in observable.terms:
            locals_ = [np.eye(dim, dtype=complex) for dim in dims]
            for index, factor in factors:
                if isinstance(factor, NumberOperator):
                    locals_[index] = np.diag(np.arange(dims[index])).astype(complex)
                elif isinstance(factor, Projector):
                    locals_[index] = np.zeros((dims[index],) * 2, dtype=complex)
                    locals_[index][factor.level, factor.level] = 1.0
                else:
                    locals_[index] = paulis[str(factor)]
            operator = locals_[0]
            for local in locals_[1:]:
                operator = np.kron(operator, local)
            total += coefficient * np.vdot(state, operator @ state).real
        values.append(total)
    return np.array(values)


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
@pytest.mark.parametrize("method", ["statevector", "density_matrix"])
def test_qudit_expectations_match_a_dense_reference(runtime, method):
    program, observables = _mixed_dimension_case()
    state = (
        Simulator("SV", runtime="numpy")
        .run(program, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )
    reference = _dense_expectations(state, observables, (3, 2, 3))
    estimator = fq.Estimator(Simulator(method, runtime=runtime))

    exact = estimator.run(program, observables).result()
    sampled = estimator.run(
        program, observables, shots=20_000, simulation_config={"seed": 3}
    ).result()

    assert exact.get_expectation() == pytest.approx(reference, abs=1e-12)
    assert np.all(
        np.abs(sampled.get_expectation() - reference)
        <= 5 * sampled.get_standard_error() + 1e-12
    )


def test_sampled_qudit_factors_read_every_level_through_readout_confusion():
    # Level 2 is always reported as level 0: the projector and the number
    # operator read the confused level, so readout covers all three levels.
    report_two_as_zero = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    noise = fq.NoiseModel()
    noise.add(ReadoutConfusion(report_two_as_zero))
    program = fq.Program([fq.QuantumRegister(1, dim=3)])
    program.add(ops.Shift(2), 0)
    estimator = fq.Estimator(Simulator(runtime="numpy", noise=noise))
    observables = [
        Observable.from_sparse([(Projector(0), (0,), 1.0)]),
        Observable.from_sparse([(NumberOperator(), (0,), 1.0)]),
    ]

    sampled = estimator.run(
        program, observables, shots=16, simulation_config={"seed": 2}
    ).result()

    assert sampled.get_expectation() == pytest.approx([1.0, 0.0])
    with pytest.raises(UnsupportedOperationError, match="readout confusion"):
        estimator.run(program, observables)
