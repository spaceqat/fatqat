"""Behavior of custom joint state implementations through public APIs."""

from dataclasses import replace

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.errors import BackendValidationError, UnsupportedOperationError
from fatqat.implementation import (
    CustomOperationImplementation,
    MatrixImplementationMap,
    default_matrix_implementation_map,
)
from fatqat.simulator import (
    AtomArraySimulator,
    ClassicalState,
    EvolutionState,
    Simulator,
)


class Correlated(ops.CustomOperation):
    name = "Correlated"
    num_subsystems = 2


class CorrelatedImplementation(CustomOperationImplementation):
    @staticmethod
    def execute(operation, state, context):
        bit = int(context.rng.integers(2))
        basis_index = sum(bit << target for target in context.targets)
        quantum = np.zeros_like(state.quantum)
        if quantum.ndim == 1:
            quantum[basis_index] = 1
        else:
            quantum[basis_index, basis_index] = 1
        # Return entirely new containers: later operations must use all of them.
        history = state.classical.extra["history"] + [bit]
        return EvolutionState(
            quantum,
            ClassicalState(clbits=[bit, 0, len(history)], extra={"history": history}),
        )


def simulator_for(
    operation, implementation, runtime="numpy", method="statevector", **kwargs
):
    implementations = default_matrix_implementation_map()
    implementations.add(operation, implementation)
    return Simulator(
        runtime=runtime,
        method=method,
        implementation_map=implementations,
        **kwargs,
    )


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
@pytest.mark.parametrize("method", ["statevector", "density_matrix"])
def test_replacement_state_preserves_quantum_classical_correlations_and_feedforward(
    runtime, method
):
    program = fq.Program(2, 3)
    program.add(Correlated(), (1, 0))
    program.add(ops.X, 0, condition=(0, 1))
    program.measure(0, 1)
    simulator = simulator_for(
        Correlated,
        CorrelatedImplementation(),
        runtime,
        method,
    )
    result = simulator.run(
        program,
        shots=64,
        initial_state=EvolutionState(classical=ClassicalState(extra={"history": []})),
        simulation_config={"seed": 7},
    ).result()
    assert set(result.get_counts()) == {"001", "101"}


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
def test_process_execution_preserves_seed_streams(runtime):
    program = fq.Program(2, 3)
    program.add(Correlated(), (1, 0))
    program.add(ops.X, 0, condition=(0, 1))
    program.measure(0, 1)
    simulator = simulator_for(
        Correlated,
        CorrelatedImplementation(),
        runtime,
    )
    serial = simulator.run(
        program,
        shots=64,
        initial_state=EvolutionState(classical=ClassicalState(extra={"history": []})),
        simulation_config={"seed": 7},
    ).result()
    parallel = simulator.run(
        program,
        shots=64,
        initial_state=EvolutionState(classical=ClassicalState(extra={"history": []})),
        simulation_config={
            "seed": 7,
            "shot_parallelism": "processes",
            "max_workers": 2,
        },
    ).result()
    assert serial.get_counts() == parallel.get_counts()


class Prepare(ops.CustomOperation):
    name = "Prepare"
    num_subsystems = 1


def flip_target(quantum, target):
    """Apply X independently for these qubit-only test operations."""
    permutation = np.arange(quantum.shape[0]) ^ (1 << target)
    if quantum.ndim == 1:
        return quantum[permutation]
    return quantum[np.ix_(permutation, permutation)]


class PrepareImplementation(CustomOperationImplementation):
    @staticmethod
    def execute(operation, state, context):
        target = context.targets[0]
        quantum = flip_target(state.quantum, target)
        return replace(state, quantum=quantum)


class FillImplementation(CustomOperationImplementation):
    @staticmethod
    def execute(operation, state, context):
        quantum = np.zeros_like(state.quantum)
        quantum.flat[0] = 1
        return EvolutionState(
            quantum, replace(state.classical, occupied=set(context.targets))
        )


def test_custom_occupancy_replacement_keeps_later_gates_without_builtin_put():
    implementations = AtomArraySimulator().implementation_map
    implementations.add(Prepare, FillImplementation())
    program = fq.Program(1, 1)
    program.add(Prepare(), 0)
    program.add(ops.RX(np.pi), 0)
    program.measure(0, 0)
    simulator = AtomArraySimulator(implementation_map=implementations)
    assert simulator.run(program, shots=4).result().get_counts() == {"1": 4}


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
@pytest.mark.parametrize("method", ["statevector", "density_matrix"])
def test_final_state_export_adopts_custom_quantum_result(runtime, method):
    program = fq.Program(1)
    program.add(Prepare(), 0)
    result = (
        simulator_for(Prepare, PrepareImplementation(), runtime, method)
        .run(program, shots=1, result_config={"counts": False, "final_state": True})
        .result()
    )
    expected = np.array([0, 1])
    if method == "statevector":
        np.testing.assert_allclose(result.get_statevector(), expected)
    else:
        np.testing.assert_allclose(
            result.get_density_matrix(), np.outer(expected, expected)
        )


class Counter(ops.CustomOperation):
    name = "Counter"
    num_subsystems = 0


class RecordedClassicalState(ClassicalState):
    def __init__(self):
        super().__init__(clbits=[0], occupied=set(), extra={"history": []})
        self.records = []
        self.report = self.clbits
        self.carriers = self.occupied


class CounterImplementation(CustomOperationImplementation):
    @staticmethod
    def execute(operation, state, context):
        history = state.classical.extra["history"]
        history.append(1)
        state.classical.records.append(len(history))
        state.classical.clbits[0] = sum(state.classical.records)
        state.classical.occupied.add(0)
        state.classical.report[0] += len(state.classical.carriers)
        return state


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
@pytest.mark.parametrize("shot_parallelism", ["serial", "processes"])
def test_custom_classical_state_is_copied_without_breaking_references(
    runtime, shot_parallelism
):
    initial = EvolutionState(classical=RecordedClassicalState())
    implementations = MatrixImplementationMap()
    implementations.add(Counter, CounterImplementation())
    simulator = Simulator(
        runtime=runtime,
        implementation_map=implementations,
    )
    simulation_config = {"shot_parallelism": shot_parallelism, "max_workers": 2}
    implementations.add(Counter, PrepareImplementation())
    program = fq.Program(1, 1)
    program.add(Counter(), ())
    for _ in range(2):
        assert simulator.run(
            program,
            shots=4,
            initial_state=initial,
            result_config={"counts": True},
            simulation_config=simulation_config,
        ).result().get_counts() == {"2": 4}
    assert initial.classical.extra == {"history": []}
    assert initial.classical.records == []
    assert initial.classical.clbits == [0]
    assert initial.classical.occupied == set()
    initial.classical.extra["history"].append(99)
    assert simulator.run(
        program,
        shots=4,
        initial_state=initial,
        result_config={"counts": True},
        simulation_config=simulation_config,
    ).result().get_counts() == {"3": 4}


def test_registration_and_unsupported_execution_modes():
    implementations = MatrixImplementationMap()
    with pytest.raises(TypeError, match="CustomOperation"):
        implementations.add(ops.X, PrepareImplementation())
    with pytest.raises(TypeError, match="CustomOperationImplementation"):
        implementations.add(Prepare, lambda *args: args[1])
    program = fq.Program(1)
    program.add(Prepare(), 0)
    with pytest.raises(UnsupportedOperationError, match="Prepare is not supported"):
        Simulator().run(program)
    for method in ("unitary", "superop"):
        with pytest.raises(UnsupportedOperationError, match="custom operations"):
            simulator_for(Prepare, PrepareImplementation(), method=method).run(program)
    simulator = simulator_for(Prepare, PrepareImplementation())
    angle = fq.Parameter("angle")
    sweep_program = fq.Program(1)
    sweep_program.add(Prepare(), 0)
    sweep_program.add(ops.RY(angle), 0)
    with pytest.raises(UnsupportedOperationError, match="run_sweep.*custom operations"):
        simulator.run_sweep(sweep_program, {angle: [0.0, 1.0]})
    with pytest.raises(UnsupportedOperationError, match="custom operations"):
        fq.Estimator(simulator).run(program, fq.Observable([("Z", 1.0)]), shots=0)
    with pytest.raises(BackendValidationError, match="shots"):
        simulator.run(program, shots=2, result_config={"final_state": True})


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
@pytest.mark.parametrize("method", ["statevector", "density_matrix"])
def test_run_accepts_combined_initial_quantum_and_classical_data(runtime, method):
    initial = EvolutionState(
        np.array([0, 1], dtype=complex), ClassicalState(clbits=[1, 0])
    )
    program = fq.Program(1, 2)
    program.add(Prepare(), 0, condition=(0, 1))
    program.measure(0, 1)
    simulator = simulator_for(Prepare, PrepareImplementation(), runtime, method)
    for _ in range(2):
        result = simulator.run(program, shots=4, initial_state=initial).result()
        assert result.get_counts() == {"10": 4}
    np.testing.assert_array_equal(initial.quantum, [0, 1])
    assert initial.classical.clbits == [1, 0]


def test_initial_occupancy_runs_gates_without_a_put_or_custom_operation():
    program = fq.Program(1, 1)
    program.add(ops.RX(np.pi), 0)
    program.measure(0, 0)
    initial = EvolutionState(classical=ClassicalState(occupied={0}))
    simulator = AtomArraySimulator()
    assert simulator.run(
        program, shots=4, initial_state=initial
    ).result().get_counts() == {"1": 4}
    assert simulator.run(program, shots=4).result().get_counts() == {"2": 4}
    assert initial.classical.occupied == {0}


class IdentityImplementation(CustomOperationImplementation):
    @staticmethod
    def execute(operation, state, context):
        return state


def test_shared_map_resolves_device_specific_custom_implementations():
    implementations = MatrixImplementationMap()
    implementations.add(Prepare, PrepareImplementation(), device_operands=(0,))
    implementations.add(Prepare, IdentityImplementation(), device_operands=(1,))
    program = fq.Program(2, 2)
    for target in (0, 1):
        program.add(Prepare(), target)
        program.measure(target, target)
    simulator = Simulator(implementation_map=implementations)
    result = simulator.run(program, shots=4).result()
    assert result.get_counts() == {"10": 4}
    unsupported = fq.Program(3)
    unsupported.add(Prepare(), 2)
    with pytest.raises(UnsupportedOperationError, match=r"device operands \(2,\)"):
        simulator.run(unsupported)


def test_one_map_resolves_gates_custom_operations_and_device_channel_rules():
    implementations = default_matrix_implementation_map()
    channel = fq.noise.PauliChannel({"X": 1.0})
    channel_rule = implementations.implementation_for(channel)
    implementations.remove(type(channel))
    implementations.add(channel, channel_rule, device_operands=(0,))
    implementations.add(Prepare, PrepareImplementation())
    noise = fq.NoiseModel()
    noise.add(channel, operation=ops.X)
    simulator = Simulator(implementation_map=implementations, noise=noise)

    def program_on(target):
        program = fq.Program(2, 1)
        program.add(Prepare(), target)
        program.add(ops.X, target)
        program.measure(target, 0)
        return program

    # Device label 0 is quantum index 1; channel lookup must use the device label.
    assert simulator.run(program_on(0), shots=4).result().get_counts() == {"1": 4}
    with pytest.raises(UnsupportedOperationError, match=r"device operands \(1,\)"):
        simulator.run(program_on(1), shots=4)


def test_run_rejects_invalid_initial_occupancy():
    program = fq.Program(1, 1)
    program.measure(0, 0)
    with pytest.raises(BackendValidationError, match="invalid subsystem index"):
        Simulator().run(
            program,
            initial_state=EvolutionState(classical=ClassicalState(occupied={4})),
        )
