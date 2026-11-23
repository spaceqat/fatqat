# Custom operations

Use a custom operation when a circuit step needs to work directly with the current
shot. It can update the quantum state, write a classical result, or carry data
from one step to the next. The `CustomOperation` describes the instruction, and a
`CustomOperationImplementation` defines what it does when the simulator runs it.

## Define and register an operation { #define-and-register-an-operation }

Register the implementation in the simulator's
[`MatrixImplementationMap`](../implementation.md) so it can dispatch the
instruction. This example defines an operation with no quantum targets. Each call
adds one to a per-shot counter and writes the value to the first classical
register entry.

```python
import fatqat as fq
import fatqat.operations as ops
from fatqat.implementation import (
    CustomOperationImplementation,
    default_matrix_implementation_map,
)
from fatqat.simulator import ClassicalState, EvolutionState, Simulator

class CountCalls(ops.CustomOperation):
    name = "CountCalls"
    num_subsystems = 0

class CountCallsImplementation(CustomOperationImplementation):
    @staticmethod
    def execute(operation, state, context):
        state.classical.extra["calls"] += 1
        state.classical.clbits[0] = state.classical.extra["calls"]
        return state

implementations = default_matrix_implementation_map()
implementations.add(CountCalls, CountCallsImplementation())
simulator = Simulator(implementation_map=implementations)
initial = EvolutionState(classical=ClassicalState(extra={"calls": 0}))
program = fq.Program(1, 1)
count = CountCalls()
program.add(count, ())
program.add(count, ())
result = simulator.run(
    program, shots=4, initial_state=initial, result_config={"counts": True}
).result()
print(result.get_counts())  # {"2": 4}
```

## Work with shot state { #state-and-execution }

Use `execute(operation, state, context)` to read or update the current shot.
Return the updated `state` or a replacement `EvolutionState`. Keep per-shot data
in `state.classical`, as the same operation and implementation can be reused
across shots. Values written to `clbits` appear in the counts when requested.

Use `context.targets` to locate operands and `context.rng` for random draws that
follow the simulation seed. The operation's classical condition is checked before
`execute`. Attached noise runs afterward and evaluates its conditions on the
updated classical register.

## Run and request results { #supported-usage }

Use a statevector or density-matrix run on `Simulator` or `AtomArraySimulator`,
with either NumPy or Numba. On an atom array, start with the simulator's
`implementation_map`. Your implementation is responsible for checking whether its
sites contain atoms and whether they are paired.

Custom operations support direct `run()` calls. Parameter sweeps, operator
methods, and expectation calculations require other operation types.

To inspect a final state, use `shots=1` and `result_config={"final_state": True}`.
To report counts from a program that has no measurement operations, request
`result_config={"counts": True}`.

For process parallelism, define operation and implementation classes at module
level and use serializable initial data.

## API reference { #api-reference }

The entries below describe how to define an instruction, implement its behavior,
and read the state and context passed to it.

::: fatqat.operations.CustomOperation

::: fatqat.implementation.CustomOperationImplementation

::: fatqat.simulator.EvolutionState

::: fatqat.simulator.ClassicalState

::: fatqat.simulator.ExecutionContext
    options:
      members:
        - targets
        - system_dims
        - rng
