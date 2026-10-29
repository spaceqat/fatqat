# Custom operations

Use a custom operation to update quantum and classical data together or keep
additional data for each shot. Define the operation as a `CustomOperation` and
register a `CustomOperationImplementation` with the simulator's
[`MatrixImplementationMap`](../implementation.md).

## Define and register an operation

This classical-only operation counts its calls and writes the count to the first
classical register entry:

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

To extend the atom-array profile, start with `AtomArraySimulator().implementation_map`.

## State and execution

`execute(operation, state, context)` must return the updated state or a replacement
`EvolutionState`. Keep shot-specific data in `state.classical`, rather than on the
operation or implementation object, which may be reused across shots. User data
is not included in counts; write reported values to `clbits` as in the example.

Use `context.targets` to locate quantum operands and `context.rng` for random
draws that respect the simulation seed. Field meanings, target ordering, and
initialization defaults are described in the API reference below.

Classical conditions are evaluated before `execute`. On the atom-array profile,
custom operations handle absent atoms and pairing requirements themselves.
Attached noise runs afterward and evaluates conditions against the updated
classical register.

## Supported usage

- `Simulator` and `AtomArraySimulator` support custom operations with statevector
  and density-matrix methods on both NumPy and Numba.
- `run_sweep`, unitary/superoperator methods, and expectation calculations do not
  support custom operations.
- Final-state output requires `shots=1` and `result_config={"final_state": True}`.
  Counts without measurements require `result_config={"counts": True}`.
- Process execution requires serializable operations, implementations, and initial
  data. Define classes at module level when using it.

## API reference

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
