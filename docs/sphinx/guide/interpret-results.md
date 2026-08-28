# Ask questions of a run

Every FatQat execution crosses the same boundary: submit a
{py:class}`~fatqat.Program`, receive a {py:class}`~fatqat.Job`, and call
{py:meth}`~fatqat.Job.result` for the answer. The useful result depends on the
question you asked.

We will build one Bell Program and use it four ways: observed outcomes, final
state, implemented map, and expectation value.

## Build the computation once

Keep the Program unmeasured at first. That leaves state-, map-, and
observable-based questions open:

```{doctest}
>>> import numpy as np
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> bell = fq.Program(2, 2)
>>> bell.add(ops.H, 0)
>>> bell.add(ops.CX, (0, 1))
>>> backend = fq.simulator.Simulator(method="statevector", runtime="numpy")
```

## What outcomes occurred?

Copy the Program and append measurement when the output you want is a sampled
classical distribution:

```{doctest}
>>> measured_bell = bell.copy()
>>> measured_bell.measure_all()
>>> counts_result = backend.run(
...     measured_bell,
...     shots=400,
...     simulation_config={"seed": 7},
... ).result()
>>> counts = counts_result.get_counts()
>>> sum(counts.values())
400
>>> set(counts) <= {"00", "11"}
True
```

Only the correlated Bell outcomes occur. Count strings display the highest
classical slot on the left and slot 0 on the right. A deliberately asymmetric
example makes that order visible:

```{doctest}
>>> order_demo = fq.Program(2, 2)
>>> order_demo.add(ops.X, 0)
>>> order_demo.measure_all()
>>> backend.run(
...     order_demo,
...     shots=8,
...     simulation_config={"seed": 7},
... ).result().get_counts()
{'01': 8}
```

Qubit 0 is `1`, so classical slot 0 appears as the rightmost digit. Use
{py:meth}`~fatqat.Result.get_counts_as_tuples` when code benefits from slot 0
being first rather than from display strings.

## What state did the run reach?

Run the unmeasured Program and request its natural final state:

```{doctest}
>>> state_result = backend.run(
...     bell,
...     result_config={"counts": False, "final_state": True},
... ).result()
>>> sorted(state_result.available_data)
['statevector']
>>> state = state_result.get_statevector()
>>> np.round(np.abs(state) ** 2, 6).tolist()
[0.5, 0.0, 0.0, 0.5]
```

`available_data` tells you what this run actually produced before you choose
an accessor. Here the state has equal probability on `|00>` and `|11>`. A
density-matrix run expresses the same pure state as a matrix and can also
represent exact mixed evolution. Hamiltonian emulators may return states that
include non-computational physical levels or model subsystems not addressed by
the logical Program.

## What transformation did the Program implement?

An unmeasured coherent Program can itself be the object of study. Running it
with the unitary method returns a matrix whose first column is the state that
the Program prepares from `|00>`:

```{doctest}
>>> unitary = (
...     fq.simulator.Simulator(method="unitary", runtime="numpy")
...     .run(bell)
...     .result()
...     .get_unitary()
... )
>>> np.allclose(unitary[:, 0], state)
True
```

Use this route for a small coherent block whose complete action matters. A
super-operator extends the idea to complete channels at substantially greater
cost.

## What physical quantity do I care about?

Counts are indirect when the answer is already an expectation value such as a
correlation or magnetization. {py:class}`~fatqat.Estimator` evolves the
unmeasured Program through a backend and evaluates an
{py:class}`~fatqat.Observable` on the resulting state:

```{doctest}
>>> estimator = fq.Estimator(
...     fq.simulator.Simulator(method="statevector", runtime="numpy")
... )
>>> zz = fq.Observable([("ZZ", 1.0)])
>>> z_on_qubit_zero = fq.Observable([("IZ", 1.0)])
>>> exact = estimator.run(bell, [zz, z_on_qubit_zero]).result()
>>> np.round(exact.get_expectation(), 6).tolist()
[1.0, 0.0]
```

The Bell pair is perfectly correlated, so `<ZZ> = 1`. Either individual qubit
is balanced between zero and one, so `<Z_0> = 0`. Observable labels place
qubit 0 at the right, matching count-string order.

By default, Estimator calculates an exact value from the final state. A
positive shot count instead shows the statistical precision of a finite-shot
request:

```{doctest}
>>> sampled = estimator.run(
...     bell,
...     z_on_qubit_zero,
...     shots=400,
...     simulation_config={"seed": 7},
... ).result()
>>> bool(abs(sampled.get_expectation()) < 0.2)
True
>>> sampled.get_std() > 0.0
True
```

The estimate fluctuates around zero, and `get_std()` reports its standard
error. Increase the shots when statistical precision—not state evolution—is
the limiting factor.

For formal state-axis, operator-vectorization, and observable-shape contracts,
use the {doc}`Result <../api/result>`, {doc}`Simulator <../api/simulator>`, and
{doc}`Estimator <../api/estimator>` references.

The next question is often whether that answer survives realistic errors.
[Compare the same Program ideally and noisily](ideal-and-noisy.md) changes the
execution model while leaving the Bell Program untouched.
