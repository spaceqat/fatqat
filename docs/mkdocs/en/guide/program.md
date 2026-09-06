# Write quantum computations with Program

[`Program`][fatqat.Program] is what you build and pass to every FatQat
backend. If you are used to a `Circuit` class, this is FatQat's broader
equivalent: it can contain circuit gates, measurements, classical conditions,
symbolic parameters, qubits and qudits, and direct physical controls. A Program
records its resources and ordered instructions without deciding how a backend
will execute them.

This chapter begins with a small qubit circuit, then introduces registers,
feedforward, reusable parameters, mixed local dimensions, drawing, and direct
controls.

## Declare the program's resources

For a small program, pass the number of quantum and classical slots directly:

```python
import fatqat as fq

program = fq.Program(2, 2)  # two qubits and two classical bits
```

As a Program grows, explicit registers give its resources useful names and let
it contain more than one quantum or classical register:

```pycon
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> data = fq.QuantumRegister(2, name="data")
>>> readout = fq.ClassicalRegister(2, name="readout")
>>> program = fq.Program([data], [readout])
```

Indexing a register identifies one of its slots. Explicit references such as
`data[0]` remain unambiguous as the Program grows. Use a bare integer only
when the Program has a single register of that kind.

!!! tip "Need rows and columns?"

    A [`GridRegister`][fatqat.GridRegister] adds row and column views:

    ```pycon
    >>> grid = fq.GridRegister(2, 3, name="grid")
    >>> grid_program = fq.Program([grid])
    >>> grid_program.add(ops.RX(0.2), grid.row(1))
    ```

    The row view groups several targets for one operation. It describes
    structure inside the Program, not placement on a physical device. Choose
    physical placement when executing the Program.

## Compose operations in order

Add operations in the order they should occur:

```pycon
>>> program.add(ops.H, data[0])
>>> program.add(ops.RY(0.3), data[1])
>>> program.add(ops.CX, (data[0], data[1]))
```

Fixed gates such as `H` and `CX` can be added directly. Create a
parameterized gate such as `RY` by passing its angle. For an operation with
several targets, pass one tuple in operand order. For `CX`, the control comes
first and the target second.

These few operations illustrate the calling pattern. The
[operations API reference](../api/operations.md) lists the complete
operation set and exact definitions.

## Use mid-circuit measurement and feedforward

A measurement is an ordered Program instruction, just like a gate. It can end
a circuit or store a value that controls a later operation:

```pycon
>>> dynamic = fq.Program(2, 2)
>>> dynamic.add(ops.H, 0)
>>> dynamic.measure(0, 0)
>>> dynamic.add(ops.X, 1, condition=(0, 1))
>>> dynamic.add(ops.Reset, 0)
>>> dynamic.measure(1, 1)
```

On each shot, the first measurement writes the result from qubit 0 to
classical bit 0. When that value is `1`, `X` flips qubit 1. Reset then returns
qubit 0 to `|0>` without changing the stored value. The final measurement
writes qubit 1 to classical bit 1, so the two classical bits agree:

```pycon
>>> counts = (
...     fq.simulator.Simulator()
...     .run(dynamic, shots=100, simulation_config={"seed": 7})
...     .result()
...     .get_counts()
... )
>>> sorted(counts)
['00', '11']
>>> sum(counts.values())
100
```

Not every backend supports this mid-program behavior. If measurement, reset,
or classical feedforward is unavailable, the backend rejects the Program
rather than silently changing it.

Mid-circuit measurement can make execution dynamic because later operations
may depend on a sampled outcome. Estimating the resulting distribution
therefore requires repeated shots of the same Program.

## Reuse a parameterized Program

A [`Parameter`][fatqat.Parameter] acts as a placeholder for a numeric
operation argument. Calling `assign_parameters` returns a new Program, so the
original template remains available for other values:

```pycon
>>> import numpy as np
>>> theta = fq.Parameter("theta")
>>> template = fq.Program(1)
>>> template.add(ops.RY(theta), 0)
>>> quarter_turn = template.assign_parameters({theta: np.pi / 2})
>>> half_turn = template.assign_parameters({theta: np.pi})
>>> template_backend = fq.simulator.Simulator("SV", runtime="numpy")
>>> quarter_state = template_backend.run(quarter_turn).result().get_statevector()
>>> half_state = template_backend.run(half_turn).result().get_statevector()
>>> round(float(abs(quarter_state[1]) ** 2), 3)
0.5
>>> round(float(abs(half_state[1]) ** 2), 3)
1.0
```

The binding map uses the parameter object, as in `{theta: value}`, rather
than its display name. Reuse one object in several gates when they should share
a value. Use [`ParameterVector`][fatqat.ParameterVector] for an explicitly
ordered group. The [simulation chapter](simulation.md) shows how
[`run_sweep`][fatqat.simulator.Simulator.run_sweep] evaluates a batch of
parameter values from the same template without rebuilding the Program.

## Mix qubits and qutrits

A register's `dim` gives the number of states available to each slot. With
the default `dim=2`, a quantum slot is a qubit and a classical slot is a bit.
Setting `dim=3` creates a qutrit or a classical trit. Registers with different
dimensions can coexist in one Program:

```pycon
>>> qubit = fq.QuantumRegister(1, name="qubit")
>>> qutrit = fq.QuantumRegister(1, name="qutrit", dim=3)
>>> bit = fq.ClassicalRegister(1, name="bit")
>>> trit = fq.ClassicalRegister(1, name="trit", dim=3)
>>> hybrid = fq.Program([qubit, qutrit], [bit, trit])
>>> hybrid.add(ops.X, qubit[0])
>>> hybrid.add(ops.Shift(2), qutrit[0])
>>> hybrid.measure(
...     (qubit[0], qutrit[0]),
...     (bit[0], trit[0]),
... )
```

`X` prepares the qubit in `|1>`, while the dimension-generic `Shift(2)` maps
the qutrit from `|0>` to `|2>`. Measurement pairs each quantum slot with a
classical slot of the same dimension:

```pycon
>>> hybrid_result = (
...     fq.simulator.Simulator("SV")
...     .run(hybrid, shots=20, simulation_config={"seed": 7})
...     .result()
... )
>>> hybrid_result.get_counts_as_tuples()
{(1, 2): 20}
>>> hybrid_result.get_counts()
{'12': 20}
```

Tuple keys list the flattened classical slots in declaration order, making
the qubit value `1` and qutrit value `2` explicit. String keys use the same
left-to-right order. Each operation determines which local dimensions it
accepts, so consult the [qudit operation reference](../api/operations/qudit-gates.md)
when you move beyond `Shift`.

## Draw a Program

Drawing lets you check the instruction order and classical conditions before
execution. Under the hood, FatQat adapts the Program to QuTiP-QIP's
circuit-drawing tools. The examples below use this Program:

```python
import fatqat as fq
import fatqat.operations as ops

program = fq.Program(2, 2)
program.add(ops.H, 0)
program.measure(0, 0)
program.add(ops.X, 1, condition=(0, 1))
program.add(ops.Reset, 0)
program.measure(1, 1)
```

### Text renderer

Pass `"text"` to return a terminal-friendly string, then print it:

```python
diagram = program.draw("text")
print(diagram)
```

```text
                      ┌───────────┐           ┌───┐
 q1 :─────────────────┤ X if c0=1 ├───────────┤ M ├───
                      └─────┬─────┘           └─╥─┘
        ┌───┐  ┌───┐        │        ┌─────┐    ║
 q0 :───┤ H ├──┤ M ├────────│────────┤ |0> ├────║─────
        └───┘  └─╥─┘        │        └─────┘    ║
                 ║          │                   ║
 c1 :════════════║═══───────│──────═════════════╩═════
                 ║          │
                 ║          │
 c0 :════════════╩═══───────█──────═══════════════════
```

The condition appears directly on `X`, and the vertical connector shows that
classical bit 0 controls whether the gate is applied.

### Matplotlib renderer

With no renderer argument, `draw()` returns a Matplotlib figure:

```python
figure = program.draw()
figure.show()
```

![Circuit drawing of a two-qubit Program with measurement, classical feedforward, reset, and final measurement.](../assets/generated/guide/program-drawing.png)

??? example "Reproduce this figure"

    ```python
    import fatqat as fq
    import fatqat.operations as ops

    program = fq.Program(2, 2)
    program.add(ops.H, 0)
    program.measure(0, 0)
    program.add(ops.X, 1, condition=(0, 1))
    program.add(ops.Reset, 0)
    program.measure(1, 1)

    figure = program.draw()
    ```

Both renderers show the instructions recorded in the Program, not the result
of executing them.

!!! important

    The circuit drawing uses one wire per quantum or classical slot and does not
    display a register's local dimension. Qubit and qutrit wires therefore look
    the same. Qudit and custom operations appear as labeled boxes. A
    [`PulseOperation`][fatqat.operations.PulseOperation] cannot be represented by this
    circuit renderer.

A Program can also contain direct physical controls. Add them as
`PulseOperation` instructions without quantum targets; their control channels
identify the physical resources. The
[Hamiltonian-emulation chapter](hamiltonian-emulation.md) visualizes these
controls as waveforms and timelines instead.

## Understand validation

`Program` validates structure as you build it: references must belong to the
Program, each operation must receive the expected number of targets, and
measured quantum and classical dimensions must agree. The selected backend
then checks whether it supports those operations, dimensions, controls, and
classical behavior.

Continue with [choose how much physics to model](execution-models.md) to see
how different backends interpret a Program. Use the
[Program API](../api/program.md) when you need exact accepted forms or
validation behavior.
