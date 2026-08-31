# Write quantum computations with Program

[`Program`][fatqat.Program] is the object you build and pass to every FatQat
backend. If you are used to a `Circuit` class, this is the object you are
looking for. FatQat uses the broader name because a Program can describe more
than circuit gates: measurements and classical conditions, symbolic
parameters, qubits and qudits, and direct physical controls. It records the
logical system and its instructions in execution order without choosing how a
backend will execute them.

Start with a familiar qubit circuit, then add registers, feedforward,
parameters, mixed local dimensions, drawing, and direct controls without
changing the authoring pattern.

## Declare the logical system

For a small qubit circuit, counts are convenient shorthand:

```python
import fatqat as fq

program = fq.Program(2, 2)  # two qubits and two classical bits
```

Explicit registers give resources useful names and become necessary when a
Program contains more than one register:

```pycon
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> data = fq.QuantumRegister(2, name="data")
>>> readout = fq.ClassicalRegister(2, name="readout")
>>> program = fq.Program([data], [readout])
```

Indexing a register returns a reference to one of its slots. Explicit
references such as `data[0]` remain unambiguous as the Program grows; a bare
integer is only convenient when there is exactly one register of that kind.

!!! tip "Need rows and columns?"

    A [`GridRegister`][fatqat.GridRegister] adds logical row and column selections:

    ```pycon
    >>> grid = fq.GridRegister(2, 3, name="grid")
    >>> grid_program = fq.Program([grid])
    >>> grid_program.add(ops.RX(0.2), grid.row(1))
    ```

    The row is a grouped target for this operation. It describes logical
    structure, not a placement on physical hardware; placement is chosen when the
    Program is executed.

## Compose operations in order

Add operations in the order they should occur:

```pycon
>>> program.add(ops.H, data[0])
>>> program.add(ops.RY(0.3), data[1])
>>> program.add(ops.CX, (data[0], data[1]))
```

Fixed gates such as `H` and `CX` are ready-to-use values. Parameterized gates
such as `RY` are created with their numeric argument. A multi-target operation
receives one tuple in operand order; for `CX`, the control comes first and the
target second.

These few operations illustrate the calling pattern. The
[operations API reference](../api/operations.md) lists the complete
operation set and exact definitions.

## Observe and react inside a Program

Measurement is another ordered instruction. It can finish a circuit, or it
can supply a classical value to a later condition:

```pycon
>>> dynamic = fq.Program(2, 2)
>>> dynamic.add(ops.H, 0)
>>> dynamic.measure(0, 0)
>>> dynamic.add(ops.X, 1, condition=(0, 1))
>>> dynamic.add(ops.Reset, 0)
>>> dynamic.measure(1, 1)
```

On each shot, the `X` acts on qubit 1 only if the earlier measurement wrote
`1` to classical bit 0. Reset then prepares qubit 0 in `|0>` without erasing
the stored classical value. The final measurement shows that classical bit 1
follows bit 0:

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

Backends decide whether they can preserve this mid-program behavior. A
backend that does not support measurement, reset, or classical feedforward
rejects the Program rather than silently changing it.

## Make a reusable template

A [`Parameter`][fatqat.Parameter] stands in for a numeric operation argument.
Binding returns a new Program, leaving the template available for another
value:

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

Bindings use the parameter object, not its display name. Reuse one object in
several gates when they should share a value; use
[`ParameterVector`][fatqat.ParameterVector] for an explicitly ordered group. The
[simulation chapter](simulation.md) executes a whole batch of values without
rebuilding the template.

## Mix qubits and qutrits

A register's `dim` is the number of local basis states. The default `2`
creates a qubit or bit; `dim=3` creates a qutrit or three-valued classical
digit. Different dimensions can coexist in one hybrid Program:

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
```

Tuple keys put the flattened classical slots in declaration order, making the
qubit value `1` and qutrit value `2` explicit. Operations decide which local
dimensions they accept, so use the [qudit operation reference](../api/operations/qudit-gates.md) when you move beyond `Shift`.

## Inspect what was written

Call `draw()` whenever a visual check is useful:

```python
diagram = dynamic.draw("text")
print(diagram)
```

The Matplotlib renderer is the default; the text renderer returns a string.
Both show instruction structure rather than execution. In this example, the
conditional operation is labeled `X if c0=1`, making the feedforward visible
before you run it. The [quickstart](quickstart.md) shows the Matplotlib output.

!!! important

    The circuit drawing uses one wire per quantum or classical slot and does not
    display a register's local dimension. Qubit and qutrit wires therefore look
    the same. Qudit and custom operations appear as labeled boxes. A
    [`PulseOperation`][fatqat.operations.PulseOperation] cannot be represented by this
    circuit renderer.

Direct physical controls still enter the same Program as a `PulseOperation`,
added without logical targets because its control channels identify physical
resources. The [Hamiltonian-emulation chapter](hamiltonian-emulation.md)
visualizes those controls as waveforms and timelines instead.

## What Program validates

`Program` checks structural questions while you write: whether references
belong to it, target counts match, and measured quantum/classical dimensions
agree. The selected backend later answers a different question: whether it
can realize those operations, dimensions, controls, and classical behavior.

Continue with [Choose how much physics to model](execution-models.md) to see
how one authoring interface leads to FatQat's different execution paths. Use
the [Program API](../api/program.md) when you need exact accepted forms or
validation behavior.
