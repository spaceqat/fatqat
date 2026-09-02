---
title: "Visualization"
---

# Visualize programs and results

FATQAT provides drawing methods on the object that owns the data. A [`Program`][fatqat.Program] draws its circuit or derives a logical interaction summary, while a [`Result`][fatqat.Result] draws the measurement outcomes from one run.

## Draw a circuit

Use [`Program.draw`][fatqat.Program.draw] for a circuit diagram:

```python
figure = program.draw()
text = program.draw(renderer="text")
```

The default Matplotlib renderer returns a `Figure`. The text renderer returns a string and does not print it. Circuit drawings use one wire per register slot; they do not depict register dimension or hardware connectivity.

![Three-qubit circuit containing H, controlled-X, controlled-Z, and measurement operations.](../assets/generated/guide/visualization-circuit.png)

??? example "Reproduce this figure"

    ```python
    import matplotlib.pyplot as plt
    import fatqat as fq
    import fatqat.operations as ops

    program = fq.Program(3, 3)
    program.add(ops.H, 0)
    program.add(ops.CX, (0, 1))
    program.add(ops.CZ, (1, 2))
    program.measure_all()

    figure, axis = plt.subplots(figsize=(9, 3))
    program.draw(ax=axis)  # (1)!
    ```

    1. Passing `ax=` embeds the circuit in a Matplotlib layout owned by the caller. 

The same composition pattern works for the FATQAT-owned Matplotlib visualizations below.

## Draw logical interaction frequency

First derive the program's logical interaction summary, then draw it:

```python
interaction_frequency = program.interaction_frequency()
figure = interaction_frequency.draw()
```

Each node represents one logical quantum slot. An edge connects two slots used by a source-level two-target operation, and its label is the number of such operations in the whole program. This is a hardware-independent aggregate. It does not represent atom pairing, physical connectivity, or connectivity at a particular instruction layer.

![Five logical quantum slots connected by edges whose labels and widths show interaction frequency.](../assets/generated/guide/visualization-interactions.png)

??? example "Reproduce this figure"

    ```python
    import matplotlib.pyplot as plt
    import fatqat as fq
    import fatqat.operations as ops

    program = fq.Program(5)
    program.add(ops.H, 0)
    program.add(ops.CX, (0, 1))
    program.add(ops.CZ, (1, 2))
    program.add(ops.Swap, (2, 3))
    program.add(ops.CX, (0, 1))
    program.add(ops.CX, (1, 3))
    program.add(ops.CZ, (3, 4))

    figure, axis = plt.subplots(figsize=(6.2, 4.6))
    program.interaction_frequency().draw(
        ax=axis,
        title="Logical interaction frequency",  # (1)!
    )
    ```

    1. The repeated interaction between slots 0 and 1 produces the thicker
       edge labeled `2`; every other edge is labeled `1`.

## Draw measurement outcomes

A result containing counts can draw raw counts or relative frequencies:

```python
figure = result.draw()
figure = result.draw(stat="frequencies")
figure = result.draw(number_to_keep=20, sort="count")
```

`number_to_keep` retains the most frequent outcomes and combines the remainder into an `other` bar. See [Result](../api/result.md#draw-counts) for the complete option reference.

![Bell-state measurement frequencies with bars for the zero-zero and one-one outcomes.](../assets/generated/guide/visualization-counts.png)

??? example "Reproduce this figure"

    ```python
    import matplotlib.pyplot as plt
    import fatqat as fq
    import fatqat.operations as ops

    program = fq.Program(2, 2)
    program.add(ops.H, 0)
    program.add(ops.CX, (0, 1))
    program.measure_all()

    result = (
        fq.simulator.Simulator(runtime="numpy")
        .run(program, shots=1000, simulation_config={"seed": 7})
        .result()
    )

    figure, axis = plt.subplots(figsize=(5.4, 3.4))
    result.draw(
        stat="frequencies",  # (1)!
        ax=axis,
        title="Bell-state outcomes",
    )
    ```

    1. `stat="frequencies"` divides each count by the total number of shots; omit it to show raw counts.

## Embed, style, and save figures

Pass `ax=` to draw into an existing Matplotlib axis:

```python
import matplotlib.pyplot as plt

figure, axis = plt.subplots()
program.interaction_frequency().draw(ax=axis, title="Logical interactions")
```

Visualizations inherit Matplotlib's active style and `rcParams`; FATQAT does not apply a separate palette. A returned figure can be saved with ordinary Matplotlib APIs:

```python
figure.savefig("interactions.png", dpi=200, bbox_inches="tight")
```
