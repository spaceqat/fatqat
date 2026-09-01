---
title: "Visualization"
---

# Visualize programs and results

FATQAT provides drawing methods on the object that owns the data. A
[`Program`][fatqat.Program] draws its circuit or derives a logical interaction
summary, while a [`Result`][fatqat.Result] draws the measurement outcomes from
one run.

## Draw a circuit

Use [`Program.draw`][fatqat.Program.draw] for a circuit diagram:

```python
figure = program.draw()
text = program.draw(renderer="text")
```

The default Matplotlib renderer returns a `Figure`. The text renderer returns a
string and does not print it. Circuit drawings use one wire per register slot;
they do not depict register dimension or hardware connectivity.

## Draw logical interaction frequency

First derive the program's logical interaction summary, then draw it:

```python
interaction_frequency = program.interaction_frequency()
figure = interaction_frequency.draw()
```

Each node represents one logical quantum slot. An edge connects two slots used
by a source-level two-target operation, and its label is the number of such
operations in the whole program. This is a hardware-independent aggregate. It
does not represent atom pairing, physical connectivity, or connectivity at a
particular instruction layer.

## Draw measurement outcomes

A result containing counts can draw raw counts or relative frequencies:

```python
figure = result.draw()
figure = result.draw(stat="frequencies")
figure = result.draw(number_to_keep=20, sort="count")
```

`number_to_keep` retains the most frequent outcomes and combines the remainder
into an `other` bar. See [Result](../api/result.md#draw-counts) for the complete
option reference.

## Embed, style, and save figures

Pass `ax=` to draw into an existing Matplotlib axis:

```python
import matplotlib.pyplot as plt

figure, axis = plt.subplots()
program.interaction_frequency().draw(ax=axis, title="Logical interactions")
```

Visualizations inherit Matplotlib's active style and `rcParams`; FATQAT does not
apply a separate palette. A returned figure can be saved with ordinary
Matplotlib APIs:

```python
figure.savefig("interactions.png", dpi=200, bbox_inches="tight")
```
