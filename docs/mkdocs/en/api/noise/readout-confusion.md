---
title: "ReadoutConfusion"
---

# ReadoutConfusion

[`ReadoutConfusion`][fatqat.noise.ReadoutConfusion] is a classical channel applied to the digit reported
after physical measurement. It changes counts and feedforward inputs, not the
post-measurement quantum state.

## Matrix convention

For confusion matrix C,

$$
C_{r,t}=P(\text{reported}=r\mid\text{true}=t).
$$

Rows are reported digits and columns are true digits. Each column must
therefore sum to 1. A qubit matrix

$$
C=\begin{pmatrix}
    0.98 & 0.04\\
    0.02 & 0.96
  \end{pmatrix}
$$

means that true 0 is reported as 1 with probability 0.02, while true 1 is
reported as 0 with probability 0.04.

## How measurements are reported

A later classical condition uses the reported digit. A later quantum operation on
the measured subsystem starts from the state set by the true measurement outcome.

## Where it applies

Readout confusion always applies at measurement in
[`NoiseModel`][fatqat.NoiseModel]:

```python
import fatqat as fq

confusion = fq.noise.ReadoutConfusion([[0.98, 0.04], [0.02, 0.96]])
noise = fq.NoiseModel()
noise.add(confusion, targets="q0")
```

For target selectors and overlapping rules, see
[`NoiseModel.add()`][fatqat.NoiseModel.add].

## Simulators

Simulators require the matrix side length to match the measured subsystem's
reported digit dimension. An empty-site erasure bypasses confusion because no
physical digit was measured. See [Simulators](backend-support.md#noise-simulator-support) for backend
and dimension details.

## Pulse emulators

The built-in pulse emulators report binary digits, so they require a `2 x 2`
matrix. [`TransmonEmulator`][fatqat.emulator.TransmonEmulator] reports either
excited level as `1` before readout confusion is applied.

## API reference { #api }

::: fatqat.noise.ReadoutConfusion
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"
