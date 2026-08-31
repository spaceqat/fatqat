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

The input may be any float-convertible array-like value. It must produce a
finite square matrix of side length at least 2, with entries in `[0, 1]` and
columns that sum to 1 within numerical tolerance. The matrix size must also
match the backend's reported digit dimension.

FATQAT converts the matrix to float and stores its own read-only copy. Changing
the input array later does not affect the noise object.

## How measurements are reported


The backend first samples the true physical outcome and collapses the quantum
state. It then samples the reported digit from the corresponding matrix
column. The reported value is written to classical memory, so subsequent
feedforward sees the confused value. Reusing the measured subsystem still
evolves from its true collapsed state.

## Where it applies


Readout confusion always applies at measurement in
[`NoiseModel`][fatqat.NoiseModel]:

```python
noise.add(confusion)  # Every measured operand.

# Alternatively, target the "q0" device label on a transmon model.
targeted_noise = fq.NoiseModel()
targeted_noise.add(confusion, targets="q0")
```

Omit `targets` to affect every measured operand, or pass one quantum
[`RegisterRef`][fatqat.RegisterRef] or device label. Correlated multi-operand
readout is not supported. Do not pass `operation` or `target_positions`,
even as `None`.

A universal registration cannot coexist with targeted registrations, and the
same target cannot be registered twice. Logical and device-label registrations
that select the same operand are rejected when the program runs with a
concrete layout.

## Simulators


Simulators require the matrix side length to match the measured subsystem's
reported digit dimension. An empty-site erasure bypasses confusion because no
physical digit was measured. See [Simulators](backend-support.md#noise-simulator-support) for backend
and dimension details.

## Pulse emulators


Readout confusion remains a classical reporting step; it is not represented by
a Lindblad operator. See [Pulse emulators](backend-support.md#noise-emulator-support) for each emulator's
reported digit dimension and physical-level mapping.

## API


::: fatqat.noise.ReadoutConfusion
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"
