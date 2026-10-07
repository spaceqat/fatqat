---
title: "Loss"
---

# Loss


[`Loss`][fatqat.noise.Loss] removes physical carriers from an occupancy-aware simulation. It
is not amplitude damping: amplitude damping keeps the subsystem in the modeled
Hilbert space, while loss marks the carrier absent and discards its quantum
correlations.

## Where loss applies


`p` is the loss probability for each selected carrier that is currently
present. FATQAT samples each carrier independently after a matched operation.
An operation-wide registration uses the same probability for every operand;
use `target_positions` to limit loss to particular operands.

[`Loss`][fatqat.noise.Loss] can be attached only to matching operations; it cannot be
registered as background noise. A condition on the matching operation also
controls its loss: a false condition skips the attached loss.

`Loss` requires a backend that tracks occupancy. See
[backend support](backend-support.md#noise-simulator-support) for supported
backends and [the occupancy lifecycle](../simulators/atom-array.md#occupancy-and-loss)
for loading, later operations, measurement, and result behavior.

## API


::: fatqat.noise.Loss
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"
