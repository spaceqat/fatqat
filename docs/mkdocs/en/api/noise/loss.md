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
controls its loss: if the operation is skipped, its attached loss is skipped.

## Occupancy-aware simulators


On an occupancy-aware simulator, the atom lifecycle is always active:

* every site starts empty;
* `Put` loads a fresh `|0>` atom into an empty site;
* a loss hit removes a present atom and its correlations;
* later gates requiring that atom are per-shot no-ops;
* a later `Put` can refill the site;
* measurement of an empty site reports erasure digit `2`.

Erasure bypasses [`ReadoutConfusion`][fatqat.noise.ReadoutConfusion] because no occupied qubit produced a
physical readout digit. Attaching loss to `Put` samples after loading, which
models loading failure or immediate post-load loss. `Put` accepts no other
noise type.

Explicit loading is independent of loss: sites begin empty even when no loss
source is registered, while a matching `Loss(p=0)` simply removes nothing.

Each shot has its own occupancy state. `statevector` and `density_matrix`
support this lifecycle, but their exported states do not encode
[atom occupancy](../simulators/atom-array.md#occupancy-and-loss). `unitary` and
`superop` do not support it. Because a final state depends on sampled loss
history, requesting one requires a single shot.

See [Simulators](backend-support.md#noise-simulator-support) for the built-in occupancy-aware backend and
its method restrictions.

## API


::: fatqat.noise.Loss
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"
