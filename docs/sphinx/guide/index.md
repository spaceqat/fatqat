# User guide

Start with [Quickstart](quickstart.md), then read [Core concepts](concepts.md)
for the small set of ideas used by every program. The remaining pages are
organized around tasks rather than internal package layers.

Complete examples import `fatqat` as `fq` and `fatqat.operations` as `op`.
Short follow-on snippets state when they assume an existing `program` or import.

## Suggested path

1. [Quickstart](quickstart.md) — install fatqat and produce your first
   counts.
2. [Core concepts](concepts.md) — understand the program, backend, and
   result boundary.
3. [Gates](gates.md), [measurement and conditions](measurement-and-conditions.md),
   and [running and results](running-and-results.md) — write and interpret
   programs.
4. [Noise](noise.md) and [advanced user topics](advanced.md) — opt into
   optional simulation features.

5. [Expectation values](estimator.md) — read observables off a state instead
   of going through counts.

6. [Superconducting transmon emulation](superconducting-pulse.md) covers both
   calibrated gates and direct controls in a three-level transmon model.
7. [Neutral-atom emulation](neutral-atoms.md) compares the three-level and
   two-level atom physics systems by capability.
8. [Three-level atom emulation](atom-3level.md) covers calibrated gates,
   selected-site direct controls, full-pair crosstalk, and qutrit results.
9. [Two-level atom emulation](atom-2level.md) covers global direct controls,
   sampled waveforms, interaction policies, and Lindblad modes.

If an example fails or its output surprises you, see
[Troubleshooting](troubleshooting.md).

```{toctree}
:maxdepth: 1

quickstart
concepts
gates
measurement-and-conditions
running-and-results
noise
advanced
estimator
superconducting-pulse
neutral-atoms
atom-3level
atom-2level
troubleshooting
```
