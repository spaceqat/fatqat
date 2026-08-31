---
title: "Custom noise implementations"
---

# Custom noise implementations


This page is for simulator extension authors. Most applications need only
FATQAT's built-in noise types. Use this API to define a new [`Channel`][fatqat.noise.Channel]
or change how a matrix simulator applies an existing type. Pulse-emulator
continuous-noise realizations are family-owned and are not a public extension
surface.

## Simulator implementation map


A [`ChannelImplementationMap`][fatqat.noise.ChannelImplementationMap] rule receives `(channel, *, targets)`,
where `targets` is the ordered tuple of program
[`RegisterRef`][fatqat.RegisterRef] objects, and returns a nonempty tuple of Kraus
matrices acting on the combined target space. It represents one finite channel
application; it is not used as a pulse-emulator Lindblad fallback.

## Define a custom noise type


A custom noise type subclasses [`Channel`][fatqat.noise.Channel] and stores its physical
parameters. Set `Channel.num_subsystems` to the fixed number of targets,
leave it as `None` to use the matched operation's width, or expose a property
when the instance determines the width. Treat instances as immutable after
adding them to a [`NoiseModel`][fatqat.NoiseModel].

The map uses exact concrete types. Given `type(channel) is MyChannel`,
only a rule registered with `add(MyChannel, rule)` matches. Registering
`Channel` or another base class does not implement its subclasses. This
keeps backend support explicit and prevents a new subclass from silently
inheriting an incompatible numerical rule.

## Register and reuse rules


The public map class has these registration operations:

**Registration operations**

| Method | Contract |
| --- | --- |
| [`add`][fatqat.noise.ChannelImplementationMap.add] | Stores the callable for exactly `channel_type`. Adding the type again replaces its rule. FATQAT checks the type and callability immediately, then checks the rule's signature and output when a backend uses it. |
| [`get`][fatqat.noise.ChannelImplementationMap.get] | Returns the callable registered for exactly that type, or `None`. There is no base-class fallback. |
| [`supported_channels`][fatqat.noise.ChannelImplementationMap.supported_channels] | Returns an immutable `frozenset` snapshot of the registered types. |
| [`copy`][fatqat.noise.ChannelImplementationMap.copy] | Returns a map whose registrations can be changed independently. |

## Simulator rules


A matrix rule returns Kraus operators for the matching operation:

```python
def rule(channel, *, targets):
    return (kraus_0, kraus_1)
```

If the ordered targets have dimensions `d_0, d_1, ...`, their combined
dimension is `D = d_0 * d_1 * ...`. The result must be nonempty, and every
element must be a NumPy array with shape `(D, D)`. FATQAT checks only those
structural requirements. It does not verify complete positivity, trace
preservation, Hermiticity preservation, or any parameter convention for a
custom rule.

Start with [`default_channel_implementation_map`][fatqat.noise.default_channel_implementation_map] when you want to keep
FATQAT's built-in simulator rules. See [Simulators](backend-support.md#noise-simulator-support) for
backend limits.

### Minimal example


This custom qubit channel and rule add bit-flip noise while
retaining all built-in channel implementations:

```python
from dataclasses import dataclass

import numpy as np
import fatqat as fq
import fatqat.operations as ops


@dataclass(frozen=True)
class BitFlip(fq.noise.Channel):
    p: float
    num_subsystems = 1

    def __post_init__(self):
        if (
            isinstance(self.p, bool)
            or not isinstance(self.p, (int, float))
            or not 0.0 <= self.p <= 1.0
        ):
            raise ValueError("p must be a real number in [0, 1]")


def bit_flip_rule(channel, *, targets):
    if targets[0].register.dim != 2:
        raise fq.errors.BackendValidationError("BitFlip requires a qubit")
    identity = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    return np.sqrt(1 - channel.p) * identity, np.sqrt(channel.p) * x


channel_map = fq.noise.default_channel_implementation_map()
channel_map.add(BitFlip, bit_flip_rule)

noise = fq.NoiseModel()
noise.add(BitFlip(p=0.05), operation=ops.X)
backend = fq.simulator.Simulator(
    method="density_matrix",
    noise=noise,
    channel_implementation_map=channel_map,
)
```

## API


### Simulator types


::: fatqat.noise.Channel
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.noise.ChannelImplementation
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"
        - "^(?:__call__)$"

::: fatqat.noise.ChannelImplementationMap
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.noise.default_channel_implementation_map
