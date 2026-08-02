"""Pulse-level emulation: the `Emulator` backend and its authoring types.

`Emulator` simulates calibrated controls on a physics model, rather than
applying gate matrices - that is the sibling package :mod:`fatqat.simulator`.
An emulator is built from an immutable physics model and a calibration bound to
that exact model, both produced by the loaders exported here::

    model = load_physics_model(...)
    calibration = load_calibration_spec(...)
    result = Emulator(model, calibration).run(program).result()

`PulseDefinition` and `PulseImplementationMap` are the pulse-authoring surface,
used to override how an operation is realized as control waveforms.

Internally the package splits into a model-neutral half and a superconducting
half. :mod:`~fatqat.emulator.model_contract` declares the abstract handle kinds
and the ``PhysicsModel`` protocol; ``pulse``, ``scheduling``, and ``engine`` are
written against those and import no concrete model. ``superconducting``,
``superconducting_realization``, ``qutip_adapter``, and ``backend`` supply the
transmon model, its realization rules, its solver binding, and the public
backend. The scheduler, the resolved (occurrence-bound) pulse representation,
and the QuTiP adapter intentionally remain private implementation details.
"""

from __future__ import annotations

from .backend import Emulator
from .pulse import (
    PhaseShift,
    PhaseSwap,
    PulseDefinition,
    PulseImplementationMap,
    SampledControl,
)
from .superconducting import (
    SCTransmonExchangeBuilder,
    load_calibration_spec,
    load_physics_model,
)
from .superconducting_realization import (
    default_superconducting_pulse_implementation_map,
)

__all__ = [
    "Emulator",
    "SCTransmonExchangeBuilder",
    "load_physics_model",
    "load_calibration_spec",
    "PulseDefinition",
    "SampledControl",
    "PhaseShift",
    "PhaseSwap",
    "PulseImplementationMap",
    "default_superconducting_pulse_implementation_map",
]
