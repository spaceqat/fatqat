"""Private implementation modules for the public superconducting pulse backend.

Use :class:`fatqat.backends.PulseBackend` and the loader helpers exported from
:mod:`fatqat.backends`. The scheduler, resolved (occurrence-bound) pulse
representation, and QuTiP adapter intentionally remain private implementation
details. The pulse-authoring definition and its implementation map -
:class:`fatqat.backends.PulseDefinition` and
:class:`fatqat.backends.PulseImplementationMap` - are public, also re-exported
from :mod:`fatqat.backends`, even though the modules that define them stay
private.

Internally the package splits into a model-neutral half and a
superconducting half. :mod:`~fatqat.emulator.model_contract` declares the
abstract handle kinds and the ``PhysicsModel`` protocol; ``pulse``,
``scheduling``, and ``engine`` are written against those and import no
concrete model. ``superconducting``, ``superconducting_realization``,
``qutip_adapter``, and ``backend`` supply the transmon model, its
realization rules, its solver binding, and the public backend.
"""
