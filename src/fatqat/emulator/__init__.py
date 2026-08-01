"""Private implementation modules for the public superconducting pulse backend.

Use :class:`fatqat.backends.PulseBackend` and the loader helpers exported from
:mod:`fatqat.backends`. The scheduler, resolved (occurrence-bound) pulse
representation, and QuTiP adapter intentionally remain private implementation
details. The pulse-authoring definition and its implementation map -
:class:`fatqat.backends.PulseDefinition` and
:class:`fatqat.backends.PulseImplementationMap` - are public, also re-exported
from :mod:`fatqat.backends`, even though the modules that define them stay
private.
"""
