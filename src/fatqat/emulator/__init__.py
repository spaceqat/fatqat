"""Pulse emulators and model-specific pulse-authoring values.

`TransmonEmulator` and `Atom2LevelEmulator` execute
`fatqat.Program` objects against physical models. Ordinary gates use a
`PulseImplementationMap`; channel-addressed
`fatqat.operations.PulseOperation` values carry direct controls created from a
model's selectors.

The ``method`` constructor argument selects the mathematical representation
and corresponding `fatqat.Result` accessor. Pulse emulators support
``"statevector"`` (the default), ``"density_matrix"``, and ``"unitary"``;
``"SV"`` and ``"DM"`` are accepted aliases.

A gate-capable emulator uses its packaged gate implementation map when none is
supplied::

    model = TransmonModel.from_document(model_document)
    result = TransmonEmulator(model, method="statevector").run(program).result()

Use calibration documents with the standard map builders to customize gate
realizations. `PulseDefinition` and `PulseImplementationMap` support custom
gate rules. Continuous-noise realizations are selected by the emulator family.
"""

from __future__ import annotations

from .atom_arrangement import AtomArrangement
from ._model_catalog import available_model_documents, load_model_document
from .atom_2level import (
    Atom2LevelEmulator,
    Atom2LevelModel,
)
from ._core.pulse import (
    PhaseShift,
    PhaseSwap,
    PulseDefinition,
    PulseImplementationMap,
)
from .._pulse_values import ControlChannel, PulseControl
from .._waveforms import SampledWaveform
from .superconducting import (
    TransmonEmulator,
    TransmonCalibration,
    TransmonModel,
    default_transmon_calibration,
    default_transmon_gate_implementation_map,
)

__all__ = [
    "TransmonEmulator",
    "Atom2LevelEmulator",
    "TransmonModel",
    "TransmonCalibration",
    "PulseDefinition",
    "ControlChannel",
    "PulseControl",
    "SampledWaveform",
    "PhaseShift",
    "PhaseSwap",
    "PulseImplementationMap",
    "default_transmon_gate_implementation_map",
    "default_transmon_calibration",
    "Atom2LevelModel",
    "AtomArrangement",
    "available_model_documents",
    "load_model_document",
]
