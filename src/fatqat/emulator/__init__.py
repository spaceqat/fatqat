"""Pulse-level emulation for superconducting and neutral-atom models.

All three public emulators are direct concrete backends. They translate
ordinary gates through a :class:`PulseImplementationMap`, accept direct
``PulseOperation`` controls, and integrate the resulting physical dynamics;
:mod:`fatqat.simulator` is their matrix-backend sibling.

The common workflow lets each gate-capable emulator compile its package
default internally::

    model = TransmonModel.from_document(model_document)
    result = TransmonEmulator(model).run(program).result()

Calibration documents are portable inputs to the standard map builders, not
emulator state. ``PulseDefinition`` and ``PulseImplementationMap`` form the
shared gate-authoring surface. Every emulator accepts replacement gate and
Lindblad maps; a family may choose an empty built-in default. Models create
portable structural control and frame addresses. Public values never expose
QuTiP.

Internally the package has model-neutral orchestration and three model-specific
implementations. One private bound target owns physical topology, control and
frame binding, device labels, and target-local scheduling claims for each
emulator. Shared preparation lowers a program once into immutable bound pulse
facts; numerical adapters consume those facts without rebinding them. The
public backends and realization modules supply family-specific physics.
"""

from __future__ import annotations

from ._core.model_document import (
    CalibrationIdentity,
    FormatIdentity,
    ModelIdentity,
)
from .atom_2level import (
    Atom2LevelEmulator,
    Atom2LevelModel,
)
from .atom_3level import (
    Atom3LevelCalibration,
    Atom3LevelEmulator,
    Atom3LevelModel,
    default_atom_3level_calibration,
    default_atom_3level_gate_implementation_map,
)
from ._core.pulse import (
    PhaseShift,
    PhaseSwap,
    PulseDefinition,
    PulseImplementationMap,
)
from .._pulse_values import ControlChannel, PulseControl
from .superconducting import (
    TransmonEmulator,
    TransmonCalibration,
    TransmonModel,
    default_transmon_calibration,
    default_transmon_gate_implementation_map,
)

__all__ = [
    "TransmonEmulator",
    "Atom3LevelEmulator",
    "Atom2LevelEmulator",
    "TransmonModel",
    "TransmonCalibration",
    "PulseDefinition",
    "ControlChannel",
    "PulseControl",
    "PhaseShift",
    "PhaseSwap",
    "PulseImplementationMap",
    "default_transmon_gate_implementation_map",
    "default_transmon_calibration",
    "Atom3LevelModel",
    "Atom3LevelCalibration",
    "default_atom_3level_calibration",
    "default_atom_3level_gate_implementation_map",
    "Atom2LevelModel",
    "FormatIdentity",
    "ModelIdentity",
    "CalibrationIdentity",
]
