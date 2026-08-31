"""Arrangement-bound target for two-level atom pulse emulation."""

from __future__ import annotations

from dataclasses import dataclass
from math import dist

import numpy as np

from ..._pulse_values import ControlChannel, PulseControl
from ..atom_arrangement import AtomArrangement
from ...errors import BackendValidationError
from ...implementation import DeviceOperands
from ...program import Program
from ...resource_layout import ResourceLayout
from ..._waveforms import SampledWaveform
from .._core.target import (
    Frame,
    _ControlAddress,
    _ControlBinding,
    _GateBinding,
    _TargetClaim,
)
from .._core.waveform import (
    _complex_spline_magnitude_maximum,
    _real_spline_minimum_and_maximum,
)
from .model import Atom2LevelModel

_FAMILY = "atom.rydberg_2level"
_LOCAL_DIMENSION = 2


@dataclass(frozen=True, slots=True)
class _Atom2LevelInteraction:
    """Store one arrangement-derived Hamiltonian interaction term.

    Ordinals follow arrangement coordinate order. The signed strength retains
    the model's C6 sign and is already divided by the sixth power of distance.
    ``coordinate_scale_um`` preserves the scale used by the inclusive cutoff's
    floating-point boundary check without retaining duplicate coordinates.
    """

    first: int
    second: int
    distance_um: float
    signed_strength_rad_per_us: float
    coordinate_scale_um: float


class _Atom2LevelTarget:
    """Bind one two-level model to immutable site geometry.

    Construction derives the deterministic unordered interaction table once.
    Later lowering uses this target to validate structural control addresses,
    resource binding, and scheduling claims without exposing a public graph.
    """

    local_dimension = _LOCAL_DIMENSION

    def __init__(
        self,
        model: Atom2LevelModel,
        arrangement: AtomArrangement,
    ) -> None:
        self.model = model
        self._limits = model._limits
        self.device_labels = tuple(range(arrangement.num_sites))
        self.hilbert_dimension = 2**arrangement.num_sites
        owner = object()
        self._claims = tuple(
            _TargetClaim(owner, "site", ordinal) for ordinal in self.device_labels
        )
        interactions = []
        coordinates = arrangement.coordinates
        for first in range(arrangement.num_sites):
            for second in range(first + 1, arrangement.num_sites):
                first_coordinate = coordinates[first]
                second_coordinate = coordinates[second]
                distance = dist(first_coordinate, second_coordinate)
                interactions.append(
                    _Atom2LevelInteraction(
                        first,
                        second,
                        distance,
                        model._c6_angular_per_us_um6 / distance**6,
                        max(
                            abs(component)
                            for point in (first_coordinate, second_coordinate)
                            for component in point
                        ),
                    )
                )
        self.interactions = tuple(interactions)

    def bind_control(self, reference: ControlChannel) -> _ControlBinding:
        if (
            not isinstance(reference, _ControlAddress)
            or reference.family != _FAMILY
            or reference.operands
        ):
            raise BackendValidationError(
                "unknown or foreign two-level atom control reference"
            )
        if reference.kind not in ("drive", "detuning"):
            raise BackendValidationError("unknown two-level atom control kind")
        return _ControlBinding(reference.kind, self.device_labels, self._claims)

    def bind_frame(self, reference: Frame) -> _GateBinding:
        del reference
        raise BackendValidationError("two-level atom target has no virtual frames")

    def validate_pulse_controls(
        self,
        controls: tuple[PulseControl, ...],
        bindings: tuple[_ControlBinding, ...],
        block_duration: float,
    ) -> None:
        if len(controls) != len(bindings):
            raise BackendValidationError("pulse controls and bindings must align")
        minimum_duration = self._limits.min_duration
        maximum_duration = self._limits.max_duration
        if minimum_duration is not None and block_duration < minimum_duration:
            raise BackendValidationError(
                "two-level pulse duration is below the minimum"
            )
        if maximum_duration is not None and block_duration > maximum_duration:
            raise BackendValidationError("two-level pulse duration exceeds the maximum")
        for child, binding in zip(controls, bindings):
            if not isinstance(child.waveform, SampledWaveform):
                raise BackendValidationError(
                    "two-level controls require SampledWaveform"
                )
            values = np.asarray(child.waveform.values)
            if binding.kind == "drive":
                maximum = self._limits.max_amplitude
                if (
                    maximum is not None
                    and _complex_spline_magnitude_maximum(child.waveform.times, values)
                    > maximum
                ):
                    raise BackendValidationError(
                        "two-level drive magnitude exceeds its maximum"
                    )
                continue
            if binding.kind != "detuning":
                raise BackendValidationError("unknown two-level atom control kind")
            if np.any(np.imag(values) != 0.0):
                raise BackendValidationError(
                    "two-level detuning coefficients must be real"
                )
            allowed_minimum = self._limits.min_detuning
            allowed_maximum = self._limits.max_detuning
            if allowed_minimum is None and allowed_maximum is None:
                continue
            minimum, maximum = _real_spline_minimum_and_maximum(
                child.waveform.times, np.real(values)
            )
            if allowed_minimum is not None and minimum < allowed_minimum:
                raise BackendValidationError("two-level detuning is below its minimum")
            if allowed_maximum is not None and maximum > allowed_maximum:
                raise BackendValidationError("two-level detuning exceeds its maximum")

    def bind_program(
        self,
        program: Program,
        resource_layout: ResourceLayout | None = None,
    ) -> ResourceLayout:
        refs = tuple(
            register[index]
            for register in program.quantum_registers
            for index in range(register.size)
        )
        if len(refs) != len(self.device_labels):
            raise BackendValidationError(
                "Atom2LevelEmulator requires exactly one declared quantum resource "
                "per arrangement site"
            )
        if any(ref.register.dim != 2 for ref in refs):
            raise BackendValidationError(
                "Atom2LevelEmulator accepts only dimension-two quantum resources"
            )
        layout = resource_layout or ResourceLayout(
            {ref: ordinal for ordinal, ref in enumerate(refs)}
        )
        if layout.refs != frozenset(refs):
            raise BackendValidationError(
                "two-level atom layout must cover exactly the arrangement"
            )
        if layout.device_labels != frozenset(self.device_labels):
            raise BackendValidationError(
                "two-level atom layout must use every arrangement site once"
            )
        return layout

    def reported_digit_map(self, device_label: object) -> tuple[int, ...]:
        self._site_index(device_label)
        return (0, 1)

    def bind_gate_operands(self, device_operands: DeviceOperands) -> _GateBinding:
        site_indices = tuple(self._site_index(value) for value in device_operands)
        return _GateBinding(
            tuple(self._claims[index] for index in site_indices),
            tuple(device_operands),
        )

    def _site_index(self, value: object) -> int:
        if type(value) is not int or not 0 <= value < len(self.device_labels):
            raise BackendValidationError("unknown two-level atom site")
        return value


__all__: list[str] = []
