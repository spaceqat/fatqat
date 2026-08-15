"""Arrangement-bound target for three-level atom pulse emulation."""

from __future__ import annotations

from dataclasses import dataclass
from math import dist

from ..._pulse_values import ControlChannel, PulseControl
from ...atom_arrangement import AtomArrangement
from ...errors import BackendValidationError
from ...implementation import DeviceOperands
from ...program import Program
from ...resource_layout import ResourceLayout
from ...waveforms import SampledWaveform
from .._core.target import (
    Frame,
    _ControlAddress,
    _ControlBinding,
    _FrameAddress,
    _GateBinding,
    _TargetClaim,
)
from .model import Atom3LevelModel

_FAMILY = "atom.rydberg_3level"


@dataclass(frozen=True, slots=True)
class _Atom3LevelInteraction:
    first: int
    second: int
    distance_um: float
    signed_strength_rad_per_us: float


class _Atom3LevelTarget:
    """Bind one three-level model to a fully occupied arrangement."""

    local_dimension = 3

    def __init__(self, model: Atom3LevelModel, arrangement: AtomArrangement) -> None:
        self.model = model
        self.device_labels = tuple(range(arrangement.cardinality))
        self.hilbert_dimension = 3**arrangement.cardinality
        owner = object()
        self._claims = tuple(
            _TargetClaim(owner, "site", ordinal) for ordinal in self.device_labels
        )
        values = []
        for first in range(arrangement.cardinality):
            for second in range(first + 1, arrangement.cardinality):
                distance = dist(
                    arrangement.coordinates[first], arrangement.coordinates[second]
                )
                values.append(
                    _Atom3LevelInteraction(
                        first,
                        second,
                        distance,
                        model.c6_angular_per_us_um6 / distance**6,
                    )
                )
        self.interactions = tuple(values)

    def bind_control(self, reference: ControlChannel) -> _ControlBinding:
        if (
            not isinstance(reference, _ControlAddress)
            or reference.family != _FAMILY
            or len(reference.operands) != 1
        ):
            raise BackendValidationError("unknown or foreign atom control reference")
        if reference.kind not in ("raman_01", "rydberg_1r"):
            raise BackendValidationError("unknown atom control transition")
        site = self._site_index(reference.operands[0])
        claim = self._claims[site]
        return _ControlBinding(reference.kind, (site,), (claim,))

    def bind_frame(self, reference: Frame) -> _GateBinding:
        if (
            not isinstance(reference, _FrameAddress)
            or reference.family != _FAMILY
            or len(reference.operands) != 1
        ):
            raise BackendValidationError("unknown or foreign atom frame reference")
        site = self._site_index(reference.operands[0])
        return _GateBinding((self._claims[site],), (site,))

    def validate_pulse_controls(
        self,
        controls: tuple[PulseControl, ...],
        bindings: tuple[_ControlBinding, ...],
        block_duration: float,
    ) -> None:
        del block_duration
        if len(controls) != len(bindings):
            raise BackendValidationError("pulse controls and bindings must align")
        if any(not isinstance(child.waveform, SampledWaveform) for child in controls):
            raise BackendValidationError("atom controls require SampledWaveform")

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
                "Atom3LevelEmulator requires exactly one declared quantum resource "
                "per arrangement site"
            )
        if any(ref.register.dim != 2 for ref in refs):
            raise BackendValidationError(
                "Atom3LevelEmulator accepts only dimension-two program quantum "
                "resources"
            )
        layout = resource_layout or ResourceLayout(
            {ref: ordinal for ordinal, ref in enumerate(refs)}
        )
        if layout.refs != frozenset(refs):
            raise BackendValidationError(
                "three-level atom layout must cover exactly the arrangement"
            )
        if layout.device_labels != frozenset(self.device_labels):
            raise BackendValidationError(
                "three-level atom layout must use every arrangement site once"
            )
        return layout

    def reported_digit_map(self, device_label: object) -> tuple[int, ...]:
        self._site_index(device_label)
        return (0, 1, 1)

    def bind_gate_operands(self, device_operands: DeviceOperands) -> _GateBinding:
        site_indices = tuple(self._site_index(value) for value in device_operands)
        return _GateBinding(
            tuple(self._claims[index] for index in site_indices),
            tuple(device_operands),
        )

    def _site_index(self, value: object) -> int:
        if type(value) is not int or not 0 <= value < len(self.device_labels):
            raise BackendValidationError("unknown atom site")
        return value


__all__: list[str] = []
