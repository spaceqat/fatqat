"""Model-bound target for superconducting transmon pulse emulation."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np

from ..._pulse_values import ControlChannel, PulseControl
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
from .model import TransmonModel

_FAMILY = "sc.transmon"


class _TransmonTarget:
    """Bind one immutable transmon model to claims and topology lookups."""

    local_dimension = 3

    def __init__(self, model: TransmonModel) -> None:
        self.model = model
        self.device_labels = model.subsystem_ids
        self.hilbert_dimension = 3 ** len(model.subsystems)
        self._subsystem_ordinals = MappingProxyType(
            {label: ordinal for ordinal, label in enumerate(self.device_labels)}
        )
        self._coupling_ordinals = MappingProxyType(
            {
                frozenset(coupling.subsystem_ids): ordinal
                for ordinal, coupling in enumerate(model.couplings)
            }
        )
        owner = object()
        self._subsystem_claims = tuple(
            _TargetClaim(owner, "subsystem", ordinal)
            for ordinal in range(len(model.subsystems))
        )
        self._coupling_claims = tuple(
            _TargetClaim(owner, "coupling", ordinal)
            for ordinal in range(len(model.couplings))
        )

    def bind_control(self, reference: ControlChannel) -> _ControlBinding:
        if not isinstance(reference, _ControlAddress) or reference.family != _FAMILY:
            raise BackendValidationError("unknown or foreign control reference")
        if reference.kind in ("drive", "detuning") and len(reference.operands) == 1:
            ordinal = self._subsystem_ordinal(reference.operands[0])
            return _ControlBinding(
                reference.kind,
                (self.device_labels[ordinal],),
                (self._subsystem_claims[ordinal],),
            )
        if reference.kind == "exchange" and len(reference.operands) == 2:
            labels = tuple(reference.operands)
            if (
                any(not isinstance(label, str) for label in labels)
                or tuple(sorted(labels)) != labels
            ):
                raise BackendValidationError(
                    "exchange control endpoints must be canonical"
                )
            coupling_ordinal = self._coupling_ordinal(labels[0], labels[1])
            subsystem_ordinals = tuple(
                self._subsystem_ordinal(label) for label in labels
            )
            claims = tuple(
                self._subsystem_claims[ordinal] for ordinal in subsystem_ordinals
            ) + (self._coupling_claims[coupling_ordinal],)
            return _ControlBinding(
                "exchange", labels, claims, allows_additional_claims=True
            )
        raise BackendValidationError("unknown or foreign control reference")

    def bind_frame(self, reference: Frame) -> _GateBinding:
        if (
            not isinstance(reference, _FrameAddress)
            or reference.family != _FAMILY
            or len(reference.operands) != 1
        ):
            raise BackendValidationError("unknown or foreign frame reference")
        ordinal = self._subsystem_ordinal(reference.operands[0])
        return _GateBinding(
            (self._subsystem_claims[ordinal],),
            (self.device_labels[ordinal],),
        )

    def validate_pulse_controls(
        self,
        controls: tuple[PulseControl, ...],
        bindings: tuple[_ControlBinding, ...],
        block_duration: float,
    ) -> None:
        del block_duration
        if len(controls) != len(bindings):
            raise BackendValidationError("pulse controls and bindings must align")
        for child, binding in zip(controls, bindings):
            if not isinstance(child.waveform, SampledWaveform):
                raise BackendValidationError(
                    "superconducting controls require SampledWaveform"
                )
            if binding.kind != "drive" and not np.allclose(
                np.asarray(child.waveform.values).imag,
                0.0,
                atol=1e-12,
                rtol=0.0,
            ):
                raise BackendValidationError(
                    f"{binding.kind} pulse coefficients must be real"
                )

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
        if len(refs) > len(self.device_labels):
            raise BackendValidationError(
                f"program requires {len(refs)} subsystems but model has "
                f"{len(self.device_labels)}"
            )
        if any(ref.register.dim != 2 for ref in refs):
            raise BackendValidationError(
                "TransmonEmulator embeds only dimension-two program subsystems "
                "into qutrits"
            )
        layout = resource_layout or ResourceLayout(
            {ref: self.device_labels[ordinal] for ordinal, ref in enumerate(refs)}
        )
        if layout.refs != frozenset(refs):
            raise BackendValidationError(
                "transmon resource layout must cover exactly this program's refs"
            )
        if not layout.device_labels <= frozenset(self.device_labels):
            raise BackendValidationError("resource layout names an unknown transmon")
        if len(layout.device_labels) != len(layout.refs):
            raise BackendValidationError(
                "resource layout maps multiple refs to one transmon"
            )
        return layout

    def reported_digit_map(self, device_label: object) -> tuple[int, ...]:
        self._subsystem_ordinal(device_label)
        return (0, 1, 1)

    def bind_gate_operands(self, device_operands: DeviceOperands) -> _GateBinding:
        ordinals = tuple(self._subsystem_ordinal(label) for label in device_operands)
        return _GateBinding(
            tuple(self._subsystem_claims[ordinal] for ordinal in ordinals),
            tuple(device_operands),
        )

    def _subsystem_ordinal(self, label: object) -> int:
        try:
            return self._subsystem_ordinals[label]
        except (KeyError, TypeError):
            raise BackendValidationError(f"unknown model subsystem {label!r}") from None

    def _coupling_ordinal(self, first: str, second: str) -> int:
        try:
            return self._coupling_ordinals[frozenset((first, second))]
        except KeyError:
            raise BackendValidationError(
                f"model has no declared coupling edge {first!r}-{second!r}"
            ) from None


__all__: list[str] = []
