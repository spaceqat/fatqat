"""Private structural vocabulary for binding pulse programs to a target."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..._pulse_values import ControlChannel, PulseControl
from ...implementation import DeviceOperands
from ...program import Program
from ...resource_layout import ResourceLayout
from ...errors import BackendValidationError


class ResourceClaim:
    """Opaque target-owned resource consumed only by pulse scheduling."""

    __slots__ = ()


class Frame:
    """Opaque structural key used by the virtual-frame ledger."""

    __slots__ = ()


def _structural_text(value: object, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise BackendValidationError(f"{owner} must be a non-empty string")
    return value


def _structural_operands(values: object, owner: str) -> tuple[str | int, ...]:
    try:
        operands = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise BackendValidationError(f"{owner} must be an iterable") from exc
    if any(
        isinstance(value, bool) or not isinstance(value, (str, int))
        for value in operands
    ):
        raise BackendValidationError(
            f"{owner} must contain only string or integer values"
        )
    return operands


@dataclass(frozen=True, slots=True)
class _ControlAddress(ControlChannel):
    """Portable model-produced address for one physical control coordinate."""

    family: str
    kind: str
    operands: tuple[str | int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", _structural_text(self.family, "family"))
        object.__setattr__(self, "kind", _structural_text(self.kind, "kind"))
        object.__setattr__(
            self,
            "operands",
            _structural_operands(self.operands, "control operands"),
        )


@dataclass(frozen=True, slots=True)
class _FrameAddress(Frame):
    """Portable model-produced address for one virtual frame."""

    family: str
    operands: tuple[str | int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", _structural_text(self.family, "family"))
        operands = _structural_operands(self.operands, "frame operands")
        if not operands:
            raise BackendValidationError("frame operands cannot be empty")
        object.__setattr__(self, "operands", operands)


@dataclass(frozen=True, slots=True)
class _ControlBinding:
    """One target-validated control in physical device vocabulary."""

    kind: str
    device_operands: DeviceOperands
    claims: tuple[ResourceClaim, ...]
    allows_additional_claims: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "device_operands", tuple(self.device_operands))
        object.__setattr__(self, "claims", tuple(self.claims))


@dataclass(frozen=True, slots=True)
class _PreparedControlBinding:
    """One control bound to owning pulse-allocation indices for a prepared run.

    Built-in QuTiP adapters use public model/factor order and pass these indices
    directly to QuTiP.
    """

    kind: str
    engine_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "engine_indices", tuple(self.engine_indices))


@dataclass(frozen=True, slots=True)
class _GateBinding:
    """Claims and physical operands resolved for one gate-side reference."""

    claims: tuple[ResourceClaim, ...]
    device_operands: DeviceOperands

    def __post_init__(self) -> None:
        object.__setattr__(self, "claims", tuple(self.claims))
        object.__setattr__(self, "device_operands", tuple(self.device_operands))


@dataclass(frozen=True, slots=True)
class _TargetClaim(ResourceClaim):
    """Canonical target-local scheduling claim."""

    owner: object = field(repr=False)
    kind: str
    ordinal: int


class _PulseTarget(Protocol):
    """Complete private binding contract shared by pulse backends."""

    model: object
    local_dimension: int
    hilbert_dimension: int
    device_labels: tuple[object, ...]

    def bind_control(self, reference: ControlChannel) -> _ControlBinding: ...

    def bind_frame(self, reference: Frame) -> _GateBinding: ...

    def validate_pulse_controls(
        self,
        controls: tuple[PulseControl, ...],
        bindings: tuple[_ControlBinding, ...],
        block_duration: float,
    ) -> None: ...

    def bind_program(
        self,
        program: Program,
        resource_layout: ResourceLayout | None = None,
    ) -> ResourceLayout: ...

    def reported_digit_map(self, device_label: object) -> tuple[int, ...]: ...

    def bind_gate_operands(self, device_operands: DeviceOperands) -> _GateBinding: ...


__all__ = [
    "Frame",
    "ResourceClaim",
    "_ControlAddress",
    "_ControlBinding",
    "_PreparedControlBinding",
    "_FrameAddress",
    "_GateBinding",
    "_PulseTarget",
    "_TargetClaim",
]
