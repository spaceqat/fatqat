"""Private quantum and classical index allocations used by execution engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from collections.abc import Mapping

from .program import Program
from .registers import Register, RegisterRef
from .resource_layout import DeviceOperand
from .resource_layout import ResourceLayout


def _describe_state_axes(
    public_operands: tuple[DeviceOperand, ...],
    resource_layout: ResourceLayout,
) -> list[dict[str, object]]:
    """Describe physical state factors in public most-significant-first order.

    Callers pass their public operand order explicitly because matrix-engine
    allocation is private little-endian while QuTiP already uses public tensor
    order. Keeping that choice at each backend boundary avoids a conversion
    layer or an array copy.
    """
    refs_by_operand = {
        resource_layout.device_label(ref): ref for ref in resource_layout.refs
    }
    return [
        {
            "device_operand": device_operand,
            "register_ref": refs_by_operand.get(device_operand),
        }
        for device_operand in public_operands
    ]


@dataclass(frozen=True, slots=True)
class _EngineAllocation:
    """Modeled device operands in backend-local subsystem-index order.

    An engine subsystem index is neither universally public order nor
    necessarily a literal NumPy tensor axis. Each backend supplies the order
    its storage and kernels expect.
    """

    device_operands: tuple[DeviceOperand, ...]
    system_dims: tuple[int, ...]
    _engine_indices: Mapping[DeviceOperand, int] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if len(self.device_operands) != len(self.system_dims):
            raise ValueError(
                "device operands and system dimensions must have equal length"
            )
        if len(set(self.device_operands)) != len(self.device_operands):
            raise ValueError("device operands must be unique")
        if any(type(dim) is not int or dim <= 0 for dim in self.system_dims):
            raise ValueError("system dimensions must be positive integers")
        object.__setattr__(
            self,
            "_engine_indices",
            MappingProxyType(
                {operand: index for index, operand in enumerate(self.device_operands)}
            ),
        )

    @property
    def n_subsystems(self) -> int:
        """Return the number of modeled quantum subsystems."""
        return len(self.system_dims)

    def engine_index(self, device_operand: DeviceOperand) -> int:
        """Return the engine subsystem index for a modeled device operand."""
        try:
            return self._engine_indices[device_operand]
        except KeyError:
            raise KeyError(
                "device operand not part of this engine allocation"
            ) from None


@dataclass(frozen=True, slots=True)
class _ClassicalAllocation:
    """Program-derived classical digit allocation."""

    classical_dims: tuple[int, ...]
    _offsets: Mapping[Register, int] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if any(type(dim) is not int or dim <= 0 for dim in self.classical_dims):
            raise ValueError("classical dimensions must be positive integers")
        object.__setattr__(self, "_offsets", MappingProxyType(dict(self._offsets)))

    @property
    def n_clbits(self) -> int:
        """Return the number of allocated classical digits."""
        return len(self.classical_dims)

    def classical_index(self, ref: RegisterRef) -> int:
        """Return the program-order classical index for ``ref``."""
        try:
            base = self._offsets[ref.register]
        except KeyError:
            raise KeyError("classical ref not part of this allocation") from None
        return base + ref.index

    @classmethod
    def from_program(cls, program: Program) -> "_ClassicalAllocation":
        """Build the independent classical allocation in declaration order."""
        offsets: dict[Register, int] = {}
        dims: list[int] = []
        for register in program.classical_registers:
            offsets[register] = len(dims)
            dims.extend(register.dim for _ in range(register.size))
        return cls(tuple(dims), offsets)
