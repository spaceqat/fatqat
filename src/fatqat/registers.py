"""Register / RegisterRef value objects (frozen)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Register:
    """Base value object for a fixed-size resource register.

    Register metadata is copied at construction time, and register objects are
    frozen. Use indexing to create ``RegisterRef`` values.

    Attributes:
        size: Number of slots in the register. Must be a positive integer.
        name: Optional user-facing register name.
        metadata: User metadata copied into the register.
        dim: Dimension of each slot (default 2 for qubits). Must be an integer >= 2.

    Examples:
        Index into a register to get a ``RegisterRef``:

        >>> import fatqat as fq
        >>> qreg = fq.QuantumRegister(2, name="q")
        >>> qreg[0]
        RegisterRef(register=QuantumRegister(size=2, name='q', metadata={}, dim=2), index=0)
        >>> qreg[5]
        Traceback (most recent call last):
            ...
        IndexError: 5

        A qutrit register uses ``dim=3``:

        >>> qutrit = fq.QuantumRegister(1, dim=3)
        >>> qutrit.dim
        3
    """

    size: int
    name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    dim: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.size, int) or isinstance(self.size, bool):
            raise TypeError(f"register size must be int, got {type(self.size)!r}")
        if self.size <= 0:
            raise ValueError(f"register size must be positive, got {self.size}")
        if not isinstance(self.dim, int) or isinstance(self.dim, bool):
            raise TypeError(f"register dim must be int, got {type(self.dim)!r}")
        if self.dim < 2:
            raise ValueError(
                f"register dim must be >= 2 (a nontrivial system), got {self.dim}"
            )
        # Copy metadata so a caller's later mutation can't reach into this frozen
        # value object.
        object.__setattr__(self, "metadata", dict(self.metadata))

    def __getitem__(self, index: int) -> "RegisterRef":
        """Return a reference to one slot in this register.

        Args:
            index: Zero-based slot index. Negative indexing is not supported.

        Returns:
            A ``RegisterRef`` pointing at this register and index.

        Raises:
            TypeError: If ``index`` is not an integer.
            IndexError: If ``index`` is outside ``0 <= index < size``.
        """
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError(f"register index must be int, got {type(index)!r}")
        if not 0 <= index < self.size:
            raise IndexError(index)
        return RegisterRef(register=self, index=index)


@dataclass(frozen=True)
class QuantumRegister(Register):
    """Register whose refs may be used as quantum operation targets."""


@dataclass(frozen=True)
class ClassicalRegister(Register):
    """Register whose refs may receive measurement results and conditions."""


@dataclass(frozen=True)
class RegisterRef:
    """Reference to one slot in a register.

    Attributes:
        register: Register object being referenced.
        index: Zero-based slot index within ``register``.
    """

    register: Register
    index: int
