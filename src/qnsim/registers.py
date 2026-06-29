"""Register / RegisterRef value objects (frozen)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Register:
    size: int
    name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.size, int) or isinstance(self.size, bool):
            raise TypeError(f"register size must be int, got {type(self.size)!r}")
        if self.size <= 0:
            raise ValueError(f"register size must be positive, got {self.size}")

    def __getitem__(self, index: int) -> "RegisterRef":
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError(f"register index must be int, got {type(index)!r}")
        if not 0 <= index < self.size:
            raise IndexError(index)
        return RegisterRef(register=self, index=index)


@dataclass(frozen=True)
class QuantumRegister(Register):
    pass


@dataclass(frozen=True)
class ClassicalRegister(Register):
    pass


@dataclass(frozen=True)
class RegisterRef:
    register: Register
    index: int
