"""Identity-based symbolic parameters for program construction."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field


def _validate_parameter_name(name: object) -> str:
    """Return a valid public parameter label."""
    if not isinstance(name, str):
        raise TypeError("parameter name must be a string")
    if not name:
        raise ValueError("parameter name must not be empty")
    return name


@dataclass(frozen=True, eq=False, slots=True)
class Parameter:
    """Immutable, identity-based scalar program parameter.

    ``name`` is used only for display and diagnostics. Two parameters with the
    same name remain distinct binding keys.
    """

    name: str

    def __post_init__(self) -> None:
        _validate_parameter_name(self.name)


@dataclass(frozen=True, eq=False, slots=True)
class ParameterVector:
    """Immutable ordered group of distinct :class:`Parameter` objects."""

    name: str
    length: int
    _parameters: tuple[Parameter, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_parameter_name(self.name)
        if type(self.length) is not int:
            raise TypeError("parameter vector length must be an integer")
        if self.length < 0:
            raise ValueError("parameter vector length must be non-negative")
        object.__setattr__(
            self,
            "_parameters",
            tuple(Parameter(f"{self.name}[{index}]") for index in range(self.length)),
        )

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[Parameter]:
        return iter(self._parameters)

    def __getitem__(self, index: int) -> Parameter:
        if type(index) is not int:
            raise TypeError("parameter vector indices must be integers")
        return self._parameters[index]
