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

    Args:
        name: Non-empty label shown in diagnostics and representations.

    Raises:
        TypeError: If ``name`` is not a string.
        ValueError: If ``name`` is empty.

    Examples:
        Reuse the same object when multiple gates should share one value:

        >>> import fatqat as fq
        >>> import fatqat.operations as op
        >>> theta = fq.Parameter("theta")
        >>> program = fq.Program(2)
        >>> program.add(op.RX(theta), 0)
        >>> program.add(op.RY(theta), 1)
        >>> bound = program.assign_parameters({theta: 0.25})
        >>> [instruction.operation.theta for instruction in bound.operations]
        [0.25, 0.25]
    """

    name: str

    def __post_init__(self) -> None:
        _validate_parameter_name(self.name)


@dataclass(frozen=True, eq=False, slots=True)
class ParameterVector:
    """Immutable ordered group of distinct :class:`Parameter` objects.

    The vector is a convenient binding key and does not make its elements
    equal by name. Indexing the same vector repeatedly returns the same
    parameter object.

    Args:
        name: Non-empty base label. Element labels use ``name[index]``.
        length: Number of parameter elements. Zero is allowed, although an
            empty vector cannot be used as a binding key.

    Raises:
        TypeError: If ``name`` is not a string or ``length`` is not an integer.
        ValueError: If ``name`` is empty or ``length`` is negative.

    Examples:
        Use the vector to keep the intended element order explicit:

        >>> import fatqat as fq
        >>> angles = fq.ParameterVector("angles", 3)
        >>> [parameter.name for parameter in angles]
        ['angles[0]', 'angles[1]', 'angles[2]']
        >>> angles[0] is angles[0]
        True
    """

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
