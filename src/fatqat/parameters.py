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
    """Create an immutable, identity-based scalar program parameter.

    ``name`` is a display and diagnostic label, not the parameter's identity.
    Equality and hashing use object identity, so two parameters with the same
    name remain distinct. Reuse one object when several operation arguments
    should share a value, then bind that object with
    `Program.assign_parameters` or a sweep API. Parameters nested inside
    another container are not discovered for binding.

    Args:
        name: Non-empty display label. Bindings use the parameter object, not
            this name.

    Attributes:
        name: Immutable display label supplied at construction.

    Raises:
        TypeError: If ``name`` is not a string.
        ValueError: If ``name`` is empty.

    Examples:
        Reuse the same object when multiple gates should share one value:

        >>> import fatqat as fq
        >>> import fatqat.operations as ops
        >>> theta = fq.Parameter("theta")
        >>> program = fq.Program(2)
        >>> program.add(ops.RX(theta), 0)
        >>> program.add(ops.RY(theta), 1)
        >>> bound = program.assign_parameters({theta: 0.25})
        >>> bound is program
        False
    """

    name: str

    def __post_init__(self) -> None:
        _validate_parameter_name(self.name)


@dataclass(frozen=True, eq=False, slots=True)
class ParameterVector:
    """Create an immutable ordered group of distinct `Parameter` objects.

    The vector and its elements use identity-based equality. Repeated indexing
    returns the same element object. Integer indexing supports normal negative
    indices, and iteration follows increasing index order; slices and other
    index types are not accepted.

    Bind a non-empty vector with `Program.assign_parameters` when every element
    is used directly as an operation argument. The value must be a
    matching-length, one-dimensional NumPy array or another non-string,
    non-bytes, non-mapping iterable. Values are consumed once and paired with
    elements in index order. Individual elements are ordinary `Parameter` keys
    and support partial binding. A zero-length vector is valid as an empty
    container but not as a binding key. Parameters nested inside another
    container are not discovered for binding.

    Args:
        name: Non-empty base label. Element labels use ``name[index]``.
        length: Exact non-negative integer number of elements. Booleans are not
            accepted. Zero creates an empty vector.

    Attributes:
        name: Immutable base display label.
        length: Immutable number of parameter elements.

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
        """Return the number of parameter elements."""
        return self.length

    def __iter__(self) -> Iterator[Parameter]:
        """Iterate over parameter elements in index order."""
        return iter(self._parameters)

    def __getitem__(self, index: int) -> Parameter:
        """Return the parameter at an integer index.

        Negative indices follow normal Python sequence rules.

        Args:
            index: Exact integer element index. Booleans and slices are not
                accepted.

        Returns:
            The stable `Parameter` object stored at that index.

        Raises:
            TypeError: If ``index`` is not an exact integer.
            IndexError: If ``index`` is outside the vector.
        """
        if type(index) is not int:
            raise TypeError("parameter vector indices must be integers")
        return self._parameters[index]
