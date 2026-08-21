"""Private immutable values used by family-owned control namespaces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .target import _ControlAddress


@dataclass(frozen=True, slots=True)
class _ControlSelector:
    """Describe and invoke one structural control-address factory.

    This private value deliberately contains only stable authoring metadata.
    Target-specific legality remains the responsibility of target binding.
    """

    scope: str
    operands: tuple[str, ...]
    coefficient_domain: str
    coefficient_unit: str
    _factory: Callable[..., _ControlAddress]

    @property
    def __wrapped__(self) -> Callable[..., _ControlAddress]:
        """Expose the family factory to standard signature introspection."""
        return self._factory

    def __getattribute__(self, name: str) -> object:
        """Forward instance docstring lookup to the family-specific factory."""
        if name == "__doc__":
            factory = object.__getattribute__(self, "_factory")
            return factory.__doc__
        return object.__getattribute__(self, name)

    def __call__(self, *args: object, **kwargs: object) -> _ControlAddress:
        """Return the structural address produced by the family factory."""
        return self._factory(*args, **kwargs)


__all__: list[str] = []
