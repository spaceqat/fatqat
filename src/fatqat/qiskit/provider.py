"""Small provider-style discovery helper for the FATQAT Qiskit backend."""

from __future__ import annotations

from typing import Any

from .backend import FatqatBackend


class FatqatProvider:
    """Create and discover configured :class:`~fatqat.qiskit.FatqatBackend` objects.

    Args:
        **default_backend_kwargs: Defaults passed to each new backend.

    This helper exposes one backend kind and creates a fresh instance for each
    query. It is not a remote service registry.
    """

    def __init__(self, **default_backend_kwargs: Any) -> None:
        self._default_backend_kwargs = dict(default_backend_kwargs)

    def backends(
        self,
        name: str | None = None,
        *,
        filters: Any = None,
        **kwargs: Any,
    ) -> list[FatqatBackend]:
        """Return a newly constructed backend when its name matches.

        Args:
            name: Required backend name, or ``None`` to accept the configured
                name.
            filters: Accepted for provider API compatibility; it has no
                effect.
            **kwargs: Per-call backend constructor values other than ``name``;
                these override the provider defaults.

        Returns:
            A one-item list for a matching name, otherwise an empty list.
        """
        del filters
        backend_kwargs = {**self._default_backend_kwargs, **kwargs}
        backend = FatqatBackend(**backend_kwargs)
        if name is not None and backend.name != name:
            return []
        return [backend]

    def get_backend(self, name: str | None = None, **kwargs: Any) -> FatqatBackend:
        """Return one newly constructed backend.

        Args:
            name: Required backend name, or ``None`` to accept the configured
                name.
            **kwargs: Per-call backend constructor values other than ``name``;
                these override the provider defaults.

        Raises:
            ValueError: If ``name`` does not match the configured backend.
        """
        matches = self.backends(name=name, **kwargs)
        if not matches:
            raise ValueError(f"no backend found for name={name!r}")
        return matches[0]
