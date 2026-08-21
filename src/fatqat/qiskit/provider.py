"""Optional provider convenience for discovering fatqat Qiskit backends."""

from __future__ import annotations

from typing import Any

from .backend import FatqatBackend


class FatqatProvider:
    """Small discovery helper for named fatqat simulator backends."""

    def __init__(self, **default_backend_kwargs: Any) -> None:
        self._default_backend_kwargs = dict(default_backend_kwargs)

    def backends(
        self,
        name: str | None = None,
        *,
        filters: Any = None,
        **kwargs: Any,
    ) -> list[FatqatBackend]:
        del filters
        backend_kwargs = {**self._default_backend_kwargs, **kwargs}
        backend = FatqatBackend(**backend_kwargs)
        if name is not None and backend.name != name:
            return []
        return [backend]

    def get_backend(self, name: str | None = None, **kwargs: Any) -> FatqatBackend:
        matches = self.backends(name=name, **kwargs)
        if not matches:
            raise ValueError(f"no backend found for name={name!r}")
        return matches[0]
