"""Private translation from FATQAT subsystem axes to QuTiP tensor factors.

FATQAT numbers modeled physical subsystems in canonical little-endian order:
axis 0 is the least-significant digit of a returned flat basis index.  QuTiP
orders tensor factors in the opposite direction, with factor 0 contributing
the most-significant digit.  This view is the sole translation between those
two namespaces; lowering and :class:`~fatqat._index_allocation._EngineAllocation`
remain canonical and independent of the solver library.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

from ..._index_allocation import _EngineAllocation

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class _QutipEngineAllocation:
    """A QuTiP factor-order view of one canonical engine allocation."""

    canonical: _EngineAllocation

    @property
    def qutip_dims(self) -> tuple[int, ...]:
        """Return local dimensions in QuTiP's most-significant-first order."""
        return tuple(reversed(self.canonical.system_dims))

    def factor_index(self, canonical_axis: int) -> int:
        """Translate one canonical little-endian axis to a QuTiP factor."""
        return self.canonical.n_subsystems - 1 - canonical_axis

    def factor_indices(self, canonical_axes: Sequence[int]) -> tuple[int, ...]:
        """Translate axes independently while preserving operand order."""
        return tuple(self.factor_index(axis) for axis in canonical_axes)

    def factor_order(self, values: Sequence[_T]) -> tuple[_T, ...]:
        """Return one value per canonical axis in QuTiP factor order."""
        return tuple(reversed(values))


__all__: list[str] = []
