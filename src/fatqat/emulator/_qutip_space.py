"""Construct full-system QuTiP values from canonical FATQAT axes.

FATQAT numbers modeled physical subsystems in canonical little-endian order:
axis 0 is the least-significant digit of a returned flat basis index. QuTiP
orders tensor factors in the opposite direction, with factor 0 contributing
the most-significant digit. This private tensor space is the sole translation
between those two namespaces; lowering and
:class:`~fatqat._index_allocation._EngineAllocation` remain solver-neutral.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from qutip import Qobj, qeye, tensor

from .._index_allocation import _EngineAllocation


@dataclass(frozen=True, slots=True)
class _QutipTensorSpace:
    """QuTiP construction boundary for one canonical engine allocation."""

    canonical: _EngineAllocation

    @property
    def dims(self) -> tuple[int, ...]:
        """Return local dimensions in QuTiP's most-significant-first order."""
        return tuple(reversed(self.canonical.system_dims))

    def target(self, canonical_axis: int) -> int:
        """Translate one canonical little-endian axis to a QuTiP factor."""
        return self.canonical.n_subsystems - 1 - canonical_axis

    def targets(self, canonical_axes: Sequence[int]) -> tuple[int, ...]:
        """Translate axes independently while preserving operand order."""
        return tuple(self.target(axis) for axis in canonical_axes)

    def expand_local(self, canonical_axis: int, operator: Qobj) -> Qobj:
        """Expand one local operator on a canonical physical axis."""
        canonical_factors = [qeye(dim) for dim in self.canonical.system_dims]
        canonical_factors[canonical_axis] = operator
        return self.full_tensor(canonical_factors)

    def full_tensor(self, canonical_factors: Sequence[Qobj]) -> Qobj:
        """Tensor one factor per canonical axis into QuTiP factor order."""
        factors = tuple(canonical_factors)
        if len(factors) != self.canonical.n_subsystems:
            raise ValueError("full tensor must cover every modeled subsystem")
        return tensor(*reversed(factors))


__all__: list[str] = []
