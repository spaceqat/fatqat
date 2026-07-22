"""Internal resource-binding infrastructure.

A frontend target expression (a scalar `RegisterRef`, or - for view-capable
operations - a structured `RegisterView`) must be resolved to the concrete
engine index and device label a backend needs to build a plan step and look
up a device-aware implementation rule. `SimulatorBackend` must not accumulate
knowledge of every future frontend resource form, so resolution goes through
an ordered, backend-installed sequence of binders: each receives one target
expression plus the run's `FlatResourceLayout` and either resolves it or
explicitly declines it (returns `None`) so the next binder in the sequence
gets a turn. `ResourceBinding.resolve` raises
`~fatqat.errors.UnsupportedResourceOperandError` if no installed binder
claims the target.

This module holds the binder registry (`ResourceBinding`) plus the one binder
every backend needs: `_scalar_identity_binder`, which resolves a plain
`RegisterRef` to identical engine-index/device-label values and declines
everything else (in particular, every `RegisterView` - no binder in this
phase claims one). It is named with a leading underscore because the binder
registry is internal/protected in this phase (no public registration API),
but it is still meant to be imported and reused as-is by a later backend that
needs to install it *after* a resource-specific binder of its own (e.g. a
grid binder tried first, falling back to this one for plain scalar refs).
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Callable

from ..errors import UnsupportedResourceOperandError
from ..flat_layout import FlatResourceLayout
from ..registers import RegisterRef, RegisterView

QuantumTarget = RegisterRef | RegisterView


@dataclass(frozen=True)
class BoundResource:
    """One frontend target expression resolved to its bound scalar identity.

    Attributes:
        ref: The concrete scalar `RegisterRef` this resource resolved to.
        engine_index: Flat subsystem index used by the simulation engine
            (`ApplyMatrixStep.target_indices`, noise selection, ...).
        device_label: Hashable key used to look up a device-aware
            implementation rule in `ImplementationMap`. Numerically equal to
            `engine_index` for every binder in this task's scope, but a later
            binder (e.g. a grid binder) may bind it to a different device
            site label.
    """

    ref: RegisterRef
    engine_index: int
    device_label: Hashable


# A binder resolves one frontend target expression against the run's flat
# layout, or explicitly declines it by returning `None` so the next binder in
# the `ResourceBinding` sequence gets a turn. No exceptions-as-control-flow
# between binders: a raised exception is a real failure, not a decline.
Binder = Callable[[QuantumTarget, FlatResourceLayout], BoundResource | None]


def _scalar_identity_binder(
    target: QuantumTarget, flat_layout: FlatResourceLayout
) -> BoundResource | None:
    """Resolve a scalar `RegisterRef` to identity engine-index/device-label.

    Declines (returns `None`) for anything that is not a `RegisterRef` - in
    particular, a `RegisterView`, which no binder claims in this task's
    scope.
    """
    if not isinstance(target, RegisterRef):
        return None
    index = flat_layout.subsystem_index(target)
    return BoundResource(ref=target, engine_index=index, device_label=index)


class ResourceBinding:
    """An ordered sequence of binders resolving frontend target expressions.

    `resolve` tries each binder in order and returns the first non-`None`
    result. If every binder declines, it raises
    `~fatqat.errors.UnsupportedResourceOperandError`.
    """

    def __init__(self, binders: Sequence[Binder]) -> None:
        """Store the ordered binder sequence tried by `resolve`.

        Args:
            binders: Binders tried in order; the first to resolve (return
                non-`None`) wins.
        """
        self._binders = tuple(binders)

    def resolve(
        self, target: QuantumTarget, flat_layout: FlatResourceLayout
    ) -> BoundResource:
        """Resolve one frontend target expression to its `BoundResource`.

        Raises:
            UnsupportedResourceOperandError: If no installed binder claims
                `target`.
        """
        for binder in self._binders:
            bound = binder(target, flat_layout)
            if bound is not None:
                return bound
        raise UnsupportedResourceOperandError(
            f"no resource binder resolved target {target!r}"
        )
