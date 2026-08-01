"""The model-facing seam the pulse representation, scheduler, and engine use.

Decision 0023 states the execution engine is general rather than
superconducting-specific. This module is what makes that true: it declares the
*abstract* handle kinds a physics model mints and the small protocol such a
model must satisfy, so :mod:`.pulse`, :mod:`.scheduling`, and :mod:`.engine`
can be written against those instead of importing the concrete transmon types
from :mod:`.superconducting`.

The handle kinds are plain marker base classes rather than runtime-checkable
``Protocol`` types on purpose. A model's handles are opaque value objects with
no methods to check structurally, so a structural protocol could not tell a
control channel from a resource claim; a nominal marker can, costs nothing,
and keeps ``isinstance`` discrimination available to :class:`.PulseDefinition`,
which carries no model and therefore cannot ask one to bind a handle.

Two model questions are deliberately behind the protocol rather than answered
by inspecting handles:

* ``required_claims_for_control`` - which resources a pulse driving a channel
  must reserve. On the transmon model an exchange channel implicates both
  endpoints and their coupling edge; that topology is the model's knowledge,
  not the pulse representation's.
* ``validate_control_coefficients`` - whether a channel accepts a complex
  envelope. On the transmon model drive channels do and detuning/exchange
  channels do not.

Both once lived as ``kind``-string switches inside the pulse layer or the
QuTiP adapter. Keeping them here means a future model with different channel
kinds needs no change in :mod:`.pulse`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


class ResourceClaim:
    """Abstract model resource a pulse block reserves for its whole duration.

    Concrete models mint subclasses (the transmon model has one per subsystem
    and one per coupling edge). The scheduler treats claims as opaque hashable
    values and only ever tests two blocks' claim sets for intersection, so a
    model may partition its hardware however it likes.
    """

    __slots__ = ()


class ControlChannel:
    """Abstract physical control a :class:`.SampledControl` drives."""

    __slots__ = ()


class Frame:
    """Abstract virtual-frame ledger key a post-block frame action updates.

    The engine carries frame angles keyed by these handles without
    interpreting them; only the model's own solver adapter turns accumulated
    angles into a basis transformation.
    """

    __slots__ = ()


@runtime_checkable
class PhysicsModel(Protocol):
    """What the pulse representation, scheduler, and engine need from a model.

    :class:`~fatqat.emulator.superconducting.SCTransmonModel` satisfies this
    structurally; nothing inherits from it. The protocol covers only questions
    the model-neutral layers actually ask - handle binding, the resources a
    control implicates, and whether a control accepts complex coefficients. It
    deliberately excludes everything specific to one model's physics
    (subsystem records, coupling topology, local operators), which the model's
    own realization rules and solver adapter read directly.

    Kept to exactly what those layers call. A model naturally exposes more -
    the transmon model also has ``physical_dimension``, ``time_unit``,
    ``subsystems``, and local operators - but every one of those is read
    through a concrete model reference by the backend, the lowering step, or
    the solver adapter, never through this protocol. Listing them here would
    burden a future model with members no model-neutral code consumes. Add a
    member when a neutral layer actually needs it.

    An implementation must accept these abstract handle types, not narrow them
    to its own concrete refs: narrowing a parameter breaks structural
    compatibility, so the model would silently stop being assignable here.
    Narrow internally, after the corresponding ``bind_*`` has validated the
    handle.

    The ``...`` bodies below are the protocol-stub convention pyright expects;
    without them it reads each method as falling off the end and returning
    ``None``. pylint's ``unnecessary-ellipsis`` disagrees, so it is disabled
    for this class only.
    """

    # pylint: disable=unnecessary-ellipsis

    def bind_claim(self, reference: ResourceClaim) -> int:
        """Validate any resource-claim handle and return its model ordinal.

        One entry point for every claim kind on purpose: a block validating
        its declared claims only needs to know they are genuinely this
        model's, and dispatching between a model's own claim kinds is that
        model's knowledge. Concrete models may still expose narrower binders
        for their own realization rules and adapters.
        """
        ...

    def bind_control(self, reference: ControlChannel) -> int:
        """Validate a control handle and return its model ordinal."""
        ...

    def bind_frame(self, reference: Frame) -> int:
        """Validate a frame handle and return its model ordinal."""
        ...

    def required_claims_for_control(
        self, channel: ControlChannel
    ) -> frozenset[ResourceClaim]:
        """Return every resource a pulse driving ``channel`` must claim.

        A block whose ``resource_claims`` do not cover this set for one of its
        driven controls is rejected at construction.
        """
        ...

    def validate_control_coefficients(
        self, channel: ControlChannel, coefficients: np.ndarray
    ) -> None:
        """Reject an envelope this channel cannot physically realize.

        Called once per control when a :class:`.PulseBlock` binds to the
        model, so both execution paths reject an invalid envelope at lowering
        with one message instead of discovering it during solver binding.
        """
        ...
