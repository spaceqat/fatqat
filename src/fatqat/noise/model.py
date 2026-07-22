"""NoiseModel: the routing container mapping gate occurrences to channels.

A `NoiseModel` holds *which* channel descriptors apply to *which* gate
occurrences - selection facts only, no Kraus arrays and no execution logic.
It is standalone and reusable: independent of any `Program`, usable across
backends, and passed to a backend by reference via its ``noise=`` constructor
parameter. Resolution into concrete Kraus payloads happens at backend
lowering, where descriptors meet the `ChannelImplementationMap` and the
program's resource layout.

Gate-channel target selectors come in two identity spaces, both compared
directly - never through the private engine allocation:

- ``tuple[RegisterRef, ...]`` - logical, frontend refs, how a user pins noise
  to their own program's subsystems. Matched by ref equality against the
  lowered occurrence's targets.
- ``tuple[Hashable, ...]`` - physical, opaque device resource labels, how a
  backend authors default noise for its device before any user program (or
  register) exists. Matched against
  :py:meth:`~fatqat.resource_layout.ResourceLayout.device_operands` for the
  lowered occurrence's targets.

A bare integer selector is a physical device-resource label, never a flat
engine index and never converted into a `RegisterRef`. See
docs/superpowers/specs/2026-07-22-fatqat-resource-layout-and-noise-selector-design.md.

Readout-error selectors are unchanged in this module for now (still resolved
through the private `_EngineAllocation`); that migration is separate scope.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import numpy as np

from .._engine_allocation import _EngineAllocation
from ..implementation.base import _resolve_operation_class
from ..operations import BarrierGate, Operation
from ..registers import QuantumRegister, RegisterRef, RegisterView
from ..resource_layout import ResourceLayout
from .base import Channel

# One entry per add_noise() call: an all-targets fallback (None), a
# logical ref-tuple selector, or a physical device-label-tuple selector
# (homogeneous, validated).
_GateSelector = tuple[RegisterRef, ...] | tuple[Hashable, ...] | None


class NoiseModel:
    """Selection container for channel-representable noise.

    Maps ``(operation class, target selector)`` occurrences to lists of
    `Channel` descriptors. Lookup prefers specific-target entries and falls
    back to the all-targets entry - a specific match fully replaces the
    default for that occurrence, while every other occurrence still gets the
    default (Qiskit Aer's precedence). Repeated ``add_noise`` calls
    accumulate: each attached channel is an independent physical mechanism,
    applied in registration order.

    Attributes:
        qubit_noise: Placeholder for continuously-active per-subsystem noise
            consumed by pulse-family backends; matrix-family backends report
            a non-empty value as unsupported rather than ignoring it.
        metadata: Free-form user annotations, never interpreted here.

    Examples:
        Depolarize every ``X`` gate and run under density-matrix semantics:

        >>> import fatqat as fq
        >>> noise = fq.NoiseModel()
        >>> noise.add_noise(fq.ops.X, fq.noise.Depolarizing(p=0.2))
        >>> program = fq.Program(1)
        >>> program.add(fq.ops.X, 0)
        >>> result = fq.backends.SimulatorBackend(method="DM", noise=noise).run(
        ...     program,
        ...     result_config={"counts": False, "density_matrix": True},
        ... ).result()
        >>> result.get_density_matrix()
        array([[0.1+0.j, 0. +0.j],
               [0. +0.j, 0.9+0.j]])
    """

    def __init__(self) -> None:
        self._gate_channels: dict[
            type[Operation], list[tuple[_GateSelector, list[Channel]]]
        ] = {}
        self._readout_errors: list[tuple[int | RegisterRef | None, np.ndarray]] = []
        self.qubit_noise: dict[Any, Any] = {}
        self.metadata: dict[str, Any] = {}

    def add_noise(
        self,
        operation: Operation | type[Operation],
        channel: Channel,
        *,
        targets: tuple[Hashable, ...] | None = None,
    ) -> None:
        """Attach a channel to every occurrence of an operation, or one target.

        Args:
            operation: An :py:class:`~fatqat.operations.Operation` instance
                (e.g. ``fq.ops.X``) or subclass, normalized to the class for
                keying. `Barrier` is rejected: it is a compiler marker with
                no execution extent for noise to attach to.
            channel: The `Channel` descriptor to apply after each occurrence.
            targets: ``None`` (default) applies to every occurrence. A tuple
                of quantum :py:class:`~fatqat.registers.RegisterRef` pins the
                channel to one logical program-target tuple; a tuple of
                opaque device resource labels (e.g. ``int``, ``str``) pins it
                to one physical occurrence in the backend's device address
                space. The two forms cannot be mixed in one selector, and a
                :py:class:`~fatqat.registers.RegisterView` is never accepted
                (scalar refs only).

        Selection semantics, precisely:

        - Entries matching the same occurrence accumulate, in registration
          order - each is an independent mechanism, so attaching a channel
          twice applies it twice.
        - A specific-target entry replaces the all-targets default on the
          occurrences it matches, and only those (Qiskit Aer's precedence).
          It can therefore *lower* the noise on its target by evicting a
          stronger default; restate the default at the specific level to
          keep it.
        - A logical selector is compared to the lowered occurrence's target
          refs by equality; a physical selector is compared to the lowered
          occurrence's device resource labels
          (:py:meth:`~fatqat.resource_layout.ResourceLayout.device_operands`)
          by equality. See :py:meth:`channels_for`.

        Raises:
            TypeError: If ``operation`` is not an operation, ``channel`` is
                not a `Channel`, or ``targets`` mixes or mistypes selector
                elements (including a `RegisterView`).
            ValueError: If ``operation`` is `Barrier`, ``targets`` is empty,
                or its length does not match a fixed-arity operation.
        """
        op_cls = _resolve_operation_class(operation)
        if op_cls is BarrierGate:
            raise ValueError(
                "Barrier is a compiler marker with no execution semantics; "
                "channel noise cannot attach to it"
            )
        if not isinstance(channel, Channel):
            raise TypeError(f"expected a Channel descriptor, got {channel!r}")
        selector = _normalize_selector(op_cls, targets)
        self._gate_channels.setdefault(op_cls, []).append((selector, [channel]))

    def channels_for(
        self,
        operation: Operation | type[Operation],
        targets: tuple[RegisterRef, ...],
        resource_layout: ResourceLayout,
    ) -> list[Channel]:
        """Return the channels selected for one lowered operation occurrence.

        A logical selector matches when it equals ``targets``; a physical
        selector matches when it equals
        ``resource_layout.device_operands(targets)``. Both kinds of specific
        match accumulate (in registration order); the all-targets (``None``)
        entries apply only when no specific selector matched.

        Args:
            operation: The occurrence's operation (instance or class).
            targets: The occurrence's logical target refs, as lowered from
                the program (never engine indices).
            resource_layout: The run's public resource layout, used to
                resolve physical selectors.
        """
        entries = self._gate_channels.get(_resolve_operation_class(operation))
        if not entries:
            return []
        targets = tuple(targets)
        matched: list[Channel] = []
        fallback: list[Channel] = []
        device_operands: tuple[Hashable, ...] | None = None
        for selector, channels in entries:
            if selector is None:
                fallback.extend(channels)
            elif _is_logical_selector(selector):
                if selector == targets:
                    matched.extend(channels)
            else:
                if device_operands is None:
                    device_operands = resource_layout.device_operands(targets)
                if device_operands == selector:
                    matched.extend(channels)
        return matched if matched else fallback

    def add_readout_error(
        self,
        confusion_matrix: np.ndarray,
        *,
        target: int | RegisterRef | None = None,
    ) -> None:
        """Attach a classical readout confusion matrix to measurements.

        Readout error is classical, not a quantum channel: the physical
        collapse is always true, and only the *reported* classical value is
        resampled through the confusion matrix. Feedforward conditions read
        the reported value (real control electronics see only the readout
        result); qubit reuse and state export see the true post-measurement
        state. It therefore never changes execution-strategy classification.

        Args:
            confusion_matrix: ``(d, d)`` column-stochastic matrix with
                ``C[i, j] = P(report i | true j)``. Copied and frozen; its
                dimension is checked against the measured subsystem at
                lowering.
            target: ``None`` (default) applies to every measured subsystem.
                A flat subsystem index (``int``) or a quantum
                :py:class:`~fatqat.registers.RegisterRef` pins it to one
                subsystem; a specific entry replaces the default there, and
                a later specific entry replaces an earlier one.

        Raises:
            TypeError: If ``target`` is not ``None``, an ``int``, or a
                quantum ref.
            ValueError: If the matrix is not square, at least ``2 x 2``, with
                entries in ``[0, 1]`` and columns summing to 1.
        """
        matrix = np.array(confusion_matrix, dtype=float, copy=True)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"confusion matrix must be square, got shape {matrix.shape}"
            )
        if matrix.shape[0] < 2:
            raise ValueError(
                f"confusion matrix side length must be >= 2, got {matrix.shape[0]}"
            )
        if np.any(matrix < 0) or np.any(matrix > 1):
            raise ValueError("confusion matrix entries must be in [0, 1]")
        if not np.allclose(matrix.sum(axis=0), 1.0):
            raise ValueError(
                "confusion matrix must be column-stochastic: each column "
                "C[:, j] = P(report | true j) must sum to 1"
            )
        if target is not None and type(target) is not int:
            if not isinstance(target, RegisterRef):
                raise TypeError(
                    f"target must be None, a flat index, or a RegisterRef, "
                    f"got {target!r}"
                )
            if not isinstance(target.register, QuantumRegister):
                raise TypeError(
                    "readout-error target refs must point into a "
                    f"QuantumRegister, got a ref into "
                    f"{type(target.register).__name__}"
                )
        if type(target) is int and target < 0:
            raise ValueError(f"flat subsystem index must be >= 0, got {target}")
        matrix.flags.writeable = False
        self._readout_errors.append((target, matrix))

    def readout_error_for(
        self, measured_index: int, layout: _EngineAllocation
    ) -> np.ndarray | None:
        """Return the confusion matrix selected for one measured subsystem.

        A specific entry (flat index or ref, resolved through ``layout``)
        replaces the all-target default; among specific entries for the same
        subsystem, the last registered wins. Ref entries whose register is
        not in the laid-out program can never match and are skipped.
        """
        specific = fallback = None
        for target, matrix in self._readout_errors:
            if target is None:
                fallback = matrix
            elif _selector_indices((target,), layout) == (measured_index,):
                specific = matrix
        return specific if specific is not None else fallback

    def has_readout_error(self) -> bool:
        """Return whether any readout-error entry is registered."""
        return bool(self._readout_errors)

    def has_noise_for(self, operation: Operation | type[Operation]) -> bool:
        """Return whether any entry is keyed on this operation family."""
        return _resolve_operation_class(operation) in self._gate_channels

    def channel_types(self) -> frozenset[type[Channel]]:
        """Return every descriptor type attached anywhere in this model."""
        return frozenset(
            type(channel)
            for entries in self._gate_channels.values()
            for _, channels in entries
            for channel in channels
        )


def _is_logical_selector(selector: tuple[Hashable, ...]) -> bool:
    """Return whether a validated, homogeneous gate selector is logical."""
    return isinstance(selector[0], RegisterRef)


def _normalize_selector(
    op_cls: type[Operation],
    targets: tuple[Hashable, ...] | None,
) -> _GateSelector:
    """Validate and normalize an ``add_noise`` target selector.

    A selector is ``None``, a tuple wholly of `RegisterRef` (logical), or a
    tuple wholly of some other hashable (physical device resource labels).
    Mixing the two forms, or including a `RegisterView`, is rejected; a
    physical label is opaque and is not itself validated (any hashable value
    is a legal device resource label until run-time validation against an
    actual `ResourceLayout`).
    """
    if targets is None:
        return None
    selector = tuple(targets)
    if len(selector) == 0:
        raise ValueError("targets must be None or a non-empty tuple")
    for t in selector:
        if isinstance(t, RegisterView):
            raise TypeError(
                "noise targets must be scalar RegisterRef or device "
                f"resource labels, not a RegisterView; got {t!r}"
            )
    is_ref = [isinstance(t, RegisterRef) for t in selector]
    if all(is_ref):
        for ref in selector:
            if not isinstance(ref.register, QuantumRegister):
                raise TypeError(
                    "noise target refs must point into a QuantumRegister, "
                    f"got a ref into {type(ref.register).__name__}"
                )
    elif any(is_ref):
        raise TypeError(
            "targets must be all RegisterRef (logical) or all device "
            f"resource labels (physical), not mixed; got {selector!r}"
        )
    expected = op_cls._num_subsystems
    if expected is not None and len(selector) != expected:
        raise ValueError(
            f"{op_cls.__name__} targets {expected} subsystem(s), "
            f"got a selector of length {len(selector)}"
        )
    return selector


def _selector_indices(
    selector: tuple[int, ...] | tuple[RegisterRef, ...],
    layout: _EngineAllocation,
) -> tuple[int, ...] | None:
    """Resolve a specific selector to flat indices, or ``None`` if unmatchable."""
    if type(selector[0]) is int:  # homogeneous, validated at add_noise
        return selector
    try:
        return tuple(layout.subsystem_index(ref) for ref in selector)
    except KeyError:
        return None  # register not in this program's layout
