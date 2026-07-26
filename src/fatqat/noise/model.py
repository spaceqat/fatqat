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

``slots`` is a third, orthogonal concept: zero-based positional indices in a
matched occurrence's target tuple. It is neither a logical ref, a device
label, nor an engine index; it determines the channel's execution extent.
- ``tuple[DeviceOperand, ...]`` - physical, opaque device resource labels, how a
  backend authors default noise for its device before any user program (or
  register) exists. Matched against
  :py:meth:`~fatqat.resource_layout.ResourceLayout.device_operands` for the
  lowered occurrence's targets.

A bare integer selector is a physical device-resource label, never a flat
engine index and never converted into a `RegisterRef`. See
docs/superpowers/specs/2026-07-22-fatqat-resource-layout-and-noise-selector-design.md.

Readout-error selectors share the same two identity spaces, but the stored
selector is scalar - ``None``, one `RegisterRef`, or one physical device
resource label - never a tuple. A logical selector is matched by ref equality
against the measured target; a physical selector is matched against
:py:meth:`~fatqat.resource_layout.ResourceLayout.device_label` for that
target. See `readout_error_for`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import BackendValidationError
from ..implementation.base import _resolve_operation_class
from ..operations import BarrierGate, Operation
from ..program import Program
from ..registers import QuantumRegister, RegisterRef, RegisterView
from ..resource_layout import DeviceOperand, ResourceLayout
from .base import Channel
from .continuous import ContinuousNoise

# One entry per add_noise() call: an all-targets fallback (None), a
# logical ref-tuple selector, or a physical device-label-tuple selector
# (homogeneous, validated).
_GateSelector = tuple[RegisterRef, ...] | tuple[DeviceOperand, ...] | None

# What add_noise() itself accepts as `targets`: everything _GateSelector
# does, plus a bare RegisterRef/device label as shorthand for a one-element
# tuple (only valid for a fixed arity-1 operation - _normalize_selector
# wraps it and the existing length check rejects it otherwise).
_GateTargetsArg = _GateSelector | RegisterRef | DeviceOperand

# One entry per add_readout_error() call: an all-subsystems fallback (None),
# a logical RegisterRef selector, or a physical device-label selector
# (scalar, unlike _GateSelector - a readout error names one measured
# subsystem, not an occurrence's whole target tuple).
_ReadoutSelector = RegisterRef | DeviceOperand | None
_ContinuousSelector = RegisterRef | DeviceOperand | None

_Slots = tuple[int, ...] | None


@dataclass(frozen=True)
class _GateNoiseEntry:
    """One registration: occurrence selector, extent, and descriptor."""

    selector: _GateSelector
    slots: _Slots
    channel: Channel


@dataclass(frozen=True)
class _ContinuousNoiseEntry:
    """One continuously active descriptor and its scalar selector."""

    selector: _ContinuousSelector
    source: ContinuousNoise


class NoiseModel:
    """Selection container for channel-representable noise.

    Maps ``(operation class, target selector)`` occurrences to lists of
    `Channel` descriptors. Lookup prefers specific-target entries and falls
    back to the all-targets entry - a specific match replaces defaults of the
    same extent for that occurrence, while every other occurrence still gets
    the default (Qiskit Aer's precedence). Repeated ``add_noise`` calls
    accumulate: each attached channel is an independent physical mechanism,
    applied in registration order.

    Attributes:
        qubit_noise: Empty-only legacy placeholder. A non-empty value is
            rejected; use :meth:`add_continuous_noise` instead.
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
        ...     result_config={"counts": False, "final_state": True},
        ... ).result()
        >>> result.get_density_matrix()
        array([[0.1+0.j, 0. +0.j],
               [0. +0.j, 0.9+0.j]])
    """

    def __init__(self) -> None:
        self._gate_channels: dict[type[Operation], list[_GateNoiseEntry]] = {}
        self._readout_errors: list[tuple[_ReadoutSelector, np.ndarray]] = []
        self._continuous_noise: list[_ContinuousNoiseEntry] = []
        self.qubit_noise: dict[Any, Any] = {}
        self.metadata: dict[str, Any] = {}

    def add_noise(
        self,
        operation: Operation | type[Operation],
        channel: Channel,
        *,
        targets: _GateTargetsArg = None,
        slots: tuple[int, ...] | int | None = None,
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
                (scalar refs only). For a fixed arity-1 operation, a bare
                `RegisterRef` or device label is also accepted as shorthand
                for a one-element tuple.
            slots: ``None`` (default) applies the channel jointly to the
                whole occurrence. An int or strictly increasing tuple of
                zero-based occurrence positions scopes the channel to those
                subsystems instead.

        Selection semantics, precisely:

        - Entries matching the same occurrence accumulate, in registration
          order - each is an independent mechanism, so attaching a channel
          twice applies it twice.
        - A specific-target entry replaces the all-targets default of the
          same extent on the occurrences it matches, and only those.
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
        normalized_slots = _normalize_slots(op_cls, channel, selector, slots)
        self._gate_channels.setdefault(op_cls, []).append(
            _GateNoiseEntry(selector, normalized_slots, channel)
        )

    def channels_for(
        self,
        operation: Operation | type[Operation],
        targets: tuple[RegisterRef, ...],
        resource_layout: ResourceLayout,
    ) -> list[tuple[Channel, tuple[RegisterRef, ...]]]:
        """Return the channels selected for one lowered operation occurrence.

        A logical selector matches when it equals ``targets``; a physical
        selector matches when it equals
        ``resource_layout.device_operands(targets)``. Both kinds of specific
        match accumulate (in registration order); all-targets (``None``)
        entries apply only when no specific selector of their same extent
        matched. Returns ``(channel, extent)`` pairs.

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
        classified: list[bool | None] = []
        has_specific: dict[_Slots, bool] = {}
        device_operands: tuple[DeviceOperand, ...] | None = None
        for entry in entries:
            selector = entry.selector
            if selector is None:
                classified.append(False)
            elif _is_ref_selector(selector):
                if selector == targets:
                    classified.append(True)
                    has_specific[entry.slots] = True
                else:
                    classified.append(None)
            else:
                if device_operands is None:
                    device_operands = resource_layout.device_operands(targets)
                if device_operands == selector:
                    classified.append(True)
                    has_specific[entry.slots] = True
                else:
                    classified.append(None)

        resolved: list[tuple[Channel, tuple[RegisterRef, ...]]] = []
        for entry, status in zip(entries, classified):
            if status is None or (status is False and has_specific.get(entry.slots)):
                continue
            if entry.slots is not None and max(entry.slots) >= len(targets):
                raise BackendValidationError(
                    f"slot {max(entry.slots)} is out of range for {type(entry.channel).__name__} "
                    f"on this {len(targets)}-subsystem occurrence"
                )
            extent = (
                targets
                if entry.slots is None
                else tuple(targets[index] for index in entry.slots)
            )
            expected = entry.channel.num_subsystems
            if expected is not None and len(extent) != expected:
                raise BackendValidationError(
                    f"{type(entry.channel).__name__} acts on {expected} subsystem(s), "
                    f"got an extent of {len(extent)}"
                )
            resolved.append((entry.channel, extent))
        return resolved

    def add_readout_error(
        self,
        confusion_matrix: np.ndarray,
        *,
        target: _ReadoutSelector = None,
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
            target: ``None`` (default) applies to every measured subsystem. A
                quantum :py:class:`~fatqat.registers.RegisterRef` pins it to
                one logical subsystem (matched by ref equality); an opaque
                device resource label (e.g. ``int``, ``str``) pins it to one
                physical measured subsystem in the backend's device address
                space (matched via
                :py:meth:`~fatqat.resource_layout.ResourceLayout.device_label`).
                A :py:class:`~fatqat.registers.RegisterView` is never
                accepted (scalar refs only). Among entries matching the same
                subsystem, the most recently registered one wins - readout
                selection does not accumulate like gate-channel selection.

        Raises:
            TypeError: If ``target`` is a `RegisterView`, or a `RegisterRef`
                not into a `QuantumRegister`.
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
        if isinstance(target, RegisterView):
            raise TypeError(
                "readout-error target must be a scalar RegisterRef or a "
                f"device resource label, not a RegisterView; got {target!r}"
            )
        if isinstance(target, RegisterRef) and not isinstance(
            target.register, QuantumRegister
        ):
            raise TypeError(
                "readout-error target refs must point into a "
                f"QuantumRegister, got a ref into {type(target.register).__name__}"
            )
        matrix.flags.writeable = False
        self._readout_errors.append((target, matrix))

    def add_continuous_noise(
        self,
        source: ContinuousNoise,
        *,
        target: _ContinuousSelector = None,
    ) -> None:
        """Attach typed continuous noise globally or to one subsystem.

        ``target=None`` applies to every physical subsystem selected by a pulse
        model, including model subsystems unused by the program. A logical
        `RegisterRef` or physical device label selects one occupied subsystem.
        Matching logical and physical specifics accumulate in registration
        order and replace defaults for that subsystem.
        """
        if not isinstance(source, ContinuousNoise):
            raise TypeError(f"expected a ContinuousNoise descriptor, got {source!r}")
        _validate_scalar_selector(target, "continuous-noise")
        self._continuous_noise.append(_ContinuousNoiseEntry(target, source))

    def continuous_noise_for(
        self,
        target: RegisterRef | None,
        device_label: DeviceOperand,
    ) -> tuple[ContinuousNoise, ...]:
        """Resolve continuous descriptors for one physical model subsystem."""
        defaults: list[ContinuousNoise] = []
        specifics: list[ContinuousNoise] = []
        for entry in self._continuous_noise:
            selector = entry.selector
            if selector is None:
                defaults.append(entry.source)
            elif isinstance(selector, RegisterRef):
                if target is not None and selector == target:
                    specifics.append(entry.source)
            elif selector == device_label:
                specifics.append(entry.source)
        return tuple(specifics if specifics else defaults)

    def continuous_noise_types(self) -> frozenset[type[ContinuousNoise]]:
        """Return every typed continuous descriptor class in this model."""
        return frozenset(type(entry.source) for entry in self._continuous_noise)

    def readout_error_for(
        self, target: RegisterRef, resource_layout: ResourceLayout
    ) -> np.ndarray | None:
        """Return the confusion matrix selected for one measured subsystem.

        A logical selector matches when it equals ``target``; a physical
        selector matches when it equals
        ``resource_layout.device_label(target)``. Readout errors do not
        accumulate: a matching specific selector replaces the all-target
        (``None``) default, and among several matching specific selectors
        the most recently registered one wins.

        Args:
            target: The measured subsystem's logical ref, as lowered from
                the program (never an engine index).
            resource_layout: The run's public resource layout, used to
                resolve physical selectors.
        """
        specific = fallback = None
        device_label_known = False
        device_label: DeviceOperand = None
        for selector, matrix in self._readout_errors:
            if selector is None:
                fallback = matrix
            elif isinstance(selector, RegisterRef):
                if selector == target:
                    specific = matrix
            else:
                if not device_label_known:
                    device_label = resource_layout.device_label(target)
                    device_label_known = True
                if device_label == selector:
                    specific = matrix
        return specific if specific is not None else fallback

    def validate_for(self, program: Program, resource_layout: ResourceLayout) -> None:
        """Validate every stored selector's identity legality for one run.

        This checks selector-identity legality only, not selector firing: a
        valid selector that matches no gate occurrence (or, for readout,
        names a subsystem that is never measured) is a permitted no-effect
        entry, not a validation error. A logical selector is legal when every
        ref it names belongs to ``program``'s quantum registers; a physical
        selector is legal when every label it names is a member of
        ``resource_layout.device_labels`` - the run's effective layout, which
        already establishes that a label denotes a legal, occupied device
        resource. Both stored shapes are checked: tuple gate-channel
        selectors and scalar readout selectors are validated independently,
        not through a shared representation.

        Args:
            program: The program this run will execute; defines the legal
                logical `RegisterRef` space.
            resource_layout: The run's effective resource layout; defines the
                legal physical device-label space.

        Raises:
            BackendValidationError: If any stored selector names a
                `RegisterRef` foreign to ``program`` or a device label absent
                from ``resource_layout.device_labels``.
        """
        program_refs = frozenset(
            ref
            for register in program.qreg
            for ref in (register[i] for i in range(register.size))
        )
        device_labels = resource_layout.device_labels

        if self.qubit_noise:
            raise BackendValidationError(
                "NoiseModel.qubit_noise is a legacy continuous-noise "
                "placeholder; migrate entries with add_continuous_noise()"
            )

        for entries in self._gate_channels.values():
            for entry in entries:
                selector = entry.selector
                if selector is None:
                    continue
                if _is_ref_selector(selector):
                    for ref in selector:
                        if ref not in program_refs:
                            raise BackendValidationError(
                                "noise selector names a RegisterRef that is "
                                f"not part of this program: {ref!r}"
                            )
                else:
                    for label in selector:
                        if label not in device_labels:
                            raise BackendValidationError(
                                "noise selector names a device resource "
                                f"label not in the effective resource "
                                f"layout: {label!r}"
                            )

        for selector, _matrix in self._readout_errors:
            if selector is None:
                continue
            if isinstance(selector, RegisterRef):
                if selector not in program_refs:
                    raise BackendValidationError(
                        "readout-error selector names a RegisterRef that is "
                        f"not part of this program: {selector!r}"
                    )
            else:
                if selector not in device_labels:
                    raise BackendValidationError(
                        "readout-error selector names a device resource "
                        f"label not in the effective resource layout: "
                        f"{selector!r}"
                    )

        for entry in self._continuous_noise:
            selector = entry.selector
            if selector is None:
                continue
            if isinstance(selector, RegisterRef):
                if selector not in program_refs:
                    raise BackendValidationError(
                        "continuous-noise selector names a RegisterRef that is "
                        f"not part of this program: {selector!r}"
                    )
            elif selector not in device_labels:
                raise BackendValidationError(
                    "continuous-noise selector names a device resource label "
                    f"not in the effective resource layout: {selector!r}"
                )

    def has_readout_error(self) -> bool:
        """Return whether any readout-error entry is registered."""
        return bool(self._readout_errors)

    def has_noise_for(self, operation: Operation | type[Operation]) -> bool:
        """Return whether any entry is keyed on this operation family."""
        return _resolve_operation_class(operation) in self._gate_channels

    def channel_types(self) -> frozenset[type[Channel]]:
        """Return every descriptor type attached anywhere in this model."""
        return frozenset(
            type(entry.channel)
            for entries in self._gate_channels.values()
            for entry in entries
        )


def _is_ref_selector(selector: tuple[DeviceOperand, ...]) -> bool:
    """Return whether a validated, homogeneous gate selector is RegisterRef-based,
    as opposed to a physical device-label selector."""
    return isinstance(selector[0], RegisterRef)


def _validate_scalar_selector(selector: Any, label: str) -> None:
    """Validate a logical ref or opaque hashable physical label."""
    if selector is None:
        return
    if isinstance(selector, RegisterView):
        raise TypeError(
            f"{label} target must be a scalar RegisterRef or a device "
            f"resource label, not a RegisterView; got {selector!r}"
        )
    if isinstance(selector, RegisterRef):
        if not isinstance(selector.register, QuantumRegister):
            raise TypeError(f"{label} target refs must point into a QuantumRegister")
        return
    try:
        hash(selector)
    except TypeError as exc:
        raise TypeError(f"{label} physical target must be hashable") from exc


def _normalize_selector(
    op_cls: type[Operation],
    targets: _GateTargetsArg,
) -> _GateSelector:
    """Validate and normalize an ``add_noise`` target selector.

    A selector is ``None``, a tuple wholly of `RegisterRef` (logical), a
    tuple wholly of some other hashable (physical device resource labels),
    or a single `RegisterRef`/hashable as shorthand for a one-element
    tuple - the length check below then rejects it for any operation whose
    fixed arity isn't 1. Mixing the two tuple forms, or including a
    `RegisterView`, is rejected; a physical label is opaque and is not
    itself validated (any hashable value is a legal device resource label
    until run-time validation against an actual `ResourceLayout`).
    """
    if targets is None:
        return None
    selector = targets if isinstance(targets, tuple) else (targets,)
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


def _normalize_slots(
    op_cls: type[Operation],
    channel: Channel,
    selector: _GateSelector,
    slots: tuple[int, ...] | int | None,
) -> _Slots:
    """Validate a positional channel extent and normalize full coverage."""
    if slots is None:
        normalized = None
    else:
        normalized = (slots,) if isinstance(slots, int) else slots
        if not isinstance(normalized, tuple):
            raise TypeError("slots must be an int, a tuple of ints, or None")
        if not normalized:
            raise ValueError("slots must be non-empty")
        if any(not isinstance(i, int) or isinstance(i, bool) for i in normalized):
            raise TypeError("each slot must be int")
        if any(i < 0 for i in normalized):
            raise ValueError("slots must be non-negative")
        if any(left >= right for left, right in zip(normalized, normalized[1:])):
            raise ValueError("slots must be strictly increasing")
        if (
            op_cls._num_subsystems is not None
            and max(normalized) >= op_cls._num_subsystems
        ):
            raise ValueError(
                f"slot {max(normalized)} is out of range for {op_cls.__name__}"
            )
        if selector is not None and max(normalized) >= len(selector):
            raise ValueError("slot is out of range for the explicit targets selector")
        if (
            channel.num_subsystems is not None
            and len(normalized) != channel.num_subsystems
        ):
            raise ValueError(
                f"{type(channel).__name__} takes {channel.num_subsystems} subsystem(s), "
                f"got {len(normalized)} slots"
            )
        if op_cls._num_subsystems is not None and normalized == tuple(
            range(op_cls._num_subsystems)
        ):
            normalized = None

    if (
        normalized is None
        and channel.num_subsystems is not None
        and op_cls._num_subsystems is not None
        and channel.num_subsystems != op_cls._num_subsystems
    ):
        raise ValueError(
            f"{type(channel).__name__} cannot attach to all {op_cls._num_subsystems} "
            "subsystems; use slots="
        )
    return normalized
