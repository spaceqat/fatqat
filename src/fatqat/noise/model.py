"""Backend-neutral authoring, scope validation, and selection for noise."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, overload

from ..errors import BackendValidationError
from ..implementation._operation_registry import _resolve_operation_class
from ..operations import BarrierGate, Operation, PutGate, ResetGate
from ..program import Program
from ..registers import QuantumRegister, RegisterRef, RegisterView
from ..resource_layout import DeviceOperand, ResourceLayout
from .base import Channel
from .catalog import Depolarizing
from .loss import Loss
from .readout import ReadoutConfusion

_Selector = tuple[RegisterRef, ...] | tuple[DeviceOperand, ...] | None
_TargetsArg = _Selector | RegisterRef | DeviceOperand
_ReadoutSelector = RegisterRef | DeviceOperand | None
_TargetPositions = tuple[int, ...] | None
_Declaration = Channel | Loss


class _OmittedArgument:
    """Distinguish an omitted scope keyword from an explicit ``None``."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<omitted>"


_OMITTED = _OmittedArgument()
_UNRESOLVED_DEVICE_LABEL = object()


@dataclass(frozen=True, slots=True)
class _NoiseRegistration:
    """One physical declaration plus its backend-neutral routing facts."""

    declaration: _Declaration
    operation: type[Operation] | None
    selector: _Selector
    target_positions: _TargetPositions


class NoiseModel:
    """Collect noise sources and choose where they apply.

    A model can attach a ``Channel`` or ``Loss`` to matching operations, apply
    a supported single-subsystem ``Channel`` as background noise, and alter
    reported measurements with ``ReadoutConfusion``. Calls to ``add()``
    accumulate. Different noise types combine, while overlapping rules of the
    same concrete type are rejected.

    Operation matching uses the exact concrete class. Parameters on an
    operation instance do not narrow the match, and a base class does not
    include its subclasses. Attached noise follows the operation's classical
    condition.

    Examples:
        Apply dephasing after every ``X`` and confuse every reported
        measurement:

        >>> import fatqat as fq
        >>> import fatqat.operations as ops
        >>> noise = fq.NoiseModel()
        >>> noise.add(fq.noise.PhaseDamping(p=0.01), operation=ops.X)
        >>> noise.add(
        ...     fq.noise.ReadoutConfusion([[0.98, 0.04], [0.02, 0.96]]),
        ... )

    """

    def __init__(self) -> None:
        self._noise_registrations: list[_NoiseRegistration] = []
        self._readout_registrations: list[
            tuple[_ReadoutSelector, ReadoutConfusion]
        ] = []

    @overload
    def add(
        self,
        declaration: ReadoutConfusion,
        *,
        targets: RegisterRef | DeviceOperand | None = None,
    ) -> None: ...

    @overload
    def add(
        self,
        declaration: Channel | Loss,
        *,
        operation: Operation | type[Operation] | None = None,
        targets: (
            RegisterRef
            | DeviceOperand
            | tuple[RegisterRef, ...]
            | tuple[DeviceOperand, ...]
            | None
        ) = None,
        target_positions: tuple[int, ...] | int | None = None,
    ) -> None: ...

    def add(
        self,
        declaration: Channel | Loss | ReadoutConfusion,
        *,
        operation: Operation | type[Operation] | None = _OMITTED,
        targets: (
            RegisterRef
            | DeviceOperand
            | tuple[RegisterRef, ...]
            | tuple[DeviceOperand, ...]
            | None
        ) = None,
        target_positions: tuple[int, ...] | int | None = _OMITTED,
    ) -> None:
        """Add a noise source to the model without replacing existing rules.

        Args:
            declaration: A ``Channel``, ``Loss``, or ``ReadoutConfusion``
                instance. The object validates its parameters at construction;
                the backend checks whether it supports the noise when
                configured.
            operation: Operation instance or subclass choosing which
                operations receive the noise. Matching uses the exact concrete
                class, so all parameterizations of ``ops.RX`` match
                ``operation=ops.RX`` but subclasses do not. Omit this argument,
                or pass ``None``, for background scope. ``ReadoutConfusion``
                must omit the keyword entirely, including an explicit ``None``.
                ``Barrier``, ``Reset``, and direct-control operations have no
                attachable boundary; ``Put`` accepts only ``Loss``.
            targets: Operands where the noise applies:

                * For operation noise, ``None`` means every operation of the
                  exact class. A scalar is unary shorthand. A
                  non-empty tuple is an exact ordered selector whose length
                  must match a fixed operation's number of operands. Its
                  elements must be all quantum ``RegisterRef`` values or all
                  hashable device labels; lists, register views, and mixed
                  logical/device-label tuples are invalid.
                * For background noise, supply exactly one logical reference or
                  hashable device label, either as a scalar or one-element
                  tuple. The noise must act on exactly one subsystem.
                * For readout confusion, ``None`` means every measured operand;
                  otherwise supply one scalar logical reference or non-tuple
                  hashable device label. Correlated readout is unsupported.

                A tuple is always interpreted as the ordered selector itself.
                To use a tuple-valued device label for operation or
                background noise, nest it in the selector, for example
                ``targets=(("site", 0),)``. Readout selectors reject tuples.
            target_positions: For operation noise only, an integer or a
                non-empty, strictly increasing tuple of non-negative integers.
                Booleans are not accepted as integer positions.
                Positions index the matching operation's targets and determine
                both the affected operands and their order. ``None`` or
                omission selects all operation targets. The selected width
                must match the noise type's fixed width. Background scope
                accepts only ``None`` or omission. Readout must omit this
                argument and also rejects an explicit ``None``.

        Returns:
            ``None``. A successful call appends one registration.

        Raises:
            TypeError: If the noise object, operation, selector kind, selector
                element, or position type is invalid, or readout receives
                ``operation`` or ``target_positions``.
            ValueError: If the scope, fixed number of operands, target
                position, or an overlap recognizable from the stored targets
                is invalid. Failed additions are atomic.

        Notes:
            When you run a program, FATQAT checks references and device labels,
            resolves variadic operation widths, and detects logical and
            device-label selectors that name the same subsystem. Failures
            raise ``BackendValidationError``. A valid selector that never
            matches is a no-op.

            Rules of the same type may use disjoint targets or positions.
            Universal and targeted readout rules cannot coexist, and at most
            one confusion matrix may apply to a measured operand.

        Examples:
            Apply independent local damping to the ordered operands of every
            ``CZ``:

            >>> import fatqat as fq
            >>> import fatqat.operations as ops
            >>> noise = fq.NoiseModel()
            >>> noise.add(
            ...     fq.noise.AmplitudeDamping(p=0.002),
            ...     operation=ops.CZ,
            ...     target_positions=0,
            ... )
            >>> noise.add(
            ...     fq.noise.AmplitudeDamping(p=0.003),
            ...     operation=ops.CZ,
            ...     target_positions=1,
            ... )
        """
        if isinstance(declaration, ReadoutConfusion):
            if operation is not _OMITTED:
                raise TypeError(
                    "ReadoutConfusion is intrinsically measurement-bound; "
                    "omit operation"
                )
            if target_positions is not _OMITTED:
                raise TypeError("ReadoutConfusion does not accept target_positions")
            selector = _normalize_readout_selector(targets)
            _check_readout_conflict(self._readout_registrations, selector)
            self._readout_registrations.append((selector, declaration))
            return

        if not isinstance(declaration, (Channel, Loss)):
            raise TypeError(
                "declaration must be a Channel, Loss, or ReadoutConfusion, "
                f"got {declaration!r}"
            )

        op_value = None if operation is _OMITTED else operation
        positions_value = None if target_positions is _OMITTED else target_positions
        if op_value is None:
            if targets is None:
                raise ValueError(
                    "background noise requires exactly one target; omitting "
                    "operation is not shorthand for every gate"
                )
            if positions_value is not None:
                raise ValueError("background noise does not accept target_positions")
            selector = _normalize_background_selector(targets)
            if isinstance(declaration, Depolarizing) and declaration.p is not None:
                raise ValueError(
                    "Depolarizing(p) is width-agnostic and cannot be registered "
                    "as single-subsystem background noise; use "
                    "Depolarizing(rate=...)"
                )
            if _declaration_arity(declaration) != 1:
                raise ValueError(
                    "background noise requires a single-subsystem declaration; "
                    f"{type(declaration).__name__} is not authored with arity 1"
                )
            op_cls = None
            positions = None
        else:
            op_cls = _normalize_noise_operation(op_value)
            if op_cls is PutGate and not isinstance(declaration, Loss):
                raise ValueError("Put accepts only Loss as loading inefficiency")
            selector = _normalize_occurrence_selector(op_cls, targets)
            positions = _normalize_target_positions(
                op_cls, declaration, selector, positions_value
            )

        proposed = _NoiseRegistration(declaration, op_cls, selector, positions)
        _check_noise_conflict(self._noise_registrations, proposed)
        self._noise_registrations.append(proposed)

    def _noise_for_occurrence(
        self,
        operation: Operation | type[Operation],
        targets: tuple[RegisterRef, ...],
        resource_layout: ResourceLayout,
    ) -> list[tuple[_Declaration, tuple[RegisterRef, ...]]]:
        """Select declarations for an operation with exact ordered targets."""
        op_cls = _resolve_operation_class(operation)
        occurrence = tuple(targets)
        physical: tuple[DeviceOperand, ...] | None = None
        matches: list[_NoiseRegistration] = []
        for registration in self._noise_registrations:
            if registration.operation is not op_cls:
                continue
            selector = registration.selector
            if selector is None or (
                _is_ref_selector(selector) and selector == occurrence
            ):
                matches.append(registration)
                continue
            if not _is_ref_selector(selector):
                if physical is None:
                    physical = resource_layout.device_labels_for(occurrence)
                if selector == physical:
                    matches.append(registration)

        _reject_actual_noise_conflicts(matches)
        resolved: list[tuple[_Declaration, tuple[RegisterRef, ...]]] = []
        for registration in matches:
            positions = registration.target_positions
            if positions is not None and max(positions) >= len(occurrence):
                raise BackendValidationError(
                    f"target position {max(positions)} is out of range for "
                    f"{type(registration.declaration).__name__} on this "
                    f"{op_cls.__name__} operation with {len(occurrence)} "
                    "subsystem(s)"
                )
            extent = (
                occurrence
                if positions is None
                else tuple(occurrence[index] for index in positions)
            )
            expected = _declaration_arity(registration.declaration)
            if expected is not None and len(extent) != expected:
                raise BackendValidationError(
                    f"{type(registration.declaration).__name__} acts on "
                    f"{expected} subsystem(s), got an extent of {len(extent)} "
                    f"for {op_cls.__name__}"
                )
            resolved.append((registration.declaration, extent))
        return resolved

    def _background_noise_for(
        self,
        target: RegisterRef | None,
        device_label: DeviceOperand,
    ) -> tuple[Channel, ...]:
        """Select local background declarations for one physical subsystem."""
        matches: list[_NoiseRegistration] = []
        for registration in self._noise_registrations:
            if registration.operation is not None:
                continue
            selector = registration.selector
            if _is_ref_selector(selector):
                if target is not None and selector == (target,):
                    matches.append(registration)
            elif selector == (device_label,):
                matches.append(registration)
        _reject_actual_noise_conflicts(matches)
        return tuple(registration.declaration for registration in matches)

    def _readout_confusion_for(
        self,
        target: RegisterRef,
        resource_layout: ResourceLayout,
    ) -> ReadoutConfusion | None:
        """Select the unique readout declaration for one measured operand."""
        device_label: DeviceOperand | object = _UNRESOLVED_DEVICE_LABEL
        matches: list[tuple[_ReadoutSelector, ReadoutConfusion]] = []
        for selector, declaration in self._readout_registrations:
            if selector is None:
                matches.append((selector, declaration))
            elif isinstance(selector, RegisterRef):
                if selector == target:
                    matches.append((selector, declaration))
            else:
                if device_label is _UNRESOLVED_DEVICE_LABEL:
                    device_label = resource_layout.device_label(target)
                if selector == device_label:
                    matches.append((selector, declaration))
        if len(matches) > 1:
            selectors = ", ".join(repr(selector) for selector, _ in matches)
            raise BackendValidationError(
                "multiple ReadoutConfusion registrations match measured target "
                f"{target!r}: {selectors}"
            )
        return matches[0][1] if matches else None

    def _validate_for(
        self,
        program: Program,
        legal_device_operands: frozenset[DeviceOperand],
    ) -> None:
        """Validate stored logical ownership and physical-label legality."""
        program_refs = frozenset(
            register[index]
            for register in program.quantum_registers
            for index in range(register.size)
        )
        for registration in self._noise_registrations:
            _validate_selector_for_run(
                registration.selector,
                program_refs,
                legal_device_operands,
                "noise",
            )
        for selector, _declaration in self._readout_registrations:
            if selector is None:
                continue
            _validate_selector_for_run(
                (selector,),
                program_refs,
                legal_device_operands,
                "readout confusion",
            )

    def _copy(self) -> NoiseModel:
        """Return an independent registration container sharing declarations."""
        copied = NoiseModel()
        copied._noise_registrations = self._noise_registrations.copy()
        copied._readout_registrations = self._readout_registrations.copy()
        return copied

    def _noise_sources(
        self,
    ) -> tuple[tuple[_Declaration, type[Operation] | None], ...]:
        """Project only facts needed for backend capability classification."""
        return tuple(
            (registration.declaration, registration.operation)
            for registration in self._noise_registrations
        )

    def _readout_confusions(self) -> tuple[ReadoutConfusion, ...]:
        """Project readout values for backend capability classification."""
        return tuple(
            declaration for _selector, declaration in self._readout_registrations
        )


def _normalize_noise_operation(value: object) -> type[Operation]:
    op_cls = _resolve_operation_class(value)
    if op_cls._is_direct_control:
        raise ValueError(
            f"{op_cls.__name__} is a direct-control operation and has no "
            "attachable noise boundary"
        )
    if op_cls is BarrierGate:
        raise ValueError("Barrier is a compiler marker with no noise boundary")
    if op_cls is ResetGate:
        raise ValueError("Reset has no attached-noise realization")
    return op_cls


def _normalize_occurrence_selector(
    op_cls: type[Operation], targets: _TargetsArg
) -> _Selector:
    if targets is None:
        return None
    if isinstance(targets, list):
        raise TypeError(
            "operation noise targets must be a scalar selector or a non-empty "
            "ordered tuple; lists are not accepted"
        )
    selector = targets if isinstance(targets, tuple) else (targets,)
    if not selector:
        raise ValueError("targets must be None or a non-empty ordered selector")
    _validate_homogeneous_selector(selector, "noise")
    arity = op_cls.num_subsystems
    if arity is not None and len(selector) != arity:
        raise ValueError(
            f"{op_cls.__name__} targets {arity} subsystem(s), got a selector "
            f"of length {len(selector)}"
        )
    return selector


def _normalize_background_selector(targets: _TargetsArg) -> _Selector:
    selector = targets if isinstance(targets, tuple) else (targets,)
    if len(selector) != 1:
        raise ValueError("background noise targets must select exactly one subsystem")
    _validate_homogeneous_selector(selector, "background noise")
    return selector


def _normalize_readout_selector(targets: _TargetsArg) -> _ReadoutSelector:
    if isinstance(targets, tuple):
        raise TypeError(
            "ReadoutConfusion targets must be one scalar RegisterRef or device "
            "label; correlated readout is not supported"
        )
    _validate_scalar_selector(targets, "ReadoutConfusion")
    return targets


def _validate_homogeneous_selector(selector: tuple[Any, ...], label: str) -> None:
    for target in selector:
        if target is None:
            raise TypeError(f"{label} physical targets cannot be None")
        _validate_scalar_selector(target, label)
    ref_flags = tuple(isinstance(target, RegisterRef) for target in selector)
    if any(ref_flags) and not all(ref_flags):
        raise TypeError(
            f"{label} targets must be all RegisterRef or all physical device labels"
        )


def _validate_scalar_selector(selector: object, label: str) -> None:
    if selector is None:
        return
    if isinstance(selector, RegisterView):
        raise TypeError(
            f"{label} target must be a scalar RegisterRef or physical device "
            f"label, not RegisterView; got {selector!r}"
        )
    if isinstance(selector, RegisterRef):
        if not isinstance(selector.register, QuantumRegister):
            raise TypeError(f"{label} target refs must point into a QuantumRegister")
        return
    try:
        hash(selector)
    except TypeError as exc:
        raise TypeError(f"{label} physical target must be hashable") from exc


def _normalize_target_positions(
    op_cls: type[Operation],
    declaration: _Declaration,
    selector: _Selector,
    positions: object,
) -> _TargetPositions:
    if positions is None:
        normalized = None
    else:
        normalized = (positions,) if isinstance(positions, int) else positions
        if not isinstance(normalized, tuple):
            raise TypeError("target_positions must be an int, tuple of ints, or None")
        if not normalized:
            raise ValueError("target_positions must be non-empty")
        if any(
            not isinstance(index, int) or isinstance(index, bool)
            for index in normalized
        ):
            raise TypeError("each target position must be an int")
        if any(index < 0 for index in normalized):
            raise ValueError("target_positions must be non-negative")
        if any(left >= right for left, right in zip(normalized, normalized[1:])):
            raise ValueError("target_positions must be strictly increasing")

    occurrence_width = op_cls.num_subsystems
    if occurrence_width is None and selector is not None:
        occurrence_width = len(selector)
    if occurrence_width is not None and normalized is not None:
        if max(normalized) >= occurrence_width:
            raise ValueError(
                f"target position {max(normalized)} is out of range for "
                f"{op_cls.__name__}"
            )
        if normalized == tuple(range(occurrence_width)):
            normalized = None

    declaration_width = _declaration_arity(declaration)
    extent_width = len(normalized) if normalized is not None else occurrence_width
    if declaration_width is not None and extent_width is not None:
        if declaration_width != extent_width:
            raise ValueError(
                f"{type(declaration).__name__} acts on {declaration_width} "
                f"subsystem(s), got an extent of {extent_width}; use "
                "target_positions to select its extent"
            )
    return normalized


def _declaration_arity(declaration: _Declaration) -> int | None:
    return None if isinstance(declaration, Loss) else declaration.num_subsystems


def _is_ref_selector(selector: _Selector) -> bool:
    return selector is not None and isinstance(selector[0], RegisterRef)


def _selectors_can_overlap(left: _Selector, right: _Selector) -> bool:
    if left is None or right is None:
        return True
    left_refs = _is_ref_selector(left)
    right_refs = _is_ref_selector(right)
    if left_refs != right_refs:
        return False
    return left == right


def _positions_overlap(left: _TargetPositions, right: _TargetPositions) -> bool:
    if left is None or right is None:
        return True
    return bool(set(left).intersection(right))


def _registrations_conflict(
    left: _NoiseRegistration,
    right: _NoiseRegistration,
    *,
    actual_match: bool = False,
) -> bool:
    if type(left.declaration) is not type(right.declaration):
        return False
    if left.operation is not right.operation:
        return False
    if not actual_match and not _selectors_can_overlap(left.selector, right.selector):
        return False
    return _positions_overlap(left.target_positions, right.target_positions)


def _check_noise_conflict(
    registrations: list[_NoiseRegistration], proposed: _NoiseRegistration
) -> None:
    for existing in registrations:
        if _registrations_conflict(existing, proposed):
            scope = (
                "background"
                if proposed.operation is None
                else proposed.operation.__name__
            )
            raise ValueError(
                f"overlapping {type(proposed.declaration).__name__} noise is "
                f"already registered in {scope} scope"
            )


def _reject_actual_noise_conflicts(matches: list[_NoiseRegistration]) -> None:
    for index, left in enumerate(matches):
        for right in matches[index + 1 :]:
            if _registrations_conflict(left, right, actual_match=True):
                raise BackendValidationError(
                    f"logical and physical noise selectors {left.selector!r} and "
                    f"{right.selector!r} both match the same "
                    f"{type(left.declaration).__name__} extent"
                )


def _check_readout_conflict(
    registrations: list[tuple[_ReadoutSelector, ReadoutConfusion]],
    proposed: _ReadoutSelector,
) -> None:
    for selector, _declaration in registrations:
        if selector is None and proposed is None:
            raise ValueError("universal ReadoutConfusion is already registered")
        if selector is None or proposed is None:
            raise ValueError(
                "universal and targeted ReadoutConfusion registrations cannot coexist"
            )
        selector_is_ref = isinstance(selector, RegisterRef)
        proposed_is_ref = isinstance(proposed, RegisterRef)
        if selector_is_ref == proposed_is_ref and selector == proposed:
            raise ValueError(
                f"ReadoutConfusion is already registered for target {proposed!r}"
            )


def _validate_selector_for_run(
    selector: _Selector,
    program_refs: frozenset[RegisterRef],
    legal_device_operands: frozenset[DeviceOperand],
    label: str,
) -> None:
    if selector is None:
        return
    if _is_ref_selector(selector):
        for ref in selector:
            if ref not in program_refs:
                raise BackendValidationError(
                    f"{label} selector names a RegisterRef outside this program: "
                    f"{ref!r}"
                )
    else:
        for device_label in selector:
            if device_label not in legal_device_operands:
                raise BackendValidationError(
                    f"{label} selector names a device resource label outside "
                    f"the backend's legal universe: {device_label!r}"
                )
