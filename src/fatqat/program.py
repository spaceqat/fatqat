"""Program container plus the AppliedOperation value object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TypeVar

from .operations import Measurement, Operation
from .registers import QuantumRegister, RegisterRef, RegisterView, ClassicalRegister

ConditionTerm = tuple[RegisterRef, int]
Condition = tuple[ConditionTerm, ...] | None
RegisterT = TypeVar("RegisterT", QuantumRegister, ClassicalRegister)

# A frontend target expression: either a resolved scalar reference, or (for
# the view-capable operations named on `Operation._accepts_views`) a
# structured `RegisterView` selecting multiple members of one `GridRegister`.
# See `AppliedOperation.targets` and `Program.add`.
QuantumTarget = RegisterRef | RegisterView


@dataclass(frozen=True)
class AppliedOperation:
    """An operation bound to resolved quantum register references.

    ``Program.add`` creates these objects after validating the operation and
    resolving integer operands or explicit ``RegisterRef`` objects.

    ``__post_init__`` intentionally does not re-validate ``targets``' element
    types or tuple-ness: ``Program.add()`` already guarantees well-formed
    ``RegisterRef`` tuples via ``_resolve_quantum_ref``, and duplicating that check
    here would just be the same validation twice for the path essentially all
    callers use. Constructing this class directly (bypassing ``Program``)
    skips that guarantee - malformed input (wrong register kind, a list
    instead of a tuple) will not raise here, and will instead surface later
    as a less specific error during backend lowering, or make the instance
    unhashable. This is a deliberate no-duplicate-validation tradeoff, not an
    oversight.

    Attributes:
        operation: Operation instance to execute.
        targets: Quantum target expressions consumed by the operation -- each
            either a resolved scalar ``RegisterRef``, or (for view-capable
            operations only) a structured ``RegisterView``.
        condition: Optional AND tuple of classical references and literal values.
    """

    operation: Operation
    targets: tuple[QuantumTarget, ...]
    condition: Condition = None

    def __post_init__(self) -> None:
        expected = self.operation.num_subsystems
        if expected is None:
            if len(self.targets) < 1:
                raise ValueError(f"{self.operation.name} expects at least one target")
        elif len(self.targets) != expected:
            raise ValueError(
                f"{self.operation.name} expects {expected} target(s), "
                f"got {len(self.targets)}"
            )

        has_view = any(isinstance(t, RegisterView) for t in self.targets)
        if has_view:
            if not self.operation.accepts_views:
                raise ValueError(
                    f"{self.operation.name} does not accept a RegisterView target; "
                    "only view-capable operations may be applied to a view"
                )
            # Scalar validation (duplicate/member checks, validate_targets())
            # is deferred to binding/expansion, which resolves each view to
            # concrete scalar refs first. See spec section 5.2.
            return

        seen: set[tuple[int, int]] = set()
        for t in self.targets:
            key = (id(t.register), t.index)
            if key in seen:
                raise ValueError(
                    f"{self.operation.name}: target qubit {t!r} appears more than once"
                )
            seen.add(key)
        self.operation.validate_targets(self.targets)


class Program:
    """User-facing quantum program container.

    A program owns read-only public quantum/classical register tuples plus an
    ordered read-only view of applied operations and measurements. Public
    mutation goes through ``add()`` and ``add_measurement()``, which keep the
    internal instruction list well formed. Operations are executed in
    insertion order. Integer operands are accepted only when there is exactly
    one register of the relevant kind; otherwise users must pass explicit
    ``RegisterRef`` objects such as ``program.qreg[0][1]``.

    Examples:
        Build a two-qubit program, add gates, then measure both qubits:

        >>> import fatqat as fq
        >>> program = fq.Program(2, 2)
        >>> program.add(fq.ops.H, 0)
        >>> program.add(fq.ops.CZ, (0, 1))
        >>> program.add_measurement(0, 0)
        >>> program.add_measurement(1, 1)
        >>> len(program.operations)
        4
    """

    def __init__(
        self,
        qreg: int | list[QuantumRegister],
        clreg: int | list[ClassicalRegister] = 0,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Create a program from register counts or explicit register lists.

        Args:
            qreg: Number of default quantum bits, or explicit quantum registers.
                Stored publicly as a read-only tuple.
            clreg: Number of default classical bits, or explicit classical
                registers. A value of `0` creates no classical register. Stored
                publicly as a read-only tuple.
            metadata: Optional user metadata copied into the program.

        Raises:
            TypeError: If register specs are neither integer counts nor lists of
                the expected register type.
            ValueError: If an integer register count is negative.
        """
        self.qreg: tuple[QuantumRegister, ...] = tuple(
            self._coerce_registers(qreg, QuantumRegister, "q")
        )
        self.clreg: tuple[ClassicalRegister, ...] = tuple(
            self._coerce_registers(clreg, ClassicalRegister, "c")
        )
        self._operations: list[AppliedOperation | Measurement] = []
        self._operations_view: tuple[AppliedOperation | Measurement, ...] | None = ()
        self.metadata: dict[str, Any] = dict(metadata) if metadata else {}

    @property
    def operations(self) -> tuple[AppliedOperation | Measurement, ...]:
        """Ordered operation and measurement steps as a cached read-only tuple view."""
        if self._operations_view is None:
            self._operations_view = tuple(self._operations)
        return self._operations_view

    @staticmethod
    def _coerce_registers(
        spec: int | list[RegisterT] | tuple[RegisterT, ...],
        cls: type[RegisterT],
        default_name: str,
    ) -> list[RegisterT]:
        """Normalize an integer count or explicit register list to a list."""
        if type(spec) is int:
            if spec < 0:
                raise ValueError(f"register count must be >= 0, got {spec}")
            return [cls(spec, name=default_name)] if spec > 0 else []
        if not isinstance(spec, (list, tuple)):
            raise TypeError(
                f"registers must be an int count or a list of {cls.__name__}, "
                f"got {type(spec).__name__!r}"
            )
        for r in spec:
            if not isinstance(r, cls):
                raise TypeError(
                    f"register list must contain {cls.__name__} instances, "
                    f"got {type(r).__name__!r}"
                )
        return list(spec)

    def _resolve_ref(
        self,
        operand: int | RegisterRef,
        regs: list[RegisterT],
        kind_cls: type[RegisterT],
        kind_name: str,
    ) -> RegisterRef:
        """Resolve an integer or explicit ref against one register kind.

        Bare integers are only accepted when there is exactly one register of
        the relevant kind; multiple registers require an explicit RegisterRef.
        """
        if isinstance(operand, RegisterRef):
            if not isinstance(operand.register, kind_cls):
                raise TypeError(f"expected a {kind_name} ref")
            if not any(operand.register is r for r in regs):
                raise ValueError(f"ref does not belong to this program's {kind_name}s")
            return operand
        if type(operand) is not int:
            raise TypeError(
                f"operand must be int or RegisterRef, got {type(operand)!r}"
            )
        if len(regs) != 1:
            raise TypeError(
                "integer operands are only allowed when there is exactly one "
                "register of the relevant kind; pass an explicit RegisterRef "
                "(e.g. qreg[0] or clreg[0]) instead"
            )
        # Bounds and negative-index checks are delegated to Register.__getitem__,
        # which raises IndexError.
        return regs[0][operand]

    def _resolve_quantum_ref(self, operand: int | RegisterRef) -> RegisterRef:
        return self._resolve_ref(
            operand, self.qreg, QuantumRegister, "quantum register"
        )

    def _resolve_quantum_target(
        self, operand: int | RegisterRef | RegisterView
    ) -> QuantumTarget:
        """Resolve one ``Program.add()`` operand, accepting a ``RegisterView``
        in addition to everything ``_resolve_quantum_ref`` accepts.

        This is deliberately a separate path from ``_resolve_quantum_ref``:
        that helper is also used by ``add_measurement``/``measure_all``,
        which must keep rejecting views (see spec section 5.1).
        """
        if isinstance(operand, RegisterView):
            if not any(operand.register is r for r in self.qreg):
                raise ValueError(
                    "view's register does not belong to this program's quantum registers"
                )
            return operand
        return self._resolve_quantum_ref(operand)

    def _resolve_classical_ref(self, operand: int | RegisterRef) -> RegisterRef:
        return self._resolve_ref(
            operand, self.clreg, ClassicalRegister, "classical register"
        )

    def add(
        self,
        op: Operation,
        targets: (
            int
            | RegisterRef
            | RegisterView
            | tuple[int | RegisterRef | RegisterView, ...]
        ),
        *,
        condition=None,
    ) -> None:
        """Append an operation to the program in place.

        Args:
            op: Operation instance to append. Fixed gates are available as
                singleton values such as ``ops.X``; parametric gates should be
                instantiated, such as ``ops.RX(0.2)``.
            targets: Target subsystem operand, or tuple of target operands for
                multi-subsystem gates (e.g. ``(0, 1)`` for ``CZ``). Each operand
                may be an integer when unambiguous, an explicit ``RegisterRef``,
                or (only for view-capable operations -- currently RX, RY, RZ,
                CX, CZ) a ``RegisterView`` such as ``atoms.row(0)``.
            condition: Optional single condition ``(clbit, value)`` or
                sequence of conditions. Conditions are normalized to an AND
                tuple.

        Raises:
            TypeError: If ``op`` is not an ``Operation``, if operands have the
                wrong register kind, or if integer operands are ambiguous.
            ValueError: If target arity is wrong, a target is repeated, a ref
                or view is foreign to the program, ``op`` does not accept a
                ``RegisterView`` target, or ``condition`` is empty.
            IndexError: If an integer operand is outside the relevant register.

        Examples:
            Add fixed and parametric gates:

            >>> import fatqat as fq
            >>> program = fq.Program(2)
            >>> program.add(fq.ops.H, 0)
            >>> program.add(fq.ops.CZ, (0, 1))
            >>> program.add(fq.ops.RX(0.2), 0)
        """
        if not isinstance(op, Operation):
            raise TypeError(
                f"op must be an Operation instance, got {type(op)!r} "
                "(did you forget to call a parametric gate, e.g. ops.RX(0.2)?)"
            )
        operands = targets if isinstance(targets, tuple) else (targets,)
        target_refs = tuple(self._resolve_quantum_target(o) for o in operands)
        normalized = self._normalize_condition(condition)
        self._operations.append(
            AppliedOperation(operation=op, targets=target_refs, condition=normalized)
        )
        self._operations_view = None

    def _normalize_condition(self, condition: Any) -> Condition:
        """Normalize user conditions to an AND tuple of classical refs and values."""
        if condition is None:
            return None
        if len(condition) == 0:
            raise ValueError(
                "condition is empty; pass None or omit the argument for an unconditional operation"
            )
        # Discriminate single term `(slot, lit)` from a sequence of terms.
        terms = condition if isinstance(condition[0], tuple) else (condition,)
        normalized: list[ConditionTerm] = []
        for slot, literal in terms:
            ref = self._resolve_classical_ref(slot)
            # bool is deliberately accepted here (not just tolerated): a
            # condition literal is itself boolean in spirit for a dim=2 clbit,
            # so `condition=(c, True)` is a legitimate spelling of `(c, 1)`,
            # unlike the strict int-only fields elsewhere (size, dim, index).
            if not isinstance(literal, int):
                raise TypeError(
                    f"condition literal must be int, got {type(literal).__name__!r}"
                )
            value = int(literal)
            if not 0 <= value < ref.register.dim:
                raise ValueError(
                    f"condition literal {value} out of range for clbit dim "
                    f"{ref.register.dim} (must be 0 <= literal < dim)"
                )
            normalized.append((ref, value))
        return tuple(normalized)

    def add_measurement(
        self,
        targets: int | RegisterRef | tuple[int | RegisterRef, ...],
        outputs: int | RegisterRef | tuple[int | RegisterRef, ...],
    ) -> None:
        """Append a measurement of one or more subsystems into classical slots.

        Args:
            targets: Quantum operand(s) to measure, as an integer, explicit
                ``RegisterRef``, or tuple of operands for a grouped
                measurement.
            outputs: Classical operand(s) to write, as an integer, explicit
                ``RegisterRef``, or tuple of operands matching ``targets`` in
                count.

        Raises:
            TypeError: If operands have the wrong register kind or integer
                operands are ambiguous.
            ValueError: If ``targets``/``outputs`` have mismatched or zero
                length, or an explicit ref is foreign to the program.
            IndexError: If an integer operand is outside the relevant register.

        Examples:
            Add a terminal measurement:

            >>> import fatqat as fq
            >>> program = fq.Program(1, 1)
            >>> program.add(fq.ops.X, 0)
            >>> program.add_measurement(0, 0)

            Add a grouped measurement:

            >>> program2 = fq.Program(2, 2)
            >>> program2.add(fq.ops.H, 0)
            >>> program2.add(fq.ops.CZ, (0, 1))
            >>> program2.add_measurement((0, 1), (0, 1))
        """
        q_operands = targets if isinstance(targets, tuple) else (targets,)
        c_operands = outputs if isinstance(outputs, tuple) else (outputs,)
        target_refs = tuple(self._resolve_quantum_ref(q) for q in q_operands)
        output_refs = tuple(self._resolve_classical_ref(c) for c in c_operands)
        # Length, non-empty, and per-pair dim invariants are enforced once in
        # Measurement.__post_init__.
        self._operations.append(Measurement(targets=target_refs, outputs=output_refs))
        self._operations_view = None

    def measure_all(self) -> None:
        """Measure every subsystem into every classical slot in declaration order.

        Raises:
            ValueError: If the program has a different number of quantum
                subsystems than classical slots, or has no registers of either
                kind.
        """
        targets = tuple(
            ref for reg in self.qreg for ref in (reg[i] for i in range(reg.size))
        )
        outputs = tuple(
            ref for reg in self.clreg for ref in (reg[i] for i in range(reg.size))
        )
        # Equal-count and non-empty invariants are enforced once in
        # Measurement.__post_init__, reached through add_measurement.
        self.add_measurement(targets, outputs)

    def copy(self) -> "Program":
        """Return an independent copy with private operation storage and copied metadata."""
        new = Program.__new__(Program)
        new.qreg = tuple(self.qreg)
        new.clreg = tuple(self.clreg)
        new._operations = list(self._operations)
        new._operations_view = tuple(new._operations)
        new.metadata = dict(self.metadata)
        return new
