"""Program container plus the AppliedOperation value object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TypeVar

from .operations import Measurement, Operation
from .registers import QuantumRegister, RegisterRef, ClassicalRegister

ConditionTerm = tuple[RegisterRef, int]
Condition = tuple[ConditionTerm, ...] | None
RegisterT = TypeVar("RegisterT", QuantumRegister, ClassicalRegister)


@dataclass(frozen=True)
class AppliedOperation:
    """An operation bound to resolved quantum register references.

    ``Program.add`` creates these objects after validating the operation and
    resolving integer operands or explicit ``RegisterRef`` objects.

    ``__post_init__`` intentionally does not re-validate ``targets``' element
    types or tuple-ness: ``Program.add()`` already guarantees well-formed
    ``RegisterRef`` tuples via ``_resolve_qubit``, and duplicating that check
    here would just be the same validation twice for the path essentially all
    callers use. Constructing this class directly (bypassing ``Program``)
    skips that guarantee - malformed input (wrong register kind, a list
    instead of a tuple) will not raise here, and will instead surface later
    as a less specific error during backend lowering, or make the instance
    unhashable. This is a deliberate no-duplicate-validation tradeoff, not an
    oversight.

    Attributes:
        operation: Operation instance to execute.
        targets: Quantum register references consumed by the operation.
        condition: Optional AND tuple of classical references and literal values.
    """

    operation: Operation
    targets: tuple[RegisterRef, ...]
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

        .. code-block:: python

            import fatqat as fc

            program = fc.Program(2, 2)
            program.add(fc.ops.H, 0)
            program.add(fc.ops.CZ, (0, 1))
            program.add_measurement(0, 0)
            program.add_measurement(1, 1)
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
        self.creg: tuple[ClassicalRegister, ...] = tuple(
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
            raise TypeError(f"operand must be int or RegisterRef, got {type(operand)!r}")
        if len(regs) != 1:
            raise TypeError(
                "integer operands are only allowed when there is exactly one "
                "register of the relevant kind; pass an explicit RegisterRef "
                "(e.g. qreg[0] or creg[0]) instead"
            )
        # Bounds and negative-index checks are delegated to Register.__getitem__,
        # which raises IndexError.
        return regs[0][operand]

    def _resolve_qubit(self, operand: int | RegisterRef) -> RegisterRef:
        return self._resolve_ref(operand, self.qreg, QuantumRegister, "quantum register")

    def _resolve_clbit(self, operand: int | RegisterRef) -> RegisterRef:
        return self._resolve_ref(
            operand, self.creg, ClassicalRegister, "classical register"
        )

    def add(
        self,
        op: Operation,
        qreg: int | RegisterRef | tuple[int | RegisterRef, ...],
        *,
        condition=None,
    ) -> None:
        """Append an operation to the program in place.

        Args:
            op: Operation instance to append. Fixed gates are available as
                singleton values such as ``ops.X``; parametric gates should be
                instantiated, such as ``ops.RX(0.2)``.
            qreg: Target qubit operand, or tuple of target operands for
                multi-qubit gates (e.g. ``(0, 1)`` for ``CZ``). Each operand
                may be an integer when unambiguous or an explicit
                ``RegisterRef``.
            condition: Optional single condition ``(clbit, value)`` or
                sequence of conditions. Conditions are normalized to an AND
                tuple.

        Raises:
            TypeError: If ``op`` is not an ``Operation``, if operands have the
                wrong register kind, or if integer operands are ambiguous.
            ValueError: If target arity is wrong, a target is repeated, a ref
                is foreign to the program, or ``condition`` is empty.
            IndexError: If an integer operand is outside the relevant register.

        Examples:
            Add fixed and parametric gates:

            .. code-block:: python

                import fatqat as fc

                program = fc.Program(2)
                program.add(fc.ops.H, 0)
                program.add(fc.ops.CZ, (0, 1))
                program.add(fc.ops.RX(0.2), 0)
        """
        if not isinstance(op, Operation):
            raise TypeError(
                f"op must be an Operation instance, got {type(op)!r} "
                "(did you forget to call a parametric gate, e.g. ops.RX(0.2)?)"
            )
        operands = qreg if isinstance(qreg, tuple) else (qreg,)
        targets = tuple(self._resolve_qubit(o) for o in operands)
        normalized = self._normalize_condition(condition)
        self._operations.append(
            AppliedOperation(operation=op, targets=targets, condition=normalized)
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
            ref = self._resolve_clbit(slot)
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
        qreg: int | RegisterRef | tuple[int | RegisterRef, ...],
        clreg: int | RegisterRef | tuple[int | RegisterRef, ...],
    ) -> None:
        """Append a measurement from one or more qubits into classical bits.

        Args:
            qreg: Quantum operand(s) to measure, as an integer, explicit
                ``RegisterRef``, or tuple of operands for a grouped
                measurement.
            clreg: Classical operand(s) to write, as an integer, explicit
                ``RegisterRef``, or tuple of operands matching ``qreg`` in
                count.

        Raises:
            TypeError: If operands have the wrong register kind or integer
                operands are ambiguous.
            ValueError: If ``qreg``/``clreg`` have mismatched or zero length,
                or an explicit ref is foreign to the program.
            IndexError: If an integer operand is outside the relevant register.

        Examples:
            Add a terminal measurement:

            .. code-block:: python

                import fatqat as fc

                program = fc.Program(1, 1)
                program.add(fc.ops.X, 0)
                program.add_measurement(0, 0)

            Add a grouped measurement:

            .. code-block:: python

                program.add_measurement((0, 1), (0, 1))
        """
        q_operands = qreg if isinstance(qreg, tuple) else (qreg,)
        c_operands = clreg if isinstance(clreg, tuple) else (clreg,)
        qs = tuple(self._resolve_qubit(q) for q in q_operands)
        cs = tuple(self._resolve_clbit(c) for c in c_operands)
        # Length, non-empty, and per-pair dim invariants are enforced once in
        # Measurement.__post_init__.
        self._operations.append(Measurement(qreg=qs, clreg=cs))
        self._operations_view = None

    def measure_all(self) -> None:
        """Measure every qubit into every clbit in flat declaration order.

        Raises:
            ValueError: If the program has a different number of quantum bits
                than classical bits, or has no registers of either kind.
        """
        qubits = tuple(ref for reg in self.qreg for ref in (reg[i] for i in range(reg.size)))
        clbits = tuple(ref for reg in self.creg for ref in (reg[i] for i in range(reg.size)))
        # Equal-count and non-empty invariants are enforced once in
        # Measurement.__post_init__, reached through add_measurement.
        self.add_measurement(qubits, clbits)

    def copy(self) -> "Program":
        """Return an independent copy with private operation storage and copied metadata."""
        new = Program.__new__(Program)
        new.qreg = tuple(self.qreg)
        new.creg = tuple(self.creg)
        new._operations = list(self._operations)
        new._operations_view = tuple(new._operations)
        new.metadata = dict(self.metadata)
        return new
