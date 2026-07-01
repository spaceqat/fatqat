"""Program container plus AppliedOperation / Measurement value objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, TypeVar

from .operations import Operation
from .registers import QuantumRegister, RegisterRef, ClassicalRegister

ConditionTerm = tuple[RegisterRef, int]
Condition = tuple[ConditionTerm, ...] | None
RegisterT = TypeVar("RegisterT", QuantumRegister, ClassicalRegister)


@dataclass(frozen=True)
class AppliedOperation:
    """An operation bound to resolved quantum register references.

    `Program.add` creates these objects after validating the operation and
    resolving integer operands or explicit `RegisterRef` objects.

    Attributes:
        operation: Operation instance to execute.
        targets: Quantum register references consumed by the operation.
        condition: Optional AND tuple of classical references and literal values.
    """

    operation: Operation
    targets: tuple[RegisterRef, ...]
    condition: Condition = None

    def __post_init__(self) -> None:
        if not isinstance(self.targets, tuple):
            raise TypeError("targets must be a tuple of RegisterRef")
        expected = self.operation.num_qubits
        if len(self.targets) != expected:
            raise ValueError(
                f"{self.operation.name} expects {expected} target(s), "
                f"got {len(self.targets)}"
            )
        seen: set[tuple[int, int]] = set()
        for t in self.targets:
            if not isinstance(t, RegisterRef):
                raise TypeError(f"target must be RegisterRef, got {type(t)!r}")
            if not isinstance(t.register, QuantumRegister):
                raise TypeError("operation targets must reference a QuantumRegister")
            key = (id(t.register), t.index)
            if key in seen:
                raise ValueError(
                    f"{self.operation.name}: target qubit {t!r} appears more than once"
                )
            seen.add(key)


@dataclass(frozen=True)
class Measurement:
    """A measurement from one quantum register reference into one classical slot.

    Measurements live in `Program.operations` alongside applied operations and
    preserve insertion order. Metadata is copied at construction time so later
    caller mutations do not alias the frozen value object.

    Attributes:
        qreg: Quantum register reference to measure.
        clreg: Classical register reference to write.
        metadata: User metadata copied into the measurement.
    """

    qreg: RegisterRef
    clreg: RegisterRef
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Copy metadata so a caller's later mutation can't reach into this frozen
        # value object.
        object.__setattr__(self, "metadata", dict(self.metadata))


class Program:
    """User-facing quantum program container.

    A program owns quantum and classical registers plus an ordered sequence of
    applied operations and measurements stored in `operations`. Operations are
    executed in insertion order. Integer operands are accepted only when there is
    exactly one register of the relevant kind; otherwise users must pass explicit
    `RegisterRef` objects such as `program.qreg[0][1]`.

    Examples:
        Build a two-qubit program, add gates, then measure both qubits:

        ```python
        import qnsim as qs

        program = qs.Program(2, 2)
        program.add(qs.ops.H, 0)
        program.add(qs.ops.CZ, (0, 1))
        program.add_measurement(0, 0)
        program.add_measurement(1, 1)
        ```
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
            clreg: Number of default classical bits, or explicit classical
                registers. A value of `0` creates no classical register.
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
        """Ordered operation and measurement steps as a read-only tuple view."""
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
                singleton values such as `ops.X`; parametric gates should be
                instantiated, such as `ops.RX(0.2)`.
            qreg: Target qubit operand, or tuple of target operands for
                multi-qubit gates (e.g. `(0, 1)` for `CZ`). Each operand may
                be an integer when unambiguous or an explicit `RegisterRef`.
            condition: Optional single condition `(clbit, value)` or sequence of
                conditions. Conditions are normalized to an AND tuple.

        Raises:
            TypeError: If `op` is not an `Operation`, if operands have the wrong
                register kind, or if integer operands are ambiguous.
            ValueError: If target arity is wrong, a target is repeated, a ref is
                foreign to the program, or `condition` is empty.
            IndexError: If an integer operand is outside the relevant register.

        Examples:
            Add fixed and parametric gates:

            ```python
            import qnsim as qs

            program = qs.Program(2)
            program.add(qs.ops.H, 0)
            program.add(qs.ops.CZ, (0, 1))
            program.add(qs.ops.RX(0.2), 0)
            ```
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
        return tuple(
            (self._resolve_conditional_slot(slot), int(literal))
            for slot, literal in terms
        )

    def _resolve_conditional_slot(self, slot: int | RegisterRef) -> RegisterRef:
        """Resolve a condition slot and require it to reference a clbit."""
        if isinstance(slot, RegisterRef):
            if not isinstance(slot.register, ClassicalRegister):
                raise TypeError("condition slot ref must reference a ClassicalRegister")
            if not any(slot.register is r for r in self.creg):
                raise ValueError("condition slot ref not in this program")
            return slot
        return self._resolve_clbit(slot)

    def add_measurement(
        self,
        qreg: int | RegisterRef,
        clreg: int | RegisterRef,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Append a measurement from a qubit into a classical bit.

        Args:
            qreg: Quantum operand to measure, as an integer or explicit
                `RegisterRef`.
            clreg: Classical operand to write, as an integer or explicit
                `RegisterRef`.
            metadata: Optional measurement metadata copied into the measurement.

        Raises:
            TypeError: If operands have the wrong register kind or integer
                operands are ambiguous.
            ValueError: If an explicit ref is foreign to the program.
            IndexError: If an integer operand is outside the relevant register.

        Examples:
            Add a terminal measurement:

            ```python
            import qnsim as qs

            program = qs.Program(1, 1)
            program.add(qs.ops.X, 0)
            program.add_measurement(0, 0)
            ```
        """
        q = self._resolve_qubit(qreg)
        c = self._resolve_clbit(clreg)
        self._operations.append(
            Measurement(qreg=q, clreg=c, metadata=dict(metadata) if metadata else {})
        )
        self._operations_view = None

    def copy(self) -> "Program":
        """Return an independent copy with its own operation list, register lists, and metadata."""
        new = Program.__new__(Program)
        new.qreg = tuple(self.qreg)
        new.creg = tuple(self.creg)
        new._operations = list(self._operations)
        new._operations_view = tuple(new._operations)
        new.metadata = dict(self.metadata)
        return new
