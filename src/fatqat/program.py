"""Program container and its private instruction records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TypeVar

from ._parameter_binding import (
    _normalize_parameter_mapping,
    _replace_parameterized_instructions,
)
from .operations import Measurement, Operation
from .parameters import Parameter, ParameterVector
from .registers import (
    QuantumRegister,
    RegisterRef,
    RegisterView,
    ClassicalRegister,
    _validate_view_pair,
)

__all__ = ["Program"]

ConditionTerm = tuple[RegisterRef, int]
Condition = tuple[ConditionTerm, ...] | None
ConditionOperand = tuple[int | RegisterRef, int]
ConditionInput = (
    ConditionOperand | tuple[ConditionOperand, ...] | list[ConditionOperand] | None
)
RegisterT = TypeVar("RegisterT", QuantumRegister, ClassicalRegister)

# A frontend target expression: either a resolved scalar reference, or (for
# the view-capable operations named on `Operation._accepts_views`) a
# structured `RegisterView` selecting multiple members of one `GridRegister`.
# See `_AppliedOperation.targets` and `Program.add`.
QuantumTarget = RegisterRef | RegisterView


@dataclass(frozen=True)
class _AppliedOperation:
    """Store one normalized operation instruction for internal lowering.

    ``Program.add()`` is the supported construction boundary. It resolves
    targets, verifies register kind and ownership, normalizes the condition,
    and supplies immutable containers. Grouped-view expansion is the only
    other internal construction path and preserves those invariants.

    This record checks operation arity, duplicate scalar targets,
    operation-specific scalar constraints, and grouped-view pairing. It trusts
    its callers for target element types, register ownership, container shape,
    and condition normalization. Per-member validation of a grouped view is
    deferred until expansion produces scalar instructions.
    """

    operation: Operation
    targets: tuple[QuantumTarget, ...]
    condition: Condition = None

    def __post_init__(self) -> None:
        expected = self.operation.num_subsystems
        if expected is None:
            minimum = self.operation.min_targets
            if len(self.targets) < minimum:
                if minimum == 1:
                    raise ValueError(
                        f"{self.operation.name} expects at least one target"
                    )
                raise ValueError(
                    f"{self.operation.name} expects at least {minimum} targets"
                )
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
            # Per-member scalar validation (validate_targets()) still needs
            # concrete refs and is deferred to binding/expansion. Pairing
            # legality (arity 2 only) does not - it is a fact about the two
            # views themselves, decidable from their selectors alone - so it
            # is checked here, not left to whichever backend/strategy later
            # chooses how to lower the group.
            if len(self.targets) == 2:
                first, second = self.targets
                first_is_view = isinstance(first, RegisterView)
                second_is_view = isinstance(second, RegisterView)
                if first_is_view != second_is_view:
                    raise ValueError(
                        f"{self.operation.name} mixes a scalar target with a "
                        "view target; a two-target gate needs both operands "
                        "scalar or both views"
                    )
                if first_is_view and second_is_view:
                    _validate_view_pair(first, second, op_name=self.operation.name)
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
    """Build an ordered quantum program.

    ``add()``, ``measure()``, and ``measure_all()`` append instructions in
    place. Register collections are stored as tuples and should be treated as
    construction-time state; replacing those public attributes is unsupported.
    ``metadata`` remains a mutable dictionary for user annotations. Calls that
    mutate one program are not synchronized; use ``copy()`` when separate
    builders need to branch from the same prefix.

    Program construction validates frontend structure, not backend support.
    A selected backend may reject an otherwise well-formed operation,
    condition, register dimension, or instruction sequence when the program
    is run.

    Attributes:
        quantum_registers: Tuple of the supplied quantum register objects.
            Replacing the attribute after construction is unsupported.
        classical_registers: Tuple of the supplied classical register objects.
            Replacing the attribute after construction is unsupported.
        metadata: Mutable, shallow-copied dictionary of user metadata. FATQAT
            does not define or interpret its keys.

    Examples:
        Build a two-qubit program, add gates, then measure both qubits:

        >>> import fatqat as fq
        >>> import fatqat.operations as ops
        >>> program = fq.Program(2, 2)
        >>> program.add(ops.H, 0)
        >>> program.add(ops.CZ, (0, 1))
        >>> program.measure(0, 0)
        >>> program.measure(1, 1)
    """

    def __init__(
        self,
        quantum_registers: int | list[QuantumRegister] | tuple[QuantumRegister, ...],
        classical_registers: (
            int | list[ClassicalRegister] | tuple[ClassicalRegister, ...]
        ) = 0,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Create a program from register counts or explicit register collections.

        Args:
            quantum_registers: Non-negative exact integer subsystem count, or
                a list or tuple of ``QuantumRegister`` objects. A positive
                count creates one dimension-2 register named ``"q"``; zero
                creates no quantum register. The outer collection is copied,
                but the register objects are retained.
            classical_registers: Non-negative exact integer slot count, or a
                list or tuple of ``ClassicalRegister`` objects. A positive
                count creates one dimension-2 register named ``"c"``. The
                default ``0`` creates no classical register. The outer
                collection is copied, but the register objects are retained.
            metadata: Optional mapping of user-defined string keys to arbitrary
                values. There are no predefined keys or per-key defaults, and
                key types are not checked at runtime. The top-level mapping is
                shallow-copied into a mutable dictionary; nested values are not
                copied. ``None`` creates an empty dictionary.

        Raises:
            TypeError: If a count is not an exact integer (booleans are not
                counts), if a register specification is not an integer or
                list, if a list contains the wrong register type, or if a
                truthy ``metadata`` value cannot be converted to a dictionary.
            ValueError: If an integer register count is negative or a truthy
                ``metadata`` iterable cannot be converted to dictionary pairs.
        """
        self.quantum_registers: tuple[QuantumRegister, ...] = tuple(
            self._coerce_registers(quantum_registers, QuantumRegister, "q")
        )
        self.classical_registers: tuple[ClassicalRegister, ...] = tuple(
            self._coerce_registers(classical_registers, ClassicalRegister, "c")
        )
        self._operations: list[_AppliedOperation | Measurement] = []
        self._operations_view: tuple[_AppliedOperation | Measurement, ...] | None = ()
        self.metadata: dict[str, Any] = dict(metadata) if metadata else {}

    @property
    def _instructions(self) -> tuple[_AppliedOperation | Measurement, ...]:
        """Return the cached instruction snapshot in insertion order.

        A previously returned tuple remains unchanged after later mutation.
        """
        if self._operations_view is None:
            self._operations_view = tuple(self._operations)
        return self._operations_view

    def draw(self, renderer: str = "matplotlib", **kwargs: Any) -> Any:
        """Render this program as a circuit diagram.

        Drawing uses one wire per quantum or classical slot. Register
        dimensions are not depicted. Built-in gates use native QuTiP-QIP
        symbols where available; other operations are labeled boxes. Direct
        ``PulseOperation`` controls cannot be represented by the circuit
        renderer.

        Args:
            renderer: ``"matplotlib"`` (default) for a matplotlib ``Figure``,
                ``"text"`` for a returned terminal-diagram string, or another
                renderer name supported by QuTiP-QIP.
            **kwargs: Renderer options forwarded to QuTiP-QIP. Matplotlib also
                accepts ``ax`` to draw on an existing axis.

        Returns:
            A matplotlib ``Figure`` for ``"matplotlib"``, a string for
            ``"text"``, or the selected QuTiP-QIP renderer's return value.

        Raises:
            ImportError: If QuTiP-QIP is unavailable.
            UnsupportedOperationError: If the program contains a
                ``PulseOperation``.
        """
        from .draw import _draw_program

        return _draw_program(self, renderer, **kwargs)

    @staticmethod
    def _coerce_registers(
        spec: int | list[RegisterT] | tuple[RegisterT, ...],
        cls: type[RegisterT],
        default_name: str,
    ) -> list[RegisterT]:
        """Normalize an integer count or explicit register collection to a list."""
        if type(spec) is int:
            if spec < 0:
                raise ValueError(f"register count must be >= 0, got {spec}")
            return [cls(spec, name=default_name)] if spec > 0 else []
        if not isinstance(spec, (list, tuple)):
            raise TypeError(
                f"registers must be an int count or a list or tuple of "
                f"{cls.__name__}, "
                f"got {type(spec).__name__!r}"
            )
        for r in spec:
            if not isinstance(r, cls):
                raise TypeError(
                    f"register collection must contain {cls.__name__} instances, "
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
                "(e.g. quantum_registers[0] or classical_registers[0]) instead"
            )
        # Bounds and negative-index checks are delegated to Register.__getitem__,
        # which raises IndexError.
        return regs[0][operand]

    def _resolve_quantum_ref(self, operand: int | RegisterRef) -> RegisterRef:
        return self._resolve_ref(
            operand, self.quantum_registers, QuantumRegister, "quantum register"
        )

    def _resolve_quantum_target(
        self, operand: int | RegisterRef | RegisterView
    ) -> QuantumTarget:
        """Resolve one ``Program.add()`` operand, accepting a ``RegisterView``
        in addition to everything ``_resolve_quantum_ref`` accepts.

        This is deliberately a separate path from ``_resolve_quantum_ref``:
        that helper is also used by ``measure``/``measure_all``,
        which must keep rejecting views (see spec section 5.1).
        """
        if isinstance(operand, RegisterView):
            if not any(operand.register is r for r in self.quantum_registers):
                raise ValueError(
                    "view's register does not belong to this program's quantum registers"
                )
            return operand
        return self._resolve_quantum_ref(operand)

    def _resolve_classical_ref(self, operand: int | RegisterRef) -> RegisterRef:
        return self._resolve_ref(
            operand, self.classical_registers, ClassicalRegister, "classical register"
        )

    def add(
        self,
        op: Operation,
        targets: (
            int
            | RegisterRef
            | RegisterView
            | tuple[int | RegisterRef | RegisterView, ...]
        ) = (),
        *,
        condition: ConditionInput = None,
    ) -> None:
        """Validate and append one operation in place.

        The appended instruction retains ``op`` and the resolved target
        objects. This method checks program-level structure immediately;
        device and backend capability checks occur when the program is run.
        For a ``PulseOperation``, omit ``targets``: its ``PulseControl``
        channels address physical resources directly, and the selected pulse
        emulator resolves and validates those addresses during program
        preparation.

        Args:
            op: Operation instance to append. Fixed gates are available as
                singleton values such as ``ops.X``; parametric gates should be
                instantiated, such as ``ops.RX(0.2)``.
            targets: One target expression, or a tuple in operation-operand
                order. A bare target must be an exact built-in ``int`` (not a
                boolean, NumPy integer, or integer subclass) and is accepted
                only when the program has exactly one quantum register. An
                explicit ``RegisterRef`` must
                belong to this program's quantum registers. ``RX``, ``RY``,
                and ``RZ`` also accept one ``RegisterView``; ``CX`` and ``CZ``
                accept a pair of compatible views. Two-target view application
                cannot mix a scalar with a view, and its views must use the
                same selector kind and cardinality without overlapping on one
                register. Omit this argument when adding a ``PulseOperation``;
                each control's channel identifies the physical resource or
                resources it drives.
            condition: ``None`` (default) for an unconditional operation,
                one ``(classical_slot, literal)`` pair, or a non-empty tuple or
                list of such pairs. Every pair is required (logical AND). A
                slot may be an exact built-in ``int`` only when exactly one
                classical register exists, or an explicit classical
                ``RegisterRef`` owned by the program. A literal must be a
                Python ``int`` in ``[0, slot_dimension)``; booleans and integer
                subclasses are accepted and normalized with ``int()``, while
                NumPy integer scalars are rejected.

        Returns:
            ``None``.

        Raises:
            TypeError: If ``op`` is not an ``Operation``, if a target or
                condition has an unsupported type or register kind, if an
                integer operand is ambiguous, or if a condition literal is not
                an integer.
            ValueError: If target arity is wrong, a target is repeated, a ref
                or view is foreign to the program, ``op`` does not accept a
                ``RegisterView`` target, a view pair is incompatible, an
                operation-specific target constraint fails, a condition is
                empty or a term does not contain exactly two items, or a
                condition literal is out of range.
            IndexError: If an integer operand is outside the relevant register.

        Examples:
            Add fixed and parametric gates:

            >>> import fatqat as fq
            >>> import fatqat.operations as ops
            >>> program = fq.Program(2)
            >>> program.add(ops.H, 0)
            >>> program.add(ops.CZ, (0, 1))
            >>> program.add(ops.RX(0.2), 0)
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
            _AppliedOperation(operation=op, targets=target_refs, condition=normalized)
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

    def measure(
        self,
        targets: int | RegisterRef | tuple[int | RegisterRef, ...],
        outputs: int | RegisterRef | tuple[int | RegisterRef, ...],
    ) -> None:
        """Append a computational-basis measurement in place.

        A grouped call creates one ``Measurement`` record. Targets and outputs
        are paired positionally, and each pair must have the same register
        dimension. Repeated targets and outputs are accepted; built-in
        backends process pairs in order, so a repeated output keeps the later
        pair's reported value. Measurement does not accept ``RegisterView``
        objects.

        Args:
            targets: Quantum operand(s) to measure, as an exact built-in
                ``int``, explicit ``RegisterRef``, or non-empty tuple of those
                operands. Booleans, NumPy integers, and integer subclasses are
                not integer operands. Bare integers require exactly one
                quantum register.
            outputs: Classical operand(s) to write, in the same forms, with a
                non-empty tuple matching ``targets`` in count. Bare integers
                require exactly one classical register.

        Returns:
            ``None``.

        Raises:
            TypeError: If operands have the wrong register kind or integer
                operands are ambiguous, or if a view or unsupported container
                is passed.
            ValueError: If ``targets``/``outputs`` have mismatched or zero
                length, an explicit ref is foreign to the program, or a
                quantum/classical pair has different dimensions.
            IndexError: If an integer operand is outside the relevant register.

        Examples:
            Add a terminal measurement:

            >>> import fatqat as fq
            >>> import fatqat.operations as ops
            >>> program = fq.Program(1, 1)
            >>> program.add(ops.X, 0)
            >>> program.measure(0, 0)

            Add a grouped measurement:

            >>> program2 = fq.Program(2, 2)
            >>> program2.add(ops.H, 0)
            >>> program2.add(ops.CZ, (0, 1))
            >>> program2.measure((0, 1), (0, 1))
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
        """Append one measurement pairing all quantum and classical slots.

        Registers and their members are flattened in declaration order. The
        method appends one grouped ``Measurement``; it does not replace earlier
        measurements.

        Returns:
            ``None``.

        Raises:
            ValueError: If either flattened side is empty, their counts differ,
                or any position pairs registers with different dimensions.
        """
        targets = tuple(
            ref
            for reg in self.quantum_registers
            for ref in (reg[i] for i in range(reg.size))
        )
        outputs = tuple(
            ref
            for reg in self.classical_registers
            for ref in (reg[i] for i in range(reg.size))
        )
        # Equal-count and non-empty invariants are enforced once in
        # Measurement.__post_init__, reached through measure.
        self.measure(targets, outputs)

    def copy(self) -> "Program":
        """Return a shallow structural copy for independent program mutation.

        The copy shares register objects and existing instruction record
        objects with the original. It owns a new instruction list and a new
        top-level metadata dictionary, so later ``add()``/``measure()`` calls
        and top-level metadata changes are independent. Values nested inside
        metadata remain shared.

        Returns:
            An independently mutable program branch.
        """
        return self._copy_with_operations(self._operations)

    def _copy_with_operations(
        self,
        operations: (
            tuple[_AppliedOperation | Measurement, ...]
            | list[_AppliedOperation | Measurement]
        ),
    ) -> "Program":
        """Copy program structure and metadata around trusted instructions.

        Callers already own instruction validation. The fresh list and
        metadata dictionary keep the returned program independent without
        rebuilding registers through the public constructor.
        """
        new = Program.__new__(Program)
        new.quantum_registers = tuple(self.quantum_registers)
        new.classical_registers = tuple(self.classical_registers)
        new._operations = list(operations)
        new._operations_view = tuple(new._operations)
        new.metadata = dict(self.metadata)
        return new

    def assign_parameters(
        self,
        values: Mapping[Parameter | ParameterVector, object],
    ) -> "Program":
        """Return a copy with selected parameter objects replaced by numbers.

        Matching is by object identity, not by name. Binding may be partial or
        empty and never mutates the template. Only ``Parameter`` values stored
        directly in dataclass operation fields are discovered; a parameter
        nested inside another container is not a binding target. Any remaining
        parameters stay in the returned program and are rejected later by
        numeric execution or export APIs. Structural discovery does not widen
        an operation field's declared value contract: reconstruction or
        backend lowering may reject a bound value that is invalid for that
        field.

        Args:
            values: Mapping with these accepted key/value forms:

                - ``Parameter`` key (built-in ``int`` or ``float``, or NumPy
                  integer or floating scalar): Replaces every direct occurrence
                  of that same object. Booleans, strings, complex numbers, and
                  other numeric classes are not accepted.
                - ``ParameterVector`` key (one-dimensional NumPy array or a
                  non-string, non-bytes, non-mapping iterable of the scalar
                  types above): The iterable is consumed once in its own
                  iteration order and paired with vector index order. Its
                  length must match the vector, and every vector element must
                  occur in a direct operation field. Bind individual elements
                  instead when only part of a vector is present.

                String keys and positional assignments are not accepted. A
                vector and one of its elements cannot both be assigned.

        Returns:
            A shallow structural copy containing the selected numeric values.
            It has independent instruction storage and a copied top-level
            metadata dictionary, while retaining the original registers.

        Raises:
            TypeError: If the mapping, a key, a value container, or a scalar
                has the wrong type.
            ValueError: If a parameter identity is absent, a vector is empty or
                not fully present, an element is assigned twice, or a vector
                value has the wrong rank or length. Errors raised while
                reconstructing a custom dataclass operation propagate
                unchanged.

        Examples:
            Bind a vector at once while leaving the template unchanged:

            >>> import fatqat as fq
            >>> import fatqat.operations as ops
            >>> angles = fq.ParameterVector("angles", 2)
            >>> program = fq.Program(2)
            >>> program.add(ops.RX(angles[0]), 0)
            >>> program.add(ops.RY(angles[1]), 1)
            >>> bound = program.assign_parameters({angles: [0.1, 0.2]})
            >>> bound is program
            False
        """
        normalized = _normalize_parameter_mapping(self._instructions, values)
        operations = _replace_parameterized_instructions(self._instructions, normalized)
        return self._copy_with_operations(operations)

    def _assign_normalized_parameters(
        self,
        values: Mapping[Parameter, object],
    ) -> "Program":
        """Bind one trusted sweep row without repeating public validation.

        ``_normalize_parameter_batch`` guarantees complete identity-keyed
        scalar rows before this method is called. Keeping this seam separate
        prevents every sweep point from rechecking the same batch contract.
        """
        operations = _replace_parameterized_instructions(self._instructions, values)
        return self._copy_with_operations(operations)
