"""Device-independent program authoring for compilation and simulation."""

from __future__ import annotations

from typing import ClassVar, Mapping, cast

from . import operations as ops
from .operations import Operation
from .parameters import Parameter, ParameterVector
from .program import ConditionInput, Program
from .registers import RegisterRef, RegisterView

__all__ = ["LogicalProgram"]


_LOGICAL_OPERATION_TYPES = frozenset(
    {
        type(ops.I),
        type(ops.H),
        type(ops.S),
        type(ops.Sdg),
        type(ops.SX),
        type(ops.T),
        type(ops.Tdg),
        type(ops.X),
        type(ops.Y),
        type(ops.Z),
        type(ops.CX),
        type(ops.CZ),
        type(ops.Swap),
        type(ops.CY),
        type(ops.CS),
        type(ops.iSwap),
        type(ops.CCX),
        type(ops.CSwap),
        ops.RX,
        ops.RY,
        ops.RZ,
        ops.Phase,
        ops.U,
        ops.U1,
        ops.U2,
        ops.U3,
        ops.CPhase,
        type(ops.Reset),
        type(ops.Barrier),
        ops.Shift,
        ops.Clock,
        type(ops.Sum),
        ops.SwapLevels,
        type(ops.Fourier),
        type(ops.InverseFourier),
        ops.SubspaceRX,
        ops.SubspaceRY,
        ops.SubspaceRZ,
        ops.CClock,
    }
)


class LogicalProgram(Program):
    """Build a device-independent circuit for compilation or simulation.

    Use Program's constructor, add(), and measurement interface. Authoring
    methods mutate in place and return None. Built-in circuit gates, including
    qudit gates, reset, and barriers are accepted; device operations and custom
    Operation subclasses are rejected. Use measure() for measurements.

    Conditions remain valid for direct simulation. Static compiler and target
    restrictions apply when compiling. The copy() and assign_parameters()
    methods preserve LogicalProgram.

    Attributes:
        IR_ID: Compiler identity for this editable source representation.
    """

    IR_ID: ClassVar[str] = "gate.logical.source.v1"

    def copy(self) -> LogicalProgram:
        """Return an independently editable LogicalProgram copy.

        Later add() and measure() calls and top-level metadata changes are
        independent. Values nested inside metadata remain shared.

        Returns:
            A LogicalProgram with the same instructions.
        """
        return cast(LogicalProgram, super().copy())

    def assign_parameters(
        self,
        values: Mapping[Parameter | ParameterVector, object],
    ) -> LogicalProgram:
        """Return a LogicalProgram with selected parameters replaced.

        Matching uses object identity rather than names. Binding may be
        partial or empty and never mutates the template. Scalar and vector
        values follow Program.assign_parameters() validation, including vector
        length and duplicate-assignment checks.

        Args:
            values: Parameter keys map to real numeric scalars. ParameterVector
                keys map to matching-length one-dimensional numeric iterables.

        Returns:
            A LogicalProgram containing the selected numeric values.

        Raises:
            TypeError: If the mapping, a key, a value container, or a scalar
                has the wrong type.
            ValueError: If a parameter is absent, a vector binding is invalid,
                or the same parameter is assigned more than once.
        """
        return cast(LogicalProgram, super().assign_parameters(values))

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
        """Append a built-in device-independent operation in place.

        Arguments and target/condition validation follow Program.add.
        Device operations and custom Operation subclasses raise ValueError
        before the program changes. Measurements use measure() instead.

        Args:
            op: A built-in device-independent Operation instance.
            targets: Scalar register operands or compatible register views,
                using the same forms as Program.add.
            condition: Optional classical condition, default None; uses the
                same register/literal pairs and conjunction rules as Program.add.

        Returns:
            None.

        Raises:
            ValueError: If op is not a built-in logical operation, or inherited
                target or condition validation fails.
            TypeError: If an argument has an unsupported type.
            IndexError: If an integer operand is outside its register.
        """
        if isinstance(op, Operation) and type(op) not in _LOGICAL_OPERATION_TYPES:
            raise ValueError(f"{op.name} is not a logical operation")
        super().add(op, targets, condition=condition)

    def _new_copy(self) -> LogicalProgram:
        return LogicalProgram.__new__(LogicalProgram)
