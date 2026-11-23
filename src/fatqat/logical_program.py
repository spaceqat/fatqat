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
    """Describe a device-independent circuit for simulation or compilation.

    Construct it with quantum and classical register counts or explicit
    register collections, as for `Program`. Use `add` for built-in logical
    operations and `Program.measure` or `Program.measure_all` for measurements.
    These methods modify the program in place. Copying or binding parameters
    returns another `LogicalProgram`.

    Direct simulation supports classical conditions. Compilation requires
    numeric parameters, terminal measurements, and no classical conditions.
    Each compilation target also restricts the supported gates.

    Attributes:
        IR_ID: Source boundary name for compiler pipelines.

    Examples:
        Prepare a Bell state and measure both qubits:

        >>> import fatqat as fq
        >>> import fatqat.operations as ops
        >>> circuit = fq.LogicalProgram(2, 2)
        >>> circuit.add(ops.H, 0)
        >>> circuit.add(ops.CX, (0, 1))
        >>> circuit.measure_all()
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

        Bind by object identity. Two parameters with the same name are
        distinct. The original program is unchanged. A binding may be partial
        or empty, but compilation and numeric simulation require concrete values.

        Args:
            values: A mapping from parameters used directly as operation
                arguments to numeric values:

                - `Parameter`: A built-in integer or float, or a NumPy
                  integer or floating scalar. Booleans are not accepted.
                - `ParameterVector`: A one-dimensional NumPy array or a
                  non-string, non-bytes, non-mapping iterable of those
                  scalars. Supply one value per element in index order. Every
                  element must be used, and no element may also be bound
                  separately.

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

        Use `Program.measure` to add measurements. A valid instruction may
        still be unsupported by a particular compilation target.

        Args:
            op: A built-in logical gate, including qudit gates, or a reset
                or barrier. Device operations and custom operations are
                rejected.
            targets: One operand, or a tuple or list in operand order.

                - ``int``: Zero-based index when there is exactly one
                  quantum register.
                - `RegisterRef`: A slot from a quantum register in this program.
                - `RegisterView`: A selection for a gate that supports views.
                  Provide one compatible view per operand, all of the same
                  selection kind and length. Do not mix views and scalar
                  targets, or overlap views from the same register.

            condition: ``None`` (default), one ``(classical_slot, value)`` pair,
                or a non-empty tuple or list of pairs that must all match.
                Use a zero-based index if there is only one classical register,
                or a classical `RegisterRef` from this program. Values are
                Python integers in the slot's range; booleans are accepted.
                Conditions can be used in direct simulation.

        Returns:
            None.

        Raises:
            ValueError: If the operation is unsupported, targets have the
                wrong arity or are incompatible, a ref is outside the program,
                or a condition has an invalid shape or value.
            TypeError: If an argument has an unsupported type.
            IndexError: If an integer operand is outside its register.
        """
        if isinstance(op, Operation) and type(op) not in _LOGICAL_OPERATION_TYPES:
            raise ValueError(f"{op.name} is not a logical operation")
        super().add(op, targets, condition=condition)

    def _new_copy(self) -> LogicalProgram:
        return LogicalProgram.__new__(LogicalProgram)
