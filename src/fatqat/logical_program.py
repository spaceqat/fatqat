"""Editable, hardware-independent input for the FatQat compiler."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, Self

from . import operations as ops
from .operations import Operation
from .parameters import Parameter, ParameterVector
from .program import ConditionInput, Program
from .registers import RegisterRef, RegisterView

__all__ = ["LogicalProgram"]

ScalarQubit = int | RegisterRef
Qubit = ScalarQubit | RegisterView
Clbit = int | RegisterRef

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


class LogicalProgram(Program):  # pylint: disable=too-many-public-methods
    """Build a device-independent circuit for compilation or simulation.

    This specialized `Program` keeps the same registers and instruction
    storage while rejecting device operations such as atom placement,
    pairing, and direct pulse controls. Classical conditions remain valid for
    direct simulation; the current static compiler reports them as unsupported
    when the circuit is frozen.
    """

    IR_ID: ClassVar[str] = "gate.logical.source.v1"

    def add(  # pylint: disable=arguments-differ
        self,
        operation: Operation,
        *qubits: Qubit | tuple[Qubit, ...],
        condition: ConditionInput = None,
    ) -> Self:
        """Append one device-independent operation.

        Built-in circuit gates, reset, and barriers are accepted. Device
        operations and custom Operation subclasses are rejected. Conditions
        use the same classical-register syntax as Program.add.

        Args:
            operation: Built-in device-independent operation to append.
            *qubits: One or more scalar targets, or compatible register views.
            condition: Optional classical condition.

        Returns:
            This program, for chained construction.

        Raises:
            TypeError: If operation, a target, or a condition has an invalid
                type.
            ValueError: If the operation is not logical, or inherited target
                or condition validation fails.
            IndexError: If an integer operand is outside its register.
        """
        targets = qubits[0] if len(qubits) == 1 else qubits
        if not isinstance(operation, Operation):
            # Preserve Program's detailed error for invalid values, including
            # the dedicated Measurement guidance.
            super().add(operation, targets, condition=condition)
        if type(operation) not in _LOGICAL_OPERATION_TYPES:
            name = getattr(operation, "name", type(operation).__name__)
            raise ValueError(f"{name} is not a logical operation")
        super().add(operation, targets, condition=condition)
        return self

    def _new_copy(self) -> LogicalProgram:
        return LogicalProgram.__new__(LogicalProgram)

    def copy(self) -> Self:
        """Return an independently editable logical-program copy."""
        return super().copy()

    def assign_parameters(
        self,
        values: Mapping[Parameter | ParameterVector, object],
    ) -> Self:
        """Return a logical-program copy with selected parameters bound."""
        return super().assign_parameters(values)

    def i(self, qubit: Qubit) -> Self:
        return self.add(ops.I, qubit)

    def h(self, qubit: Qubit) -> Self:
        return self.add(ops.H, qubit)

    def x(self, qubit: Qubit) -> Self:
        return self.add(ops.X, qubit)

    def y(self, qubit: Qubit) -> Self:
        return self.add(ops.Y, qubit)

    def z(self, qubit: Qubit) -> Self:
        return self.add(ops.Z, qubit)

    def s(self, qubit: Qubit) -> Self:
        return self.add(ops.S, qubit)

    def sdg(self, qubit: Qubit) -> Self:
        return self.add(ops.Sdg, qubit)

    def t(self, qubit: Qubit) -> Self:
        return self.add(ops.T, qubit)

    def tdg(self, qubit: Qubit) -> Self:
        return self.add(ops.Tdg, qubit)

    def sx(self, qubit: Qubit) -> Self:
        return self.add(ops.SX, qubit)

    def rx(self, theta: float | Parameter, qubit: Qubit) -> Self:
        return self.add(ops.RX(theta), qubit)

    def ry(self, theta: float | Parameter, qubit: Qubit) -> Self:
        return self.add(ops.RY(theta), qubit)

    def rz(self, theta: float | Parameter, qubit: Qubit) -> Self:
        return self.add(ops.RZ(theta), qubit)

    def phase(self, theta: float | Parameter, qubit: Qubit) -> Self:
        return self.add(ops.Phase(theta), qubit)

    def cx(self, control: Qubit, target: Qubit) -> Self:
        return self.add(ops.CX, control, target)

    def cz(self, first: Qubit, second: Qubit) -> Self:
        return self.add(ops.CZ, first, second)

    def swap(self, first: Qubit, second: Qubit) -> Self:
        return self.add(ops.Swap, first, second)

    def reset(self, qubit: Qubit) -> Self:
        return self.add(ops.Reset, qubit)

    def measure(
        self,
        targets: ScalarQubit | tuple[ScalarQubit, ...],
        outputs: Clbit | tuple[Clbit, ...],
    ) -> Self:
        """Measure qubits into classical slots and return this program."""
        super().measure(targets, outputs)
        return self

    def measure_all(self) -> Self:
        """Measure every qubit into its corresponding classical slot."""
        super().measure_all()
        return self
