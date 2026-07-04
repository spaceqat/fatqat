"""ResourceLayout: the single source of truth for flat qubit/clbit indices."""

from __future__ import annotations

from collections.abc import Mapping

from .program import Program
from .registers import RegisterRef


class ResourceLayout:
    """Flat index mapping for a program's quantum and classical resources.

    Quantum registers are concatenated in program order, and classical registers
    are concatenated independently in program order.
    """

    def __init__(
        self,
        system_dims: tuple[int, ...],
        classical_dims: tuple[int, ...],
        q_offsets: Mapping[int, int],
        c_offsets: Mapping[int, int],
        n_clbits: int,
    ) -> None:
        """Create a resource layout from precomputed flat offsets.

        Args:
            system_dims: Per-subsystem Hilbert-space dimensions for quantum registers.
            classical_dims: Per-subsystem dimensions for classical registers.
            q_offsets: Mapping from `id(QuantumRegister)` to flat qubit offset.
            c_offsets: Mapping from `id(ClassicalRegister)` to flat clbit offset.
            n_clbits: Total number of classical bits.
        """
        self.system_dims: tuple[int, ...] = system_dims
        self.classical_dims: tuple[int, ...] = classical_dims
        self._q_offsets = dict(q_offsets)  # id(register) -> base flat index
        self._c_offsets = dict(c_offsets)
        self._n_clbits = n_clbits

    @property
    def n_subsystems(self) -> int:
        """Total number of subsystems in the layout."""
        return len(self.system_dims)

    @property
    def n_clbits(self) -> int:
        """Total number of classical bits in the layout."""
        return self._n_clbits

    def subsystem_index(self, ref: RegisterRef) -> int:
        """Return the flat subsystem index for a quantum register reference.

        Raises:
            KeyError: If the reference's register is not part of this layout.
        """
        try:
            base = self._q_offsets[id(ref.register)]
        except KeyError:
            raise KeyError("subsystem ref not part of this layout") from None
        return base + ref.index

    def clbit_index(self, ref: RegisterRef) -> int:
        """Return the flat classical-bit index for a classical register reference.

        Raises:
            KeyError: If the reference's register is not part of this layout.
        """
        try:
            base = self._c_offsets[id(ref.register)]
        except KeyError:
            raise KeyError("clbit ref not part of this layout") from None
        return base + ref.index

    @classmethod
    def from_program(cls, program: Program) -> "ResourceLayout":
        """Build a layout by flattening a program's registers in order."""
        q_offsets: dict[int, int] = {}
        system_dims: list[int] = []
        offset = 0
        for reg in program.qreg:
            q_offsets[id(reg)] = offset
            system_dims.extend(reg.dim for _ in range(reg.size))
            offset += reg.size

        c_offsets: dict[int, int] = {}
        classical_dims: list[int] = []
        coffset = 0
        for reg in program.creg:
            c_offsets[id(reg)] = coffset
            classical_dims.extend(reg.dim for _ in range(reg.size))
            coffset += reg.size

        return cls(
            system_dims=tuple(system_dims),
            classical_dims=tuple(classical_dims),
            q_offsets=q_offsets,
            c_offsets=c_offsets,
            n_clbits=coffset,
        )
