"""ResourceLayout: the single source of truth for flat qubit/clbit indices."""

from __future__ import annotations

from .registers import RegisterRef


class ResourceLayout:
    def __init__(self, system_dims, q_offsets, c_offsets, n_clbits):
        self.system_dims: tuple[int, ...] = system_dims
        self._q_offsets = q_offsets  # id(register) -> base flat index
        self._c_offsets = c_offsets
        self._n_clbits = n_clbits

    @property
    def n_qubits(self) -> int:
        return len(self.system_dims)

    @property
    def n_clbits(self) -> int:
        return self._n_clbits

    def qubit_index(self, ref: RegisterRef) -> int:
        try:
            base = self._q_offsets[id(ref.register)]
        except KeyError:
            raise KeyError("qubit ref not part of this layout") from None
        return base + ref.index

    def clbit_index(self, ref: RegisterRef) -> int:
        try:
            base = self._c_offsets[id(ref.register)]
        except KeyError:
            raise KeyError("clbit ref not part of this layout") from None
        return base + ref.index

    @classmethod
    def from_program(cls, program) -> "ResourceLayout":
        q_offsets: dict[int, int] = {}
        offset = 0
        for reg in program.qreg:
            q_offsets[id(reg)] = offset
            offset += reg.size
        n_qubits = offset

        c_offsets: dict[int, int] = {}
        coffset = 0
        for reg in program.creg:
            c_offsets[id(reg)] = coffset
            coffset += reg.size

        return cls(
            system_dims=(2,) * n_qubits,
            q_offsets=q_offsets,
            c_offsets=c_offsets,
            n_clbits=coffset,
        )
