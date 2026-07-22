"""Fake configurable-shape neutral-atom-grid backend for compiler prototyping.

Configurable `rows x cols` device (default 4x5), row-major backend site
labels, e.g. for the default shape:

.. code-block:: text

    0   1   2   3   4
    5   6   7   8   9
    10 11  12  13  14
    15 16  17  18  19

Native gate set is exactly `RX`, `RY`, `RZ` (single-qubit, any device label)
and `CX`/`CZ` (nearest-neighbor edges only, both directions stored, using
*backend* site labels - not flat engine indices - as the device-operand
keys). This is a prototype execution target, not a realistic device model: no
routing, no timing, no reshape/transport, and ideal by default. See
``docs/superpowers/specs/2026-07-22-fatqat-grid-register-resource-binding-and-fake-atom-grid-backend-design.md``.

A program built against a `~fatqat.GridRegister` binds top-left: frontend
`(row, col)` maps to backend site `(row, col)`, i.e. device label
`row * backend_cols + col`. A plain scalar-only program with no
`GridRegister` binds identically to `FakeSuperconducting4x4Backend` (plain
scalar/identity binding), since for a program with no grid register, backend
device label and flat engine index coincide.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import operations as ops
from ..errors import BackendValidationError
from .._engine_allocation import _EngineAllocation
from ..implementation import (
    ImplementationMap,
    default_matrix_implementation_map,
)
from ..noise import NoiseModel
from ..program import Program
from ..registers import (
    GridRegister,
    RegisterRef,
)
from .simulator_backend import SimulatorBackend

if TYPE_CHECKING:
    from ..implementation import MatrixImplementation
    from ..operations import Operation

DEFAULT_ROWS = 4
DEFAULT_COLS = 5


def _nearest_neighbor_edges(rows: int, cols: int) -> tuple[tuple[int, int], ...]:
    """Return directed nearest-neighbor edges for a row-major `rows x cols` grid.

    Both directions of every edge are included (e.g. `(0, 1)` and `(1, 0)`),
    matching `fake_superconducting._nearest_neighbor_edges`'s convention.
    Edge endpoints are backend site labels (`row * cols + col`).
    """
    edges: list[tuple[int, int]] = []
    for row in range(rows):
        for col in range(cols):
            q = row * cols + col
            if col + 1 < cols:
                right = q + 1
                edges.extend(((q, right), (right, q)))
            if row + 1 < rows:
                down = q + cols
                edges.extend(((q, down), (down, q)))
    return tuple(edges)


def fake_atom_grid_implementation_map(rows: int, cols: int) -> ImplementationMap:
    """Build the native gate map for a `rows x cols` fake atom-grid backend.

    `RX`, `RY`, `RZ` are legal on any device label (registered uniformly via
    `add`); `CX` and `CZ` are legal only on nearest-neighbor grid edges, both
    directions, keyed by *backend* site labels (added with explicit
    `device_operands`, one call per edge). Every other operation family has
    no entry and is therefore unsupported.
    """
    defaults = default_matrix_implementation_map()
    rx_rule = defaults.implementation_for(ops.RX)
    ry_rule = defaults.implementation_for(ops.RY)
    rz_rule = defaults.implementation_for(ops.RZ)
    cx_rule = defaults.implementation_for(ops.CX)
    cz_rule = defaults.implementation_for(ops.CZ)

    m = ImplementationMap()
    m.add(ops.RX, rx_rule)
    m.add(ops.RY, ry_rule)
    m.add(ops.RZ, rz_rule)
    for edge in _nearest_neighbor_edges(rows, cols):
        m.add(ops.CX, cx_rule, device_operands=edge)
        m.add(ops.CZ, cz_rule, device_operands=edge)
    return m


class FakeAtomGridBackend(SimulatorBackend):
    """Statevector backend constrained to a fake configurable-shape atom-grid target.

    A thin statevector-method `~fatqat.backends.SimulatorBackend`
    specialization: same execution engine, same `~fatqat.Result`/`~fatqat.Job`
    semantics. The differences are a configurable `rows x cols` device shape
    (default 4x5), a fixed native gate set (`RX`, `RY`, `RZ`, nearest-neighbor
    `CX`/`CZ`), and grid-aware resource mapping: a program's sole
    `~fatqat.GridRegister` (if any) binds top-left onto the device, with
    every other quantum-register shape (scalar-only, or a grid combined with
    any other register, or more than one grid register) either bound
    identically (scalar-only) or rejected.
    """

    def __init__(
        self,
        rows: int = DEFAULT_ROWS,
        cols: int = DEFAULT_COLS,
        options: dict[str, Any] | None = None,
        noise: NoiseModel | None = None,
    ) -> None:
        """Create a fake atom-grid backend of the given shape.

        Args:
            rows: Number of device rows. Must be a positive integer.
            cols: Number of device columns. Must be a positive integer.
            options: Same execution-strategy options as
                `~fatqat.backends.SimulatorBackend` (`max_workers`,
                `parallel_mode`). The implementation map is fixed to
                `fake_atom_grid_implementation_map(rows, cols)` and cannot be
                overridden.
            noise: Optional `~fatqat.NoiseModel`, exactly as on
                `~fatqat.backends.SimulatorBackend`. `None` (the default)
                keeps the backend ideal.

        Raises:
            TypeError: If `rows` or `cols` is not an `int` (bools rejected).
            ValueError: If `rows` or `cols` is not positive.
        """
        if not isinstance(rows, int) or isinstance(rows, bool):
            raise TypeError(f"rows must be int, got {type(rows)!r}")
        if rows <= 0:
            raise ValueError(f"rows must be positive, got {rows}")
        if not isinstance(cols, int) or isinstance(cols, bool):
            raise TypeError(f"cols must be int, got {type(cols)!r}")
        if cols <= 0:
            raise ValueError(f"cols must be positive, got {cols}")
        self._rows = rows
        self._cols = cols
        super().__init__(
            method="statevector",
            options=options,
            implementation_map=fake_atom_grid_implementation_map(rows, cols),
            noise=noise,
        )

    @property
    def implementation_map(self) -> ImplementationMap:
        """Return a copy of the compiler-facing device-aware implementation map.

        Examples:
            >>> import fatqat as fq
            >>> backend = fq.backends.FakeAtomGridBackend()
            >>> impl_map = backend.implementation_map
            >>> sorted(op.name for op in impl_map.supported_operations())
            ['CX', 'CZ', 'RX', 'RY', 'RZ']
            >>> impl_map.supports(fq.ops.CCX)
            False
        """
        return self._impl_map.copy()

    def resolve_layout(self, program: Program) -> _EngineAllocation:
        """Build the flat layout, then reject any shape the fake device can't run.

        Applies equally to a scalar-only program with no `GridRegister`:
        total qubit count and per-subsystem dimension are checked regardless
        of register structure.

        Raises:
            BackendValidationError: If the program declares more subsystems
                than `rows * cols`, or any non-qubit-dimension (`dim != 2`)
                register.
        """
        layout = super().resolve_layout(program)
        capacity = self._rows * self._cols
        if layout.n_subsystems > capacity:
            raise BackendValidationError(
                f"FakeAtomGridBackend({self._rows}x{self._cols}) supports at "
                f"most {capacity} qubits, got {layout.n_subsystems}"
            )
        if any(dim != 2 for dim in layout.system_dims):
            raise BackendValidationError(
                "FakeAtomGridBackend only supports qubit dimensions"
            )
        grid_registers = [r for r in program.qreg if isinstance(r, GridRegister)]
        if len(grid_registers) > 1:
            raise BackendValidationError(
                "FakeAtomGridBackend accepts at most one GridRegister per "
                f"program, got {len(grid_registers)}"
            )
        if grid_registers:
            grid = grid_registers[0]
            if len(program.qreg) != 1:
                raise BackendValidationError(
                    "FakeAtomGridBackend rejects a GridRegister combined with "
                    "any other quantum register"
                )
            if grid.rows > self._rows or grid.cols > self._cols:
                raise BackendValidationError(
                    f"grid register ({grid.rows}x{grid.cols}) does not fit the "
                    f"backend's ({self._rows}x{self._cols}) device shape"
                )
        return layout

    def _device_label_for(
        self, ref: RegisterRef, flat_layout: _EngineAllocation
    ) -> int:
        """Map a scalar ref to its row-major fake-device site label."""
        if not isinstance(ref.register, GridRegister):
            return flat_layout.subsystem_index(ref)
        row, col = divmod(ref.index, ref.register.cols)
        return row * self._cols + col
