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

from typing import Any

from .. import operations as ops
from ..errors import BackendValidationError
from ..flat_layout import FlatResourceLayout
from ..implementation import (
    ImplementationMap,
    MatrixImplementation,
    default_matrix_implementation_map,
)
from ..noise import NoiseModel
from ..operations import Operation
from ..program import Program
from ..registers import (
    AllSelector,
    BlockSelector,
    ColumnSelector,
    GridRegister,
    RegisterRef,
    RegisterView,
    RowSelector,
)
from .resource_binding import BoundResource, ResourceBinding, _scalar_identity_binder
from .simulator_backend import SimulatorBackend

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


def _require_rule(
    implementation_map: ImplementationMap,
    op: Operation,
) -> MatrixImplementation:
    rule = implementation_map.implementation_for(op)
    if rule is None:
        raise RuntimeError(f"default matrix implementation missing for {op!r}")
    return rule


def fake_atom_grid_implementation_map(rows: int, cols: int) -> ImplementationMap:
    """Build the native gate map for a `rows x cols` fake atom-grid backend.

    `RX`, `RY`, `RZ` are legal on any device label (registered uniformly via
    `add`); `CX` and `CZ` are legal only on nearest-neighbor grid edges, both
    directions, keyed by *backend* site labels (added with explicit
    `device_operands`, one call per edge). Every other operation family has
    no entry and is therefore unsupported.
    """
    defaults = default_matrix_implementation_map()
    rx_rule = _require_rule(defaults, ops.RX)
    ry_rule = _require_rule(defaults, ops.RY)
    rz_rule = _require_rule(defaults, ops.RZ)
    cx_rule = _require_rule(defaults, ops.CX)
    cz_rule = _require_rule(defaults, ops.CZ)

    m = ImplementationMap()
    m.add(ops.RX, rx_rule)
    m.add(ops.RY, ry_rule)
    m.add(ops.RZ, rz_rule)
    for edge in _nearest_neighbor_edges(rows, cols):
        m.add(ops.CX, cx_rule, device_operands=edge)
        m.add(ops.CZ, cz_rule, device_operands=edge)
    return m


class GridBinding:
    """Grid-specific binder: resolves targets from one bound `GridRegister`.

    Closes over the bound `GridRegister` instance and the backend's column
    count. Declines (returns `None`) any target that is not a `RegisterRef`
    or `RegisterView` whose `.register` is (by identity) the bound
    `GridRegister`. Top-left placement: frontend `(row, col)` maps to backend
    device label `row * backend_cols + col`, using the *backend*'s column
    count, not the frontend grid's. Engine index always comes from the run's
    `FlatResourceLayout`, never hand-computed.
    """

    def __init__(self, grid_register: GridRegister, backend_cols: int) -> None:
        """Store the bound grid register and the backend's column count.

        Args:
            grid_register: The `GridRegister` this binder resolves targets
                for; any target from a different register is declined.
            backend_cols: The *backend*'s column count, used to compute
                device labels (`row * backend_cols + col`).
        """
        self._grid_register = grid_register
        self._backend_cols = backend_cols

    def __call__(
        self, target: RegisterRef | RegisterView, flat_layout: FlatResourceLayout
    ) -> BoundResource | tuple[BoundResource, ...] | None:
        """Resolve one target expression from the bound grid register.

        Declines any target whose `.register` is not (by identity) the bound
        `GridRegister`. A scalar `RegisterRef` resolves to a single
        `BoundResource`; a `RegisterView` resolves to a tuple of
        `BoundResource`, one per member, in the view's deterministic order.
        """
        if isinstance(target, RegisterRef):
            if target.register is not self._grid_register:
                return None
            return self._bind_scalar(target, flat_layout)
        if isinstance(target, RegisterView):
            if target.register is not self._grid_register:
                return None
            return tuple(
                self._bind_scalar(ref, flat_layout) for ref in self._members(target)
            )
        return None

    def _bind_scalar(
        self, ref: RegisterRef, flat_layout: FlatResourceLayout
    ) -> BoundResource:
        frontend_row, frontend_col = divmod(ref.index, self._grid_register.cols)
        device_label = frontend_row * self._backend_cols + frontend_col
        engine_index = flat_layout.subsystem_index(ref)
        return BoundResource(
            ref=ref, engine_index=engine_index, device_label=device_label
        )

    def _members(self, view: RegisterView) -> list[RegisterRef]:
        """Enumerate a view's member refs in `RegisterView`'s documented order.

        Row-major for `AllSelector`/`BlockSelector`, increasing-column for
        `RowSelector`, increasing-row for `ColumnSelector`.
        """
        reg = self._grid_register
        sel = view.selector
        if isinstance(sel, AllSelector):
            coords = [(r, c) for r in range(reg.rows) for c in range(reg.cols)]
        elif isinstance(sel, RowSelector):
            coords = [(sel.row, c) for c in range(reg.cols)]
        elif isinstance(sel, ColumnSelector):
            coords = [(r, sel.col) for r in range(reg.rows)]
        elif isinstance(sel, BlockSelector):
            (r0, r1), (c0, c1) = sel.rows, sel.cols
            coords = [(r, c) for r in range(r0, r1) for c in range(c0, c1)]
        else:  # pragma: no cover - exhaustive over the Selector union
            raise AssertionError(f"unhandled selector {sel!r}")
        return [reg[r * reg.cols + c] for r, c in coords]


class FakeAtomGridBackend(SimulatorBackend):
    """Statevector backend constrained to a fake configurable-shape atom-grid target.

    A thin statevector-method `~fatqat.backends.SimulatorBackend`
    specialization: same execution engine, same `~fatqat.Result`/`~fatqat.Job`
    semantics. The differences are a configurable `rows x cols` device shape
    (default 4x5), a fixed native gate set (`RX`, `RY`, `RZ`, nearest-neighbor
    `CX`/`CZ`), and grid-aware resource binding: a program's sole
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

    def resolve_layout(self, program: Program) -> FlatResourceLayout:
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
        return layout

    def _create_resource_binding(
        self, program: Program, flat_layout: FlatResourceLayout
    ) -> ResourceBinding:
        """Build this run's resource binding: grid-aware, or plain scalar/identity.

        Finds every `GridRegister` in `program.qreg`. No grid register means
        a plain scalar-only program, unchanged from the base backend's
        identity binding (device label == flat engine index already, for a
        program with no grid register). Exactly one grid register must be
        the program's sole quantum register and must fit the device
        (`grid.rows <= self._rows and grid.cols <= self._cols`, checked
        per-axis); its `GridBinding` is tried first, with the scalar/identity
        binder installed as a defensive fallback that never fires once the
        sole-register rule holds.

        Raises:
            BackendValidationError: If more than one `GridRegister` is
                present, a `GridRegister` is combined with any other quantum
                register, or the grid register does not fit the device shape.
        """
        grid_registers = [r for r in program.qreg if isinstance(r, GridRegister)]
        if not grid_registers:
            return super()._create_resource_binding(program, flat_layout)
        if len(grid_registers) > 1:
            raise BackendValidationError(
                "FakeAtomGridBackend accepts at most one GridRegister per "
                f"program, got {len(grid_registers)}"
            )
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
        return ResourceBinding([GridBinding(grid, self._cols), _scalar_identity_binder])
