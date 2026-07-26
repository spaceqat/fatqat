"""Fake configurable-shape neutral-atom-grid backend for compiler prototyping.

Configurable `rows x cols` device (default 4x5), row-major backend site
labels, e.g. for the default shape:

.. code-block:: text

    0   1   2   3   4
    5   6   7   8   9
    10 11  12  13  14
    15 16  17  18  19

Native gate set is exactly `RX`, `RY`, `RZ` (single-qubit, any device label)
and `CZ` (nearest-neighbor edges only, both directions stored, using
*backend* site labels - not flat engine indices - as the device-operand
keys). This is a prototype execution target, not a realistic device model: no
routing, no timing, no reshape/transport, and ideal by default.

Every device site starts empty. A program's first instruction must be
`~fatqat.ops.LoadAtom(rows, cols)` (unconditional, sized to fit the device);
it marks the top-left `rows x cols` block of sites as loaded and appears at
most once per program. Any later gate or `~fatqat.ops.Reset` whose targets
are not all loaded is silently dropped - an empty site cannot hold a gate.
`~fatqat.ops.Measurement` is never filtered by load state: a site no
surviving gate ever touched stays in its initial `|0>`, so it reads `0`
deterministically under ideal execution, though a configured readout-error
model can still flip the reported classical bit like any other qubit. See
``docs/superpowers/specs/2026-07-22-fatqat-grid-register-resource-binding-and-fake-atom-grid-backend-design.md``
and
``docs/superpowers/specs/2026-07-24-fake-atom-grid-loading-and-fake-superconducting-grid-mapping-design.md``.

A program built against a `~fatqat.GridRegister` binds top-left: frontend
`(row, col)` maps to backend site `(row, col)`, i.e. device label
`row * backend_cols + col`. A plain scalar-only program with no
`GridRegister` binds identically to `~fatqat.backends.SCQubitIBMSimulator`/
`~fatqat.backends.SCQubitGoogleSimulator` (plain scalar/identity binding),
since for a program with no grid register, backend device label and flat
engine index coincide.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import operations as ops
from ..errors import BackendValidationError
from ..implementation import (
    ImplementationMap,
    default_matrix_implementation_map,
)
from ..noise import NoiseModel
from ..program import AppliedOperation, Program
from ..registers import (
    GridRegister,
    RegisterRef,
)
from ..resource_layout import ResourceLayout
from .backend_utils import _validate_grid_size
from .simulator_backend import SimulatorBackend

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..implementation import MatrixImplementation
    from ..operations import Operation
    from .backend_utils import _LoweringContext, _PlanFacts
    from .simulator_backend import ProgramInstruction
    from .steps import ResolvedStep

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
    `add`); `CZ` is legal only on nearest-neighbor grid edges, both
    directions, keyed by *backend* site labels (added with explicit
    `device_operands`, one call per edge). Every other operation family
    (including `CX`) has no entry and is therefore unsupported.
    """
    rows, cols = _validate_grid_size((rows, cols))
    defaults = default_matrix_implementation_map()
    rx_rule = defaults.implementation_for(ops.RX)
    ry_rule = defaults.implementation_for(ops.RY)
    rz_rule = defaults.implementation_for(ops.RZ)
    cz_rule = defaults.implementation_for(ops.CZ)

    m = ImplementationMap()
    m.add(ops.RX, rx_rule)
    m.add(ops.RY, ry_rule)
    m.add(ops.RZ, rz_rule)
    for edge in _nearest_neighbor_edges(rows, cols):
        m.add(ops.CZ, cz_rule, device_operands=edge)
    return m


class AtomGridBackend(SimulatorBackend):
    """Statevector backend constrained to a fake configurable-shape atom-grid target.

    A thin statevector-method `~fatqat.backends.SimulatorBackend`
    specialization: same execution engine, same `~fatqat.Result`/`~fatqat.Job`
    semantics. The differences are a configurable `rows x cols` device shape
    (default 4x5), a fixed native gate set
    (:py:class:`~fatqat.operations.RX`,
    :py:class:`~fatqat.operations.RY`, :py:class:`~fatqat.operations.RZ`, and
    nearest-neighbor :py:data:`~fatqat.operations.CZ`), grid-aware resource
    mapping: a program's sole :py:class:`~fatqat.GridRegister` (if any) binds
    top-left onto the device, with
    every other quantum-register shape (scalar-only, or a grid combined with
    any other register, or more than one grid register) either bound
    identically (scalar-only) or rejected, and an explicit atom-loading
    lifecycle driven by :py:class:`~fatqat.operations.LoadAtom` - see
    :py:meth:`_lower`.

    Example:
        This two-row, three-column circuit prepares a Hadamard on every site
        in the first row, then creates pairwise row-1-to-row-2 CNOTs. Neither
        ``H`` nor ``CX`` is native: the circuit emits ``H`` as ``RZ(pi)`` then
        ``RY(pi / 2)`` (up to global phase), and each CNOT as ``H(target)``,
        ``CZ(control, target)``, ``H(target)``.

        **Circuit construction and result check**

        .. doctest:: atom_grid_cx

           >>> import numpy as np
           >>> import fatqat as fq
           >>> atoms = fq.GridRegister(2, 3, name="atoms")
           >>> program = fq.Program([atoms], 6)
           >>> program.add(fq.ops.LoadAtom(2, 3))
           >>> def native_h(targets):
           ...     program.add(fq.ops.RZ(np.pi), targets)
           ...     program.add(fq.ops.RY(np.pi / 2), targets)
           >>> native_h(atoms.row(0))                 # H on every control
           >>> native_h(atoms.row(1))                 # H on every target
           >>> program.add(fq.ops.CZ, (atoms.row(0), atoms.row(1)))
           >>> native_h(atoms.row(1))                 # completes pairwise CX
           >>> program.measure_all()

           >>> backend = fq.backends.AtomGridBackend()  # default 4x5 device
           >>> counts = backend.run(
           ...     program, shots=1000, simulation_config={"seed": 1}
           ... ).result().get_counts()
           >>> all(bits[:3] == bits[3:] for bits in counts)
           True
    """

    def __init__(
        self,
        *,
        grid_size: tuple[int, int] = (DEFAULT_ROWS, DEFAULT_COLS),
        method: str = "statevector",
        runtime: str = "numpy",
        noise: NoiseModel | None = None,
    ) -> None:
        """Create a fake atom-grid backend of the given shape.

        Args:
            grid_size: Device shape as ``(rows, columns)``. Both values must
                be positive integers.
            method: State representation, exactly as on
                :py:class:`~fatqat.backends.SimulatorBackend`.
            runtime: Numeric execution runtime, exactly as on
                :py:class:`~fatqat.backends.SimulatorBackend`.
            noise: Optional `~fatqat.NoiseModel`, exactly as on
                `~fatqat.backends.SimulatorBackend`. `None` (the default)
                keeps the backend ideal.

        Raises:
            TypeError: If ``grid_size`` is not a two-item tuple of integers
                (bools rejected).
            ValueError: If ``grid_size`` does not contain exactly two values
                or either value is not positive.
        """
        self._rows, self._cols = _validate_grid_size(grid_size)
        super().__init__(
            method=method,
            runtime=runtime,
            implementation_map=fake_atom_grid_implementation_map(
                self._rows, self._cols
            ),
            noise=noise,
        )

    @property
    def implementation_map(self) -> ImplementationMap:
        """Return a copy of the compiler-facing device-aware implementation map.

        Examples:
            >>> import fatqat as fq
            >>> backend = fq.backends.AtomGridBackend()
            >>> impl_map = backend.implementation_map
            >>> sorted(op.name for op in impl_map.supported_operations())
            ['CZ', 'RX', 'RY', 'RZ']
            >>> impl_map.supports(fq.ops.CCX)
            False
        """
        return self._impl_map.copy()

    def _resolve_resource_layout(self, program: Program) -> ResourceLayout:
        """Reject any shape the fake device can't run, then map top-left.

        Applies equally to a scalar-only program with no `GridRegister`:
        total qubit count and per-subsystem dimension are checked regardless
        of register structure. A program's sole `GridRegister` (if any) then
        binds top-left onto the device: frontend `(row, col)` maps to device
        label `row * cols + col`, using the *backend*'s column count, not the
        grid's own. A scalar-only program (no `GridRegister`) delegates to
        the base class's generic declaration-order identity mapping.

        Raises:
            BackendValidationError: If the program declares more subsystems
                than `rows * cols`; any non-qubit-dimension (`dim != 2`)
                register; more than one `GridRegister`; a `GridRegister`
                combined with any other quantum register; or a `GridRegister`
                whose shape does not fit the device's, axis by axis.
        """
        n_subsystems = sum(register.size for register in program.qreg)
        capacity = self._rows * self._cols
        if n_subsystems > capacity:
            raise BackendValidationError(
                f"AtomGridBackend({self._rows}x{self._cols}) supports at "
                f"most {capacity} qubits, got {n_subsystems}"
            )
        dims = (register.dim for register in program.qreg for _ in range(register.size))
        if any(dim != 2 for dim in dims):
            raise BackendValidationError(
                "AtomGridBackend only supports qubit dimensions"
            )
        grid_registers = [r for r in program.qreg if isinstance(r, GridRegister)]
        if len(grid_registers) > 1:
            raise BackendValidationError(
                "AtomGridBackend accepts at most one GridRegister per "
                f"program, got {len(grid_registers)}"
            )
        if not grid_registers:
            return super()._resolve_resource_layout(program)

        grid = grid_registers[0]
        if len(program.qreg) != 1:
            raise BackendValidationError(
                "AtomGridBackend rejects a GridRegister combined with "
                "any other quantum register"
            )
        if grid.rows > self._rows or grid.cols > self._cols:
            raise BackendValidationError(
                f"grid register ({grid.rows}x{grid.cols}) does not fit the "
                f"backend's ({self._rows}x{self._cols}) device shape"
            )
        labels: dict[RegisterRef, int] = {}
        for index in range(grid.size):
            row, col = divmod(index, grid.cols)
            labels[grid[index]] = row * self._cols + col
        return ResourceLayout(labels)

    def _lower(
        self,
        operations: Sequence[ProgramInstruction],
        context: _LoweringContext,
    ) -> tuple[list[ResolvedStep], _PlanFacts]:
        """Apply this program's atom-loading lifecycle, then lower normally.

        Every device site starts empty. The program's first instruction must
        be `~fatqat.ops.LoadAtom` (unconditional, sized to fit this device);
        it marks the top-left `rows x cols` block of sites as loaded and is
        itself dropped before lowering, since it has no matrix. Any later
        `LoadAtom` is rejected - loading happens exactly once, up front.
        Every other gate or `Reset` whose targets are not all loaded is
        silently dropped (no-op): an empty site cannot hold a gate.
        `Measurement` always lowers normally; a site no surviving gate ever
        touched stays in its initial |0>, so measuring an unloaded site
        reads 0 deterministically under ideal execution - though a
        configured readout-error model can still flip the reported bit,
        exactly as for any other qubit.

        Raises:
            BackendValidationError: If the program's first instruction is
                not `LoadAtom`; if any later instruction is `LoadAtom`; if
                `LoadAtom` carries a condition; or if `LoadAtom`'s shape
                does not fit this backend's device.
        """
        resource_layout = context.resource_layout
        loaded: set[int] = set()
        realized: list[ProgramInstruction] = []
        for i, step in enumerate(operations):
            is_load = isinstance(step, AppliedOperation) and isinstance(
                step.operation, ops.LoadAtom
            )
            if i == 0:
                if not is_load:
                    raise BackendValidationError(
                        "AtomGridBackend requires the program's first "
                        "operation to be LoadAtom"
                    )
            elif is_load:
                raise BackendValidationError(
                    "AtomGridBackend accepts LoadAtom only as the "
                    "program's first operation"
                )
            if is_load:
                if step.condition is not None:
                    raise BackendValidationError("LoadAtom must be unconditional")
                load_rows, load_cols = step.operation.rows, step.operation.cols
                if load_rows > self._rows or load_cols > self._cols:
                    raise BackendValidationError(
                        f"LoadAtom({load_rows}x{load_cols}) does not fit "
                        f"the backend's ({self._rows}x{self._cols}) device "
                        "shape"
                    )
                loaded = {
                    r * self._cols + c
                    for r in range(load_rows)
                    for c in range(load_cols)
                }
                continue
            if isinstance(step, AppliedOperation) and any(
                resource_layout.device_label(t) not in loaded for t in step.targets
            ):
                continue
            realized.append(step)
        return super()._lower(tuple(realized), context)
