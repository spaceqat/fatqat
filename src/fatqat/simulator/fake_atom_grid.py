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
`~fatqat.operations.LoadAtoms(rows, cols)` (unconditional, sized to fit the device);
it marks the top-left `rows x cols` block of sites as loaded and appears at
most once per program. Any later gate or `~fatqat.operations.Reset` whose targets
are not all loaded is silently dropped - an empty site cannot hold a gate.
`~fatqat.operations.Measurement` is never filtered by load state: a site no
surviving gate ever touched stays in its initial `|0>`, so it reads `0`
deterministically under ideal execution, though a configured readout-error
model can still flip the reported classical bit like any other qubit. See
``docs/superpowers/specs/2026-07-22-fatqat-grid-register-resource-binding-and-fake-atom-grid-backend-design.md``
and
``docs/superpowers/specs/2026-07-24-fake-atom-grid-loading-and-fake-superconducting-grid-mapping-design.md``.

A program built against a `~fatqat.GridRegister` binds top-left: frontend
`(row, col)` maps to backend site `(row, col)`, i.e. device label
`row * backend_cols + col`. A plain scalar-only program with no
`GridRegister` binds identically to `~fatqat.simulator.SCQubitIBMSimulator`/
`~fatqat.simulator.SCQubitGoogleSimulator` (plain scalar/identity binding),
since for a program with no grid register, backend device label and flat
engine index coincide.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, cast

from .. import operations as ops
from ..errors import BackendValidationError
from ..implementation import (
    MatrixImplementationMap,
    default_matrix_implementation_map,
)
from ..noise import NoiseModel
from ..program import AppliedOperation, Program
from ..registers import (
    GridRegister,
    RegisterRef,
)
from ..resource_layout import DeviceOperand, ResourceLayout
from .._index_allocation import _EngineAllocation
from .._backends.backend_utils import (
    _PlanFacts,
    _validate_grid_size,
)
from .._backends.steps import (
    LossStep,
    OccupancyInitStep,
    RefillStep,
)
from .planning import _lower_channels
from .simulator import Simulator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..implementation import MatrixImplementation
    from ..operations import Operation
    from ..result import _ResultConfig
    from .._backends.backend_utils import _LoweringContext
    from .simulator import ProgramInstruction
    from .._backends.steps import ResolvedStep

DEFAULT_ROWS = 4
DEFAULT_COLS = 5


@dataclass(frozen=True)
class _AtomGridPlanFacts(_PlanFacts):
    """Plan facts owned by the atom-grid occupancy lifecycle."""

    has_loss: bool = False
    has_refill: bool = False

    @classmethod
    def from_common(
        cls,
        common: _PlanFacts,
        *,
        has_loss: bool,
        has_refill: bool,
    ) -> _AtomGridPlanFacts:
        """Extend common facts without manually copying their fields."""
        common_values = {
            field.name: getattr(common, field.name) for field in fields(_PlanFacts)
        }
        return cls(
            **common_values,
            has_loss=has_loss,
            has_refill=has_refill,
        )

    @property
    def has_atom_lifecycle(self) -> bool:
        """Whether execution carries occupancy state outside the quantum state."""
        return self.has_loss or self.has_refill


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


def fake_atom_grid_implementation_map(rows: int, cols: int) -> MatrixImplementationMap:
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

    m = MatrixImplementationMap()
    m.add(ops.RX, rx_rule)
    m.add(ops.RY, ry_rule)
    m.add(ops.RZ, rz_rule)
    for edge in _nearest_neighbor_edges(rows, cols):
        m.add(ops.CZ, cz_rule, device_operands=edge)
    return m


class AtomGridSimulator(Simulator):
    """Statevector backend constrained to a fake configurable-shape atom-grid target.

    A thin statevector-method `~fatqat.simulator.Simulator`
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
    lifecycle driven by :py:class:`~fatqat.operations.LoadAtoms` - see
    :py:meth:`_lower`.

    Atom lifecycle:
        Beyond loading, three atom effects are available. Attach
        :py:class:`~fatqat.noise.Loss` to a gate to eject atoms per
        shot (a lost atom reads the erasure digit ``2``, distinct from a real
        ``|0>``); use :py:class:`~fatqat.operations.Rearrange` to move atoms to
        new sites mid-circuit so a two-qubit gate becomes legal on a pair that
        started non-adjacent; use :py:data:`~fatqat.operations.Refill` to
        reload emptied sites. Imperfect loading efficiency is expressed by
        attaching ``Loss`` to ``Refill``. Only this backend models atom
        loss; a generic backend rejects ``Loss`` via
        :py:meth:`validate_noise` rather than ignoring it.

        .. doctest:: atom_grid_loss

           >>> import numpy as np
           >>> import fatqat as fq
           >>> import fatqat.operations as op
           >>> noise = fq.NoiseModel()
           >>> noise.add(fq.noise.Loss(p=1.0), operation=op.RX)
           >>> program = fq.Program(1, 1)
           >>> program.add(op.LoadAtoms(1, 1))
           >>> program.add(op.RX(np.pi), 0)
           >>> program.measure(0, 0)
           >>> backend = fq.simulator.AtomGridSimulator(grid_size=(1, 1), noise=noise)
           >>> backend.run(
           ...     program, shots=10, simulation_config={"seed": 0}
           ... ).result().get_counts()
           {'2': 10}

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
           >>> import fatqat.operations as op
           >>> atoms = fq.GridRegister(2, 3, name="atoms")
           >>> program = fq.Program([atoms], 6)
           >>> program.add(op.LoadAtoms(2, 3))
           >>> def native_h(targets):
           ...     program.add(op.RZ(np.pi), targets)
           ...     program.add(op.RY(np.pi / 2), targets)
           >>> native_h(atoms.row(0))                 # H on every control
           >>> native_h(atoms.row(1))                 # H on every target
           >>> program.add(op.CZ, (atoms.row(0), atoms.row(1)))
           >>> native_h(atoms.row(1))                 # completes pairwise CX
           >>> program.measure_all()

           >>> backend = fq.simulator.AtomGridSimulator()  # default 4x5 device
           >>> counts = backend.run(
           ...     program, shots=1000, simulation_config={"seed": 1}
           ... ).result().get_counts()
           >>> all(bits[:3] == bits[3:] for bits in counts)
           True
    """

    _supports_loss = True

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
                :py:class:`~fatqat.simulator.Simulator`.
            runtime: Numeric execution runtime, exactly as on
                :py:class:`~fatqat.simulator.Simulator`.
            noise: Optional `~fatqat.NoiseModel`, exactly as on
                `~fatqat.simulator.Simulator`. `None` (the default)
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
    def implementation_map(self) -> MatrixImplementationMap:
        """Return a copy of the compiler-facing device-aware implementation map.

        Examples:
            >>> import fatqat as fq
            >>> import fatqat.operations as op
            >>> backend = fq.simulator.AtomGridSimulator()
            >>> impl_map = backend.implementation_map
            >>> sorted(op.name for op in impl_map.supported_operations())
            ['CZ', 'RX', 'RY', 'RZ']
            >>> impl_map.supports(op.CCX)
            False
        """
        return self._impl_map.copy()

    def _resolve_resource_layout(
        self,
        program: Program,
        supplied_layout: ResourceLayout | None = None,
    ) -> ResourceLayout:
        if supplied_layout is not None:
            raise BackendValidationError(
                "AtomGridSimulator does not accept a supplied resource layout"
            )
        return super()._resolve_resource_layout(program)

    def _legal_device_operands(
        self, program: Program, resource_layout: ResourceLayout
    ) -> frozenset[DeviceOperand]:
        return frozenset(range(self._rows * self._cols))

    def _physical_dimension(
        self, device_operand: DeviceOperand, resource_layout: ResourceLayout
    ) -> int:
        return 2

    def _default_resource_layout(self, program: Program) -> ResourceLayout:
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
        n_subsystems = sum(register.size for register in program.quantum_registers)
        capacity = self._rows * self._cols
        if n_subsystems > capacity:
            raise BackendValidationError(
                f"AtomGridSimulator({self._rows}x{self._cols}) supports at "
                f"most {capacity} qubits, got {n_subsystems}"
            )
        dims = (
            register.dim
            for register in program.quantum_registers
            for _ in range(register.size)
        )
        if any(dim != 2 for dim in dims):
            raise BackendValidationError(
                "AtomGridSimulator only supports qubit dimensions"
            )
        grid_registers = [
            r for r in program.quantum_registers if isinstance(r, GridRegister)
        ]
        if len(grid_registers) > 1:
            raise BackendValidationError(
                "AtomGridSimulator accepts at most one GridRegister per "
                f"program, got {len(grid_registers)}"
            )
        if not grid_registers:
            return super()._default_resource_layout(program)

        grid = grid_registers[0]
        if len(program.quantum_registers) != 1:
            raise BackendValidationError(
                "AtomGridSimulator rejects a GridRegister combined with "
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

    def _lower_with_context(
        self, operations: Sequence[ProgramInstruction], context: _LoweringContext
    ) -> tuple[list[ResolvedStep], _LoweringContext]:
        """Apply this program's atom lifecycle, then lower normally.

        Every device site starts empty. The program's first instruction must
        be `~fatqat.operations.LoadAtoms` (unconditional, sized to fit this
        device); it marks the top-left `rows x cols` block of sites as loaded
        and is itself dropped before lowering, since it has no matrix. Any
        later `LoadAtoms` is rejected - loading happens exactly once, up front.

        Occupancy is tracked by ref (an atom keeps its slot when moved), so a
        gate or `Reset` is statically dropped only when a target can never hold
        an atom: never loaded AND never named in any `Refill` (the narrowed
        static drop). Everything else survives to the engine's per-shot
        occupancy guard. `Rearrange` and `Refill` are themselves never dropped:
        `Rearrange` relabels an operand's position even when empty (M-B7), and
        `Refill` may fill a never-loaded site (M-C4).

        `Rearrange` changes only device-site labels, not engine indices, so it
        emits no execution step (M-B2): the plan is lowered in segments split
        at each `Rearrange`, each under the layout current at that point, so a
        gate's legality follows the atoms' current positions (M-B1). A
        `Rearrange` still emits any channel noise attached to it (transport
        cost) for the moved atoms.

        `Measurement` always lowers normally; a site no surviving gate ever
        touched stays in its initial |0>, so measuring an unloaded site reads 0
        deterministically under ideal execution - though a configured
        readout-error model can still flip the reported bit, exactly as for any
        other qubit.

        Raises:
            BackendValidationError: If the program's first instruction is not
                `LoadAtoms`; if a later instruction is `LoadAtoms`; if
                `LoadAtoms` or `Rearrange` carries a condition; if a shape or a
                rearrange destination site does not fit the device; or if a
                rearrange result is not injective.
        """
        resource_layout = context.resource_layout
        occupied: set[RegisterRef] = set()
        realized: list[ProgramInstruction] = []
        refill_targets = {
            t
            for step in operations
            if isinstance(step, AppliedOperation)
            and isinstance(step.operation, ops.RefillGate)
            for t in step.targets
        }
        for i, step in enumerate(operations):
            is_load = isinstance(step, AppliedOperation) and isinstance(
                step.operation, ops.LoadAtoms
            )
            if i == 0:
                if not is_load:
                    raise BackendValidationError(
                        "AtomGridSimulator requires the program's first "
                        "operation to be LoadAtoms"
                    )
            elif is_load:
                raise BackendValidationError(
                    "AtomGridSimulator accepts LoadAtoms only as the "
                    "program's first operation"
                )
            if is_load:
                if step.condition is not None:
                    raise BackendValidationError("LoadAtoms must be unconditional")
                load_rows, load_cols = step.operation.rows, step.operation.cols
                if load_rows > self._rows or load_cols > self._cols:
                    raise BackendValidationError(
                        f"LoadAtoms({load_rows}x{load_cols}) does not fit "
                        f"the backend's ({self._rows}x{self._cols}) device "
                        "shape"
                    )
                load_sites = {
                    r * self._cols + c
                    for r in range(load_rows)
                    for c in range(load_cols)
                }
                occupied = {
                    ref
                    for ref in resource_layout.refs
                    if resource_layout.device_label(ref) in load_sites
                }
                continue
            # Rearrange and Refill are exempt from the load-state drop; a gate
            # is dropped only when a target can never hold an atom: never
            # loaded AND never named in any Refill.
            if (
                isinstance(step, AppliedOperation)
                and not isinstance(step.operation, (ops.Rearrange, ops.RefillGate))
                and any(
                    t not in occupied and t not in refill_targets for t in step.targets
                )
            ):
                continue
            realized.append(step)

        current_layout = resource_layout
        current_allocation = context.engine_allocation
        plan: list[ResolvedStep] = []
        segment: list[ProgramInstruction] = []
        for step in realized:
            if isinstance(step, AppliedOperation) and isinstance(
                step.operation, ops.Rearrange
            ):
                seg_plan = super()._lower(
                    tuple(segment),
                    replace(
                        context,
                        resource_layout=current_layout,
                        engine_allocation=current_allocation,
                    ),
                )
                plan.extend(seg_plan)
                segment = []
                next_layout = self._apply_rearrange(current_layout, step)
                next_allocation = self._rebind_engine_allocation(
                    current_layout,
                    current_allocation,
                    next_layout,
                )
                plan.extend(
                    _lower_channels(
                        type(step.operation),
                        step.targets,
                        None,
                        current_layout,
                        current_allocation,
                        self._noise_model,
                        self._channel_map,
                    )
                )
                current_layout = next_layout
                current_allocation = next_allocation
                continue
            segment.append(step)
        seg_plan = super()._lower(
            tuple(segment),
            replace(
                context,
                resource_layout=current_layout,
                engine_allocation=current_allocation,
            ),
        )
        plan.extend(seg_plan)

        if any(isinstance(step, (LossStep, RefillStep)) for step in plan):
            occupied_indices = tuple(context.engine_index(ref) for ref in occupied)
            plan.insert(0, OccupancyInitStep(occupied_indices=occupied_indices))
        return plan, replace(
            context,
            resource_layout=current_layout,
            engine_allocation=current_allocation,
        )

    @staticmethod
    def _rebind_engine_allocation(
        current_layout: ResourceLayout,
        current_allocation: _EngineAllocation,
        next_layout: ResourceLayout,
    ) -> _EngineAllocation:
        """Move site bindings while preserving every carrier's engine slot."""
        next_operands = tuple(
            next_layout.device_label(current_layout._ref_for_label(operand))
            for operand in current_allocation.device_operands
        )
        return _EngineAllocation(next_operands, current_allocation.system_dims)

    def _analyze_plan_facts(self, plan: Sequence[ResolvedStep]) -> _AtomGridPlanFacts:
        """Extend common plan facts with atom-grid lifecycle facts."""
        common = super()._analyze_plan_facts(plan)
        return _AtomGridPlanFacts.from_common(
            common,
            has_loss=any(isinstance(step, LossStep) for step in plan),
            has_refill=any(isinstance(step, RefillStep) for step in plan),
        )

    def _state_is_stochastic(self, facts: _PlanFacts) -> bool:
        """Interpret atom loss using this backend's state representation."""
        atom_facts = cast(_AtomGridPlanFacts, facts)
        return super()._state_is_stochastic(facts) or atom_facts.has_loss

    def _validate_method_support(
        self, config: _ResultConfig, facts: _PlanFacts
    ) -> None:
        """Reject operator methods that cannot carry atom occupancy state."""
        super()._validate_method_support(config, facts)
        atom_facts = cast(_AtomGridPlanFacts, facts)
        if self._is_operator and atom_facts.has_atom_lifecycle:
            raise BackendValidationError(
                f"method={self._state_field!r} cannot represent atom occupancy, "
                "loss, or refill; use method='statevector' or 'density_matrix'"
            )

    def _apply_rearrange(
        self, layout: ResourceLayout, applied: AppliedOperation
    ) -> ResourceLayout:
        """Return a new layout with this Rearrange's atoms relabeled to new sites.

        Changes only device-site labels, never engine indices, so the caller
        emits no step and the quantum state is unchanged (M-B2). Updates each
        listed ref unconditionally, without checking whether its site holds an
        atom (M-B7); atomicity means a swap needs no temporary site (S-B1).
        Injectivity is checked over the full layout, ignoring occupancy (M-B3).

        Raises:
            BackendValidationError: If the Rearrange carries a condition
                (M-B6); a destination site does not exist on the device; a
                named ref is foreign to the layout; or the result is not
                injective.
        """
        if applied.condition is not None:
            raise BackendValidationError("Rearrange must be unconditional")
        sites = applied.operation.sites
        capacity = self._rows * self._cols
        for site in sites:
            if not 0 <= site < capacity:
                raise BackendValidationError(
                    f"Rearrange target site {site} does not exist on the "
                    f"({self._rows}x{self._cols}) device"
                )
        new_labels = {ref: layout.device_label(ref) for ref in layout.refs}
        for ref, site in zip(applied.targets, sites):
            if ref not in new_labels:
                raise BackendValidationError(
                    "Rearrange names a ref that is not part of this layout"
                )
            new_labels[ref] = site
        if len(set(new_labels.values())) != len(new_labels):
            raise BackendValidationError(
                "Rearrange result is not injective: two atoms would share a site"
            )
        return ResourceLayout(new_labels)
