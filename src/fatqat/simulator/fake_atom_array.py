"""Neutral-atom simulator with occupancy and dynamic pairing.

The backend has a fixed native gate set but no fixed geometry. Programs with
neither ``Put`` nor an operation selected by a matching atom-loss source start
with every declared site occupied. Once either lifecycle feature is active,
every site starts empty and ``Put`` loads atoms explicitly.

This simulator is intended for program and compiler testing. It does not model
pulse timing, transport, or Hamiltonian dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from .. import operations as ops
from ..errors import BackendValidationError
from ..implementation import (
    MatrixImplementationMap,
    default_matrix_implementation_map,
)
from ..noise import NoiseModel
from ..program import Program, _AppliedOperation
from ..resource_layout import ResourceLayout
from .._backends.steps import (
    LossStep,
    PutStep,
)
from ._connectivity import _AtomConnectivity
from ._execution_contract import _PlanFacts
from .planning import _lower_channels, _lower_put
from .simulator import Simulator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..implementation import MatrixImplementation
    from ..operations import Operation
    from ..result import _ResultConfig
    from .._backends.backend_utils import _LoweringContext
    from .simulator import ProgramInstruction
    from .._backends.steps import ResolvedStep


@dataclass(frozen=True, slots=True)
class _AtomArrayPlanFacts:
    """Plan facts owned by the atom occupancy lifecycle."""

    has_loss: bool = False
    has_atom_lifecycle: bool = False


def fake_atom_array_implementation_map() -> MatrixImplementationMap:
    """Native gate map: RX/RY/RZ and CZ without fixed operand restrictions.

    CZ is registered as a single universal rule; two-qubit-gate *legality* is
    decided at lowering against the program's dynamic pairing graph (see
    AtomArraySimulator._lower).
    """
    defaults = default_matrix_implementation_map()
    rx_rule = defaults.implementation_for(ops.RX)
    ry_rule = defaults.implementation_for(ops.RY)
    rz_rule = defaults.implementation_for(ops.RZ)
    cz_rule = defaults.implementation_for(ops.CZ)

    m = MatrixImplementationMap()
    m.add(ops.RX, rx_rule)
    m.add(ops.RY, ry_rule)
    m.add(ops.RZ, rz_rule)
    m.add(ops.CZ, cz_rule)
    return m


class AtomArraySimulator(Simulator):
    """Simulate a neutral-atom connectivity and occupancy hardware profile.

    Hardware profile:

    - Native gates: ``RX``, ``RY``, ``RZ``, and ``CZ``.
    - Connectivity: ``Pair`` and ``Unpair`` update the pairing graph; ``CZ``
      is legal only while its atoms are paired. There is no fixed geometry.
    - Layout: registers map to flat labels in declaration order. ``num_sites``
      optionally limits capacity.
    - Occupancy: programs using ``Put`` or matched atom loss, even with
      ``p=0``, start empty; otherwise every declared site starts occupied.
      Gates and reset do nothing on an empty site, which measures as the
      erasure digit ``2``.
    - Methods: all four methods are selectable without an atom lifecycle;
      ``Put`` and loss require ``statevector`` or ``density_matrix``.

    The simulator validates the program as written; it does not transport,
    pair, route, or transpile atoms automatically.
    """

    _supports_loss = True

    def __init__(
        self,
        *,
        num_sites: int | None = None,
        method: str = "statevector",
        runtime: str = "numpy",
        noise: NoiseModel | None = None,
    ) -> None:
        """Create a constrained neutral-atom simulator.

        Args:
            num_sites: Number of trap sites the device holds (its atom
                capacity). ``None`` (the default) imposes no capacity limit,
                so a program of any size binds; if given, must be a positive
                integer.
            method: ``"statevector"`` (or ``"SV"``), ``"density_matrix"``
                (or ``"DM"``), ``"unitary"``, or ``"superop"``. Names are
                case-insensitive.
            runtime: ``"numpy"`` (default, direct execution) or ``"numba"``
                (lazy JIT). See ``Simulator`` for runtime-specific
                execution controls.
            noise: Optional ``NoiseModel``. ``None`` keeps the backend ideal;
                this class has no built-in reference noise model.

        Raises:
            TypeError: If ``num_sites`` is neither ``None`` nor an int (bools
                rejected).
            ValueError: If ``num_sites`` is given and not positive.
            BackendValidationError: If ``method`` or ``runtime`` is invalid,
                or ``noise`` contains a source this simulator cannot run.
        """
        if num_sites is not None:
            if not isinstance(num_sites, int) or isinstance(num_sites, bool):
                raise TypeError(
                    f"num_sites must be an int or None, got {type(num_sites)!r}"
                )
            if num_sites <= 0:
                raise ValueError(f"num_sites must be positive, got {num_sites}")
        self._num_sites = num_sites
        super().__init__(
            method=method,
            runtime=runtime,
            implementation_map=fake_atom_array_implementation_map(),
            noise=noise,
        )

    @property
    def implementation_map(self) -> MatrixImplementationMap:
        """Return the native operation map.

        The map contains ``RX``, ``RY``, ``RZ``, and ``CZ``. A program's
        current ``Pair`` state determines whether a particular ``CZ`` can run.
        """
        return self._impl_map.copy()

    def _resolve_resource_layout(
        self,
        program: Program,
        supplied_layout: ResourceLayout | None = None,
    ) -> ResourceLayout:
        """Reject shapes the device can't run, then map by declaration order.

        No coordinates: a GridRegister (if any) is treated as a plain flat
        register (its rows/cols carry no physical meaning here), so binding is
        the base class's declaration-order identity mapping.
        """
        n_subsystems = sum(register.size for register in program.quantum_registers)
        capacity = self._num_sites
        if capacity is not None and n_subsystems > capacity:
            raise BackendValidationError(
                f"AtomArraySimulator supports at most {capacity} atoms, "
                f"got {n_subsystems}"
            )
        dims = (
            register.dim
            for register in program.quantum_registers
            for _ in range(register.size)
        )
        if any(dim != 2 for dim in dims):
            raise BackendValidationError(
                "AtomArraySimulator only supports qubit dimensions"
            )
        return super()._resolve_resource_layout(program, supplied_layout)

    def _lower(
        self, operations: Sequence[ProgramInstruction], context: _LoweringContext
    ) -> list[ResolvedStep]:
        """Apply this program's atom lifecycle, then lower normally.

        When the program contains ``Put`` or atom loss, every site starts empty
        and `~fatqat.operations.Put` loads a fresh ``|0>`` atom into its
        targets. A target that can never hold an atom (never named in any
        ``Put``) has its gates statically dropped; ``Put``, ``Pair``, and
        ``Unpair`` are themselves never dropped this way. Without ``Put`` or
        loss, no occupancy is imposed and every declared qubit is present,
        exactly like the plain backend.

        Two-qubit-gate legality follows the connectivity graph, not a fixed
        topology. Connectivity starts empty and evolves at each
        `~fatqat.operations.Pair`/`~fatqat.operations.Unpair`; the plan is
        lowered in segments split at each of those, and within a segment a
        two-qubit gate whose pair is not currently paired raises
        `~fatqat.errors.BackendValidationError` (a compile-time program error,
        unlike a run-time atom loss, which the engine drops silently per shot).
        Pair/Unpair change no quantum state and
        emit no execution step, but do emit any attached movement-cost channel
        noise. The resource layout carries no coordinates and never changes, so
        every segment lowers under the same layout.

        Occupancy is seeded empty whenever the program has any atom lifecycle
        (a ``Put`` or an atom loss); ``Put`` then fills its targets per shot.
        That empty seed is not a plan step - it is an initialization input the
        engine receives at run start (see ``_initial_occupancy``), because
        seeding occupancy is the atom simulator's own setup, not an operation.
        ``Measurement`` always lowers normally; an empty (never-``Put``) site
        measures the erasure digit ``2`` under the occupancy guard.

        Raises:
            BackendValidationError: If a ``Pair`` or ``Unpair`` carries a
                condition, or if a two-qubit gate targets a pair that is not
                currently paired (see :py:meth:`_require_pairing`).
        """
        resource_layout = context.resource_layout

        # Targets that can ever hold an atom: those named in some Put.
        put_targets = {
            t
            for step in operations
            if isinstance(step, _AppliedOperation)
            and isinstance(step.operation, ops.PutGate)
            for t in step.targets
        }

        realized: list[ProgramInstruction] = []
        for step in operations:
            if (
                put_targets
                and isinstance(step, _AppliedOperation)
                and not isinstance(
                    step.operation, (ops.PutGate, ops.PairGate, ops.UnpairGate)
                )
                and any(t not in put_targets for t in step.targets)
            ):
                continue  # a target can never hold an atom -> static drop
            realized.append(step)

        connectivity = _AtomConnectivity()
        plan: list[ResolvedStep] = []
        segment: list[ProgramInstruction] = []
        for step in realized:
            if isinstance(step, _AppliedOperation) and isinstance(
                step.operation, (ops.PairGate, ops.UnpairGate)
            ):
                plan.extend(self._lower_segment(segment, connectivity, context))
                segment = []
                plan.extend(
                    _lower_channels(
                        type(step.operation),
                        step.targets,
                        None,
                        resource_layout,
                        context.engine_allocation,
                        self._noise_model,
                        self._channel_map,
                    )
                )
                connectivity = self._apply_pairing(connectivity, step)
                continue
            segment.append(step)
        plan.extend(self._lower_segment(segment, connectivity, context))
        return plan

    def _initial_occupancy(self, facts: _AtomArrayPlanFacts) -> frozenset[int] | None:
        """Seed occupancy empty whenever the program has an atom lifecycle.

        A ``Put`` or an atom loss makes occupancy shot-dependent: every site
        starts empty and `~fatqat.operations.Put` loads a fresh ``|0>`` into its
        targets per shot. Without either, no occupancy is imposed and every
        declared qubit is present (``None``), exactly like the plain backend.
        This is the engine's per-shot occupancy seed, delivered as a run
        initialization input rather than a lowered plan step.
        """
        return frozenset() if facts.has_atom_lifecycle else None

    def _lower_segment(
        self,
        segment: Sequence[ProgramInstruction],
        connectivity: _AtomConnectivity,
        context: _LoweringContext,
    ) -> list[ResolvedStep]:
        """Lower one inter-pairing segment, rejecting unpaired two-qubit gates.

        The layout never changes, so the segment lowers under the unchanged
        context. A two-qubit gate whose pair is not currently paired is a
        program-construction error - the pairing graph is fixed at compile time
        by ``Pair``/``Unpair``, independent of any shot - so it is rejected
        here (see :py:meth:`_require_pairing`), distinct from a per-shot atom
        loss, which the engine drops silently.
        """
        for step in segment:
            self._require_pairing(step, connectivity)

        plan: list[ResolvedStep] = []
        ordinary: list[ProgramInstruction] = []
        lower_common = super()._lower

        def flush_ordinary() -> None:
            if ordinary:
                plan.extend(lower_common(tuple(ordinary), context))
                ordinary.clear()

        for step in segment:
            if isinstance(step, _AppliedOperation) and isinstance(
                step.operation, ops.PutGate
            ):
                flush_ordinary()
                plan.extend(
                    _lower_put(
                        step,
                        context.resource_layout,
                        context.engine_allocation,
                        context.classical_allocation,
                        self._noise_model,
                    )
                )
            else:
                ordinary.append(step)
        flush_ordinary()
        return plan

    def _require_pairing(
        self, step: ProgramInstruction, connectivity: _AtomConnectivity
    ) -> None:
        """Reject a two-qubit gate whose atoms are not currently paired.

        Only entangling two-qubit gates need a pairing. Measurements and the
        boundary/lifecycle operations
        ``Put``/``Pair``/``Unpair``/``Reset``/``Barrier`` never need one;
        single-qubit gates never need one either.

        An unpaired two-qubit gate is a compile-time program error: the
        connectivity graph is deterministic, set only by ``Pair``/``Unpair``, so
        a gate on an unpaired pair can never take effect and almost always means
        a missing ``Pair``. This is reported here, as opposed to the per-shot,
        stochastic atom loss that prevents an otherwise-legal gate, which the
        engine drops silently as a physical effect.

        Raises:
            BackendValidationError: If ``step`` is a two-qubit gate whose two
                atoms are not currently paired.
        """
        if not isinstance(step, _AppliedOperation):
            return
        if isinstance(
            step.operation,
            (
                ops.PutGate,
                ops.PairGate,
                ops.UnpairGate,
                ops.ResetGate,
                ops.BarrierGate,
            ),
        ):
            return
        if len(step.targets) != 2:
            return
        a, b = step.targets
        if not connectivity.are_paired(a, b):
            a_label = getattr(a, "index", a)
            b_label = getattr(b, "index", b)
            raise BackendValidationError(
                f"{step.operation.name} on atoms ({a_label}, {b_label}) requires "
                "the atoms to be paired first: a two-qubit gate is legal only "
                "while its atoms are connected in the dynamic pairing graph. "
                "Add ops.Pair on this pair before the gate. This is a program "
                "error, distinct from an atom lost mid-circuit, which is dropped "
                "silently per shot."
            )

    def _analyze_lowered_plan(
        self, plan: tuple[ResolvedStep, ...]
    ) -> tuple[_PlanFacts, frozenset[int] | None]:
        """Translate atom lifecycle semantics into common plan consequences."""
        common = self._analyze_common_plan_facts(
            plan,
            claimed_step_types=(LossStep, PutStep),
        )
        atom_facts = self._analyze_atom_plan_facts(plan)
        execution_shape = common.execution_shape
        deferred_measurements = common.deferred_measurements
        if not self._is_operator and atom_facts.has_atom_lifecycle:
            execution_shape = "per_shot"
            deferred_measurements = ()
        translated = replace(
            common,
            execution_shape=execution_shape,
            deferred_measurements=deferred_measurements,
            stochastic_final_state=(
                common.stochastic_final_state or atom_facts.has_loss
            ),
        )
        return translated, self._initial_occupancy(atom_facts)

    def _analyze_atom_plan_facts(
        self, plan: Sequence[ResolvedStep]
    ) -> _AtomArrayPlanFacts:
        """Collect the atom-only lifecycle facts in one cohesive scan."""
        has_loss = False
        has_put = False
        for step in plan:
            has_loss = has_loss or isinstance(step, LossStep)
            has_put = has_put or isinstance(step, PutStep)
        return _AtomArrayPlanFacts(
            has_loss=has_loss,
            has_atom_lifecycle=has_loss or has_put,
        )

    def _validate_method_support(
        self,
        config: _ResultConfig,
        facts: _PlanFacts,
        *,
        initial_occupied: frozenset[int] | None,
    ) -> None:
        """Reject operator methods that cannot carry atom occupancy state."""
        super()._validate_method_support(
            config,
            facts,
            initial_occupied=initial_occupied,
        )
        if self._is_operator and initial_occupied is not None:
            raise BackendValidationError(
                f"method={self._state_field!r} cannot represent atom occupancy, "
                "loss, or refill; use method='statevector' or 'density_matrix'"
            )

    def _apply_pairing(
        self, connectivity: _AtomConnectivity, applied: _AppliedOperation
    ) -> _AtomConnectivity:
        """Return the connectivity after this Pair/Unpair (must be unconditional)."""
        if applied.condition is not None:
            raise BackendValidationError(
                f"{applied.operation.name} must be unconditional"
            )
        a, b = applied.targets
        if isinstance(applied.operation, ops.PairGate):
            return connectivity.pair(a, b)
        return connectivity.unpair(a, b)
