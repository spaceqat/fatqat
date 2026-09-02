"""Neutral-atom simulator with occupancy and dynamic pairing.

The backend has a fixed native gate set but no fixed geometry. Every site
declared by the program starts empty, and ``Put`` loads atoms explicitly.

This simulator is intended for program and compiler testing. It does not model
pulse timing, transport, or Hamiltonian dynamics.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from .. import operations as ops
from .._backends.backend_utils import _canonicalize_method
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
    from ..registers import RegisterRef
    from .._backends.backend_utils import _LoweringContext
    from .simulator import ProgramInstruction
    from .._backends.steps import ResolvedStep


def _fake_atom_array_implementation_map() -> MatrixImplementationMap:
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
    - Layout: the program declares the site count, and registers map to flat
      labels in declaration order.
    - Occupancy: every declared site starts empty. ``Put`` loads an atom;
      supported, correctly paired gates and reset do nothing on an empty site,
      which measures as the erasure digit ``2``.
    - Methods: atom occupancy requires ``statevector`` or ``density_matrix``.

    The simulator validates the program as written; it does not transport,
    pair, route, or transpile atoms automatically.
    """

    _supports_loss = True

    def __init__(
        self,
        *,
        method: str = "statevector",
        runtime: str = "numba",
        noise: NoiseModel | None = None,
    ) -> None:
        """Create a constrained neutral-atom simulator.

        Args:
            method: ``"statevector"`` (or ``"SV"``), ``"density_matrix"``
                (or ``"DM"``). Names are case-insensitive.
            runtime: ``"numba"`` (default, lazy JIT) or ``"numpy"`` (direct
                execution). See ``Simulator`` for runtime-specific
                execution controls.
            noise: Optional ``NoiseModel``. ``None`` keeps the backend ideal;
                this class has no built-in reference noise model.

        Raises:
            BackendValidationError: If ``method`` or ``runtime`` is invalid,
                or ``noise`` contains a source this simulator cannot run.
        """
        canonical_method = _canonicalize_method(
            method, {"statevector", "density_matrix"}
        )
        if canonical_method is None:
            raise BackendValidationError(
                f"unsupported method={method!r}; AtomArraySimulator supports "
                "only 'statevector'/'SV' or 'density_matrix'/'DM'"
            )
        super().__init__(
            method=canonical_method,
            runtime=runtime,
            implementation_map=_fake_atom_array_implementation_map(),
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
        """Reject unsupported dimensions, then map by declaration order.

        No coordinates: a GridRegister (if any) is treated as a plain flat
        register (its rows/cols carry no physical meaning here), so binding is
        the base class's declaration-order identity mapping.
        """
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

        Every site starts empty and `~fatqat.operations.Put` loads a fresh
        ``|0>`` atom into its targets. Operation support and pairing are
        validated independently of occupancy. After validation, gates and
        resets targeting a site that is never named in any ``Put`` are omitted
        from the execution plan; a valid operation on a site emptied by loss
        has no effect for that shot.

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

        Occupancy is seeded empty and ``Put`` then fills its targets per shot.
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
        put_targets = frozenset(
            target
            for step in operations
            if isinstance(step, _AppliedOperation)
            and isinstance(step.operation, type(ops.Put))
            for target in step.targets
        )

        connectivity = _AtomConnectivity()
        plan: list[ResolvedStep] = []
        segment: list[ProgramInstruction] = []
        for step in operations:
            if isinstance(step, _AppliedOperation) and isinstance(
                step.operation, (type(ops.Pair), type(ops.Unpair))
            ):
                plan.extend(
                    self._lower_segment(segment, connectivity, put_targets, context)
                )
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
        plan.extend(self._lower_segment(segment, connectivity, put_targets, context))
        return plan

    def _initial_occupancy(self) -> frozenset[int]:
        """Return the empty per-shot occupancy seed for this atom array."""
        return frozenset()

    def _lower_segment(
        self,
        segment: Sequence[ProgramInstruction],
        connectivity: _AtomConnectivity,
        put_targets: frozenset[RegisterRef],
        context: _LoweringContext,
    ) -> list[ResolvedStep]:
        """Lower one inter-pairing segment, rejecting unpaired two-qubit gates.

        The layout never changes, so the segment lowers under the unchanged
        context. A two-qubit gate whose pair is not currently paired is a
        program-construction error - the pairing graph is fixed at compile time
        by ``Pair``/``Unpair``, independent of any shot - so it is rejected
        here (see :py:meth:`_require_pairing`), distinct from a per-shot atom
        loss, which the engine drops silently. Operations on sites that no
        ``Put`` can load still pass through common lowering for validation;
        their resolved work is then omitted from the execution plan.
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
                step.operation, type(ops.Put)
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
            elif isinstance(step, _AppliedOperation) and any(
                target not in put_targets for target in step.targets
            ):
                flush_ordinary()
                # Reuse canonical lowering for validation, then omit work that
                # can never execute because at least one target is never loaded.
                lower_common((step,), context)
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
                type(ops.Put),
                type(ops.Pair),
                type(ops.Unpair),
                type(ops.Reset),
                type(ops.Barrier),
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
        has_loss = any(isinstance(step, LossStep) for step in plan)
        translated = replace(
            common,
            execution_shape="per_shot",
            deferred_measurements=(),
            stochastic_final_state=common.stochastic_final_state or has_loss,
        )
        return translated, self._initial_occupancy()

    def _apply_pairing(
        self, connectivity: _AtomConnectivity, applied: _AppliedOperation
    ) -> _AtomConnectivity:
        """Return the connectivity after this Pair/Unpair (must be unconditional)."""
        if applied.condition is not None:
            raise BackendValidationError(
                f"{applied.operation.name} must be unconditional"
            )
        a, b = applied.targets
        if isinstance(applied.operation, type(ops.Pair)):
            return connectivity.pair(a, b)
        return connectivity.unpair(a, b)
