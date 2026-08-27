"""Fake neutral-atom backend for compiler prototyping (connectivity model).

A configurable-capacity statevector target for the neutral-atom model. The
device holds ``num_sites`` trap sites (an optional capacity, unbounded by
default), each holding at most one atom; it carries no fixed topology and no
coordinates. This
is a prototype execution target, not a realistic device model: no routing, no
timing, no transport, and ideal by default.

Native gate set is exactly `~fatqat.operations.RX`, `~fatqat.operations.RY`,
`~fatqat.operations.RZ` (single-qubit, any atom) and
`~fatqat.operations.CZ`. Two-qubit-gate legality is
dynamic: a ``CZ`` is legal only on a pair that is currently *paired* in the
program's connectivity graph, which `~fatqat.operations.Pair` and
`~fatqat.operations.Unpair` mutate mid-circuit.
A ``CZ`` on a pair that is not currently paired raises
`~fatqat.errors.BackendValidationError`: because the connectivity graph is
fixed at compile time, an unpaired ``CZ`` can never take effect and is treated
as a program error (a missing ``Pair``). That is distinct from a ``CZ`` an
atom loss prevents at run time, which is dropped silently per shot.

Atom occupancy is separate from the quantum state. Every site starts empty;
`~fatqat.operations.Put` loads a fresh ``|0>`` atom into its targets (``Put``
may appear any number of times). A gate
whose target is never named in any ``Put`` can never hold an atom and is
statically dropped; a program that uses neither ``Put`` nor atom loss keeps
every declared qubit present, behaving like a plain statevector backend.
`~fatqat.noise.Loss` attached to a gate (or to ``Put``, for imperfect
loading) ejects atoms per shot; a lost or empty site measures the erasure
digit ``2``, distinct from a real ``|0>``.

Binding carries no coordinates: quantum registers map to device labels in
declaration order (a `~fatqat.GridRegister`, if passed, is treated as a plain
flat register). Connectivity, occupancy, and loss are the only atom-specific
lifecycle; see :py:meth:`AtomArraySimulator._lower`.
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
from ..program import AppliedOperation, Program
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
    """Native gate map: RX/RY/RZ and CZ, all legal on any target.

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
    """Statevector backend for the neutral-atom connectivity model.

    A thin statevector-method `~fatqat.simulator.Simulator` specialization:
    same execution engine, same `~fatqat.Result`/`~fatqat.Job` semantics. The
    differences are an optional capacity (``num_sites`` trap sites,
    unbounded by default, with no fixed topology), a fixed native gate
    set (:py:class:`~fatqat.operations.RX`, :py:class:`~fatqat.operations.RY`,
    :py:class:`~fatqat.operations.RZ`, and :py:data:`~fatqat.operations.CZ`),
    coordinate-free resource mapping (declaration-order device labels; a
    :py:class:`~fatqat.GridRegister`, if passed, is treated as a plain flat
    register), and an atom lifecycle of connectivity, occupancy, and loss - see
    :py:meth:`_lower`.

    Connectivity:
        Two-qubit-gate legality is dynamic. Use
        :py:data:`~fatqat.operations.Pair` to connect two atoms and
        :py:data:`~fatqat.operations.Unpair` to disconnect them; a
        :py:data:`~fatqat.operations.CZ` is legal only while its pair is
        currently paired, and raises
        :py:class:`~fatqat.errors.BackendValidationError` otherwise - an
        unpaired ``CZ`` is a program error (a missing ``Pair``), unlike a
        ``CZ`` an atom loss prevents at run time, which is dropped silently per
        shot. Pair/Unpair change no quantum state and emit no execution
        step, but may carry movement-cost channel noise (e.g.
        :py:class:`~fatqat.noise.Loss` or a decoherence channel).

    Atom lifecycle:
        Every site starts empty. :py:data:`~fatqat.operations.Put` loads a
        fresh ``|0>`` atom into its targets; a target never named in any
        ``Put`` can never hold an atom and
        its gates are statically dropped. Attach
        :py:class:`~fatqat.noise.Loss` to a gate to eject atoms per
        shot, or to ``Put`` to model imperfect loading; a lost or empty site
        measures the erasure digit ``2``, distinct from a real ``|0>``. A
        program that uses neither ``Put`` nor loss keeps every declared qubit
        present, behaving like a plain statevector backend. Only this backend
        models atom loss; a generic backend rejects ``Loss`` via
        :py:meth:`check_noise_support` rather than ignoring it.

        .. doctest:: atom_array_loss

           >>> import numpy as np
           >>> import fatqat as fq
           >>> import fatqat.operations as ops
           >>> noise = fq.NoiseModel()
           >>> noise.add(fq.noise.Loss(p=1.0), operation=ops.RX)
           >>> program = fq.Program(1, 1)
           >>> program.add(ops.Put, 0)             # load an atom into site 0
           >>> program.add(ops.RX(np.pi), 0)       # applies, then the atom is lost
           >>> program.measure(0, 0)
           >>> backend = fq.simulator.AtomArraySimulator(num_sites=1, noise=noise)
           >>> backend.run(
           ...     program, shots=10, simulation_config={"seed": 0}
           ... ).result().get_counts()
           {'2': 10}

    Example:
        A Bell pair built as ``CX(0 -> 1)`` on two atoms. ``CX`` is not native
        (it lowers to ``H(target)``, ``CZ``, ``H(target)``, with ``H`` emitted
        as ``RZ(pi)`` then ``RY(pi / 2)`` up to global phase), and the ``CZ``
        only takes effect because ``ops.Pair`` connects the two atoms first;
        without the pairing the ``CZ`` would raise a
        ``BackendValidationError``.

        .. doctest:: atom_array_cx

           >>> import numpy as np
           >>> import fatqat as fq
           >>> import fatqat.operations as ops
           >>> atoms = fq.QuantumRegister(2, name="atoms")
           >>> program = fq.Program([atoms], 2)
           >>> program.add(ops.Put, (0, 1))            # load both atoms
           >>> program.add(ops.Pair, (0, 1))           # connect them so CZ is legal
           >>> def native_h(target):
           ...     program.add(ops.RZ(np.pi), target)
           ...     program.add(ops.RY(np.pi / 2), target)
           >>> native_h(0)                            # superpose the control
           >>> native_h(1)                            # H on the target ...
           >>> program.add(ops.CZ, (0, 1))
           >>> native_h(1)                            # ... completes CX(0 -> 1)
           >>> program.measure_all()

           >>> backend = fq.simulator.AtomArraySimulator()  # unbounded capacity
           >>> counts = backend.run(
           ...     program,
           ...     shots=1000,
           ...     simulation_config={
           ...         "seed": 1,
           ...         "shot_parallelism": "serial",
           ...         "kernel_parallelism": "serial",
           ...     },
           ... ).result().get_counts()
           >>> all(bits[0] == bits[1] for bits in counts)  # only 00 and 11 occur
           True
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
        """Create a fake atom-array backend with the given number of sites.

        Args:
            num_sites: Number of trap sites the device holds (its atom
                capacity). ``None`` (the default) imposes no capacity limit,
                so a program of any size binds; if given, must be a positive
                integer.
            method: State representation, exactly as on
                :py:class:`~fatqat.simulator.Simulator`.
            runtime: Numeric execution runtime, exactly as on
                :py:class:`~fatqat.simulator.Simulator`.
            noise: Optional `~fatqat.NoiseModel`, exactly as on
                `~fatqat.simulator.Simulator`. `None` (the default)
                keeps the backend ideal.

        Raises:
            TypeError: If ``num_sites`` is neither ``None`` nor an int (bools
                rejected).
            ValueError: If ``num_sites`` is given and not positive.
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
        """Return a copy of the compiler-facing device-aware implementation map.

        Examples:
            >>> import fatqat as fq
            >>> import fatqat.operations as ops
            >>> backend = fq.simulator.AtomArraySimulator()
            >>> impl_map = backend.implementation_map
            >>> sorted(operation.name for operation in impl_map.supported_operations())
            ['CZ', 'RX', 'RY', 'RZ']
            >>> impl_map.supports(ops.CCX)
            False
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

        Every site starts empty; `~fatqat.operations.Put` loads a fresh ``|0>``
        atom into its targets. A target that can never hold an atom (never
        named in any ``Put``) has its gates statically dropped; ``Put``,
        ``Pair``, and ``Unpair`` are themselves never dropped this way. When a
        program uses no ``Put`` (and no atom loss), no occupancy is imposed and
        every declared qubit is present, exactly like the plain backend.

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
            if isinstance(step, AppliedOperation)
            and isinstance(step.operation, ops.PutGate)
            for t in step.targets
        }

        realized: list[ProgramInstruction] = []
        for step in operations:
            if (
                put_targets
                and isinstance(step, AppliedOperation)
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
            if isinstance(step, AppliedOperation) and isinstance(
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
            if isinstance(step, AppliedOperation) and isinstance(
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

        Only entangling two-qubit gates need a pairing. ``Measurement`` (not an
        ``AppliedOperation``) and the boundary/lifecycle ops
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
        if not isinstance(step, AppliedOperation):
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
        self, connectivity: _AtomConnectivity, applied: AppliedOperation
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
