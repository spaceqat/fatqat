"""Complete matrix-engine contract and shared Python execution coordination."""

from __future__ import annotations

from copy import deepcopy

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, nullcontext
from functools import partial
from typing import Any, Generic

import numpy as np

from ..._execution_contract import (
    RawResult,
    _SimulationConfig as SimulationConfig,
    _EngineCapabilities,
    _RunContext,
)
from ...._backends.steps import (
    ApplyChannelStep,
    ApplyMatrixStep,
    ApplyParametricMatrixStep,
    MeasurementStep,
    ResolvedStep,
)
from ....result import reduce_to_counts
from .._execution_policy import (
    _ExecutionPolicy as ExecutionPolicy,
)
from ..binding import _bind_parameter_batch
from ..parallel import _run_shots_in_processes
from ..state import EvolutionState, QuantumDataT
from ..workflows.python_workflow import (
    _PythonExecution,
    _run_operator_plan,
    _run_single_pass,
    _run_one_shot,
)


class MatrixEngine(ABC, Generic[QuantumDataT]):
    """Runtime-native numerical operations with shared execution coordination.

    The default preparation and local entries serve Python workflows. Engines
    can override them for compiled execution without changing the run interface.
    """

    # Concrete engines declare all support; numeric/execution helpers may be shared.
    _capabilities: _EngineCapabilities

    def __init__(self, name: str):
        self.name = name

        self._evolution_state: EvolutionState[QuantumDataT] | None = None
        self._dims: tuple[int, ...] = ()
        self._n_clbits = 0

    @property
    def _state(self) -> QuantumDataT | None:
        """Forward existing kernel access to the owned quantum data."""
        if self._evolution_state is None:
            return None
        return self._evolution_state.quantum

    @_state.setter
    def _state(self, value: QuantumDataT | None) -> None:
        """Replace the quantum data without losing current classical state."""
        if value is None:
            self._evolution_state = None
        elif self._evolution_state is None:
            self._evolution_state = EvolutionState(value)
        else:
            self._evolution_state.quantum = value

    @property
    def state(self) -> QuantumDataT:
        """Expose internal runtime storage without requesting a host export."""
        if self._state is None:
            raise RuntimeError("MatrixEngine state has not been initialized.")
        return self._state

    @state.setter
    def state(self, value: QuantumDataT) -> None:
        self._state = value

    @property
    def n_subsystems(self) -> int:
        return len(self._dims)

    @property
    def _state_field(self) -> str:
        """Use the declared representation for evolving state and results."""
        return self.capabilities.quantum.representation

    @property
    def capabilities(self) -> _EngineCapabilities:
        """Return supported quantum, classical, and numerical execution."""
        return self._capabilities

    @abstractmethod
    def _select_policy(
        self,
        plan: Sequence[ResolvedStep],
        *,
        context: _RunContext,
        simulation_config: SimulationConfig,
        initial_state: EvolutionState[QuantumDataT],
    ) -> ExecutionPolicy:
        """Validate the request and choose its workflow and scheduling.

        Implementations use shared request classification and validation, then
        resolve runtime-specific choices before preparing numerical payloads.
        """

    def configure_system(self, system_dims: Sequence[int], n_clbits: int = 0) -> None:
        """Configure dimensions without allocating an evolving state."""
        self._dims = tuple(int(d) for d in system_dims)
        self._n_clbits = int(n_clbits)
        self._state = None

    @abstractmethod
    def initialize(
        self,
        system_dims: Sequence[int],
        n_clbits: int = 0,
        *,
        initial_state: EvolutionState[QuantumDataT] | None = None,
    ) -> None:
        """Initialize owned quantum and classical storage from validated input.

        For state methods, ``initial_state.quantum`` is a `_LoweredBasisState`
        (``Simulator`` lowers the ground state to one too) or a validated dense
        array; the engine builds a basis state itself, in its own layout and
        array type. Operator methods receive None: they have no initial state.
        """

    def _retain_step_caches(self, plan: tuple[ResolvedStep, ...]) -> None:
        """Retain only the effective plan's identity-keyed step resolutions.

        Payload preparation calls this before resolving or compiling the plan.
        Shot initialization leaves these entries intact for trajectory reuse.
        Engines without step caches have nothing to prune.
        """

    def prepare_payload(
        self,
        plan: tuple[ResolvedStep, ...],
        *,
        system_dims: tuple[int, ...],
        n_clbits: int,
        policy: ExecutionPolicy,
    ) -> Any:
        """Retain the effective plan and finalized deferred measurements.

        The payload is opaque at the dispatch boundary: overrides may return
        their own records paired with their local and shot-batch entries. This
        default returns ``_PythonExecution`` for the Python workflows.

        Overrides establish readiness for runtime-specific preparation before
        execution consumes it, retaining resources throughout pending work.
        The later execution scope does not retroactively cover preparation.
        """
        self.configure_system(system_dims, n_clbits)
        self._retain_step_caches(plan)
        measurements = (
            tuple(
                pair
                for step in plan
                if isinstance(step, MeasurementStep)
                for pair in zip(step.measured_indices, step.classical_indices)
            )
            if policy.execution_path == "single_pass"
            else ()
        )
        return self._bind_kernels_to_python_workflow(plan, measurements, policy=policy)

    @abstractmethod
    def _bind_kernels_to_python_workflow(
        self,
        plan: tuple[ResolvedStep, ...],
        measurements: tuple[tuple[int, int], ...] = (),
        *,
        policy: ExecutionPolicy | None = None,
    ) -> _PythonExecution:
        """Bind numerical functions and execution-specific layouts to the Python plan.

        The optional policy supports direct single-operation callers; normal
        execution always supplies its selected scheduling policy.
        """

    def _execution_scope(self, policy: ExecutionPolicy) -> AbstractContextManager[None]:
        """Set up the runtime context for local or shot-batch execution.

        Overrides keep dependent work ordered and its resources alive. Successful
        exit completes this invocation's numerical work, transfers, required
        classical updates, and preparation dependencies, even without host output.
        Unrelated device work need not finish.

        On failure, clean up and propagate execution errors. If cleanup also
        fails, keep the original error inspectable. Quantum state may outlive
        the scope. This does not require per-gate synchronization or make
        concurrent calls safe.
        """
        return nullcontext()

    def execute(
        self,
        plan: tuple[ResolvedStep, ...],
        *,
        context: _RunContext,
        simulation_config: SimulationConfig,
        initial_state: EvolutionState[QuantumDataT],
    ) -> Iterator[RawResult[QuantumDataT]]:
        """Bind and execute a plan for every entry of its parameter batch.

        A plan without parametric steps is a batch of one; a value-injected
        sweep plan has one entry per set of parameter values. Results are
        yielded in batch order as each entry finishes, so a caller that
        consumes them one at a time keeps one entry's state alive. The policy
        reads only step kinds and run settings, so it is planned once from the
        first entry; each bound plan then executes like an ordinary plan.
        """
        policy: ExecutionPolicy | None = None
        for bound in _bind_parameter_batch(plan, context.system_dims):
            if policy is None:
                policy = self._select_policy(
                    bound,
                    context=context,
                    simulation_config=simulation_config,
                    initial_state=initial_state,
                )
            yield self._dispatch_by_shot_strategy(bound, context, policy, initial_state)

    def continue_from_states(
        self,
        tail: tuple[ResolvedStep, ...],
        states: Sequence[EvolutionState[QuantumDataT]],
        *,
        context: _RunContext,
        simulation_config: SimulationConfig,
    ) -> list[RawResult[QuantumDataT]]:
        """Continue already-evolved states with one shared concrete tail.

        Sampled expectations evolve each batch entry once, then run every
        term's measurement tail from those states. The tail is the same for
        all states, so its policy is selected once from the first state and
        its payload prepared once. ``tail`` has no parametric steps, and
        ``states`` come from one plan, so they share their classical layout.
        The selected policy is followed as in any run: a tail that needs
        per-shot evolution (from states with atom occupancy) may run its shots
        in worker processes, which prepare their own payloads per state.
        """
        policy = self._select_policy(
            tail,
            context=context,
            simulation_config=simulation_config,
            initial_state=states[0],
        )
        if policy.shot_strategy == "processes":
            return [
                self._dispatch_by_shot_strategy(tail, context, policy, state)
                for state in states
            ]
        payload = self.prepare_payload(
            tail,
            system_dims=context.system_dims,
            n_clbits=context.n_clbits,
            policy=policy,
        )
        return [
            self.execute_payload(context, payload, policy, initial_state=state)
            for state in states
        ]

    def expectation_values(
        self,
        plan: tuple[ResolvedStep, ...],
        observables: Sequence[tuple[tuple[float, tuple[tuple[int, str], ...]], ...]],
        *,
        context: _RunContext,
        simulation_config: SimulationConfig,
        initial_state: EvolutionState[QuantumDataT],
    ) -> list[tuple[float, ...]]:
        """Evolve every parameter entry and contract Pauli sums on its state.

        Returns one tuple of values per entry, in batch order. Empty
        observables validate execution controls without evolving a state.
        """
        policy = self._select_policy(
            plan,
            context=context,
            simulation_config=simulation_config,
            initial_state=initial_state,
        )
        if not observables:
            return []
        values = []
        # Contract each entry as it is yielded, so only one evolved state is
        # alive at a time; the caller keeps only the scalar values. Unlike
        # `Simulator._execute_plan`, which collects every entry first, this
        # interleaves engine runs with contraction, which measured ~10-20%
        # slower for small (6-qubit) states.
        for raw in self.execute(
            plan,
            context=context,
            simulation_config=simulation_config,
            initial_state=initial_state,
        ):
            if raw.state is None:
                raise RuntimeError("expectation execution returned no state")
            values.append(
                self._expectation_values(raw.state, observables, policy=policy)
            )
        return values

    def _dispatch_by_shot_strategy(
        self,
        plan: tuple[ResolvedStep, ...],
        context: _RunContext,
        policy: ExecutionPolicy,
        initial_state: EvolutionState[QuantumDataT],
    ) -> RawResult[QuantumDataT]:
        """Execute a plan under ``policy``: here or in worker processes.

        In-process shot strategies prepare a payload and execute it here; the
        process strategy runs shots in workers and reduces their rows.
        """
        if any(isinstance(step, ApplyParametricMatrixStep) for step in plan):
            raise RuntimeError("parametric steps must be bound before execution")
        if policy.shot_strategy in ("none", "serial", "threads"):
            payload = self.prepare_payload(
                plan,
                system_dims=context.system_dims,
                n_clbits=context.n_clbits,
                policy=policy,
            )
            return self.execute_payload(
                context, payload, policy, initial_state=initial_state
            )

        if policy.shot_strategy != "processes":
            raise RuntimeError(f"Unknown shot strategy: {policy.shot_strategy!r}")
        # Workers own numerical bindings. Release resources from any previous local run
        # without building a second, unused set of numerical bindings here.
        self.configure_system(context.system_dims, context.n_clbits)
        self._retain_step_caches(())
        snapshots = _run_shots_in_processes(
            self._process_engine_factory(), context, plan, policy, initial_state
        )
        rows = np.asarray(snapshots, dtype=int).reshape(
            (len(snapshots), context.n_clbits)
        )
        outcome_keys, outcome_counts = reduce_to_counts(rows)
        return RawResult(
            outcome_keys=outcome_keys,
            outcome_counts=outcome_counts,
        )

    def _process_engine_factory(self) -> Callable[[], MatrixEngine[QuantumDataT]]:
        """Carry constructor configuration to workers without live execution state.

        Engines with additional constructor options override this hook.
        """
        return partial(type(self), name=self.name)

    def execute_payload(
        self,
        context: _RunContext,
        payload: Any,
        policy: ExecutionPolicy,
        *,
        initial_state: EvolutionState[QuantumDataT],
    ) -> RawResult[QuantumDataT]:
        """Execute a prepared state or operator payload here through Python."""
        assert policy.shot_strategy != "processes"
        self.configure_system(context.system_dims, context.n_clbits)
        with self._execution_scope(policy):
            state_requested = context.request.final_state
            outcomes = None
            if policy.execution_path == "per_shot":
                rows = self._run_per_shot(
                    payload,
                    context,
                    initial_state,
                    state_requested=(
                        context.request.final_state
                        or context.request.final_classical_state
                    ),
                )
                outcomes = reduce_to_counts(rows) if rows is not None else None
            else:
                rng = np.random.default_rng(context.seed)
                self.initialize(
                    context.system_dims,
                    context.n_clbits,
                    initial_state=initial_state,
                )
                if policy.execution_path == "operator":
                    assert policy.shot_strategy == "none"
                    assert self._evolution_state is not None
                    _run_operator_plan(
                        payload,
                        self._evolution_state,
                        context.system_dims,
                        rng,
                    )
                else:
                    assert policy.execution_path == "single_pass"
                    assert self._evolution_state is not None
                    outcomes = _run_single_pass(
                        payload,
                        self._evolution_state,
                        context,
                        rng,
                        state_requested=state_requested,
                    )
            outcome_keys, outcome_counts = outcomes or (None, None)
            state = self.export_state() if state_requested else None
            classical_state = None
            if context.request.final_classical_state:
                # Copy so later execution cannot change the returned digits.
                assert self._evolution_state is not None
                classical_state = deepcopy(self._evolution_state.classical)
                if classical_state.clbits is None:
                    classical_state.clbits = [0] * context.n_clbits
            return RawResult(
                outcome_keys=outcome_keys,
                outcome_counts=outcome_counts,
                state=state,
                classical_state=classical_state,
            )

    def execute_worker_shots(
        self,
        context: _RunContext,
        payload: Any,
        seed_batch: list[np.random.SeedSequence],
        policy: ExecutionPolicy,
        *,
        initial_state: EvolutionState[QuantumDataT],
    ) -> list[tuple[int, ...]]:
        """Non-dispatching worker entry for one ordered batch of shots."""
        assert policy.shot_strategy == "serial"
        assert policy.kernel_strategy == "serial"
        self.configure_system(context.system_dims, context.n_clbits)
        with self._execution_scope(policy):
            return self._run_shot_batch(payload, context, seed_batch, initial_state)

    def _run_per_shot(
        self,
        payload: _PythonExecution,
        context: _RunContext,
        initial_state: EvolutionState[QuantumDataT],
        *,
        state_requested: bool,
    ) -> np.ndarray | None:
        """Assign local trajectory streams and return requested classical samples."""
        n_iters = context.shots if context.request.counts else int(state_requested)
        snapshots = self._run_shot_batch(
            payload,
            context,
            np.random.SeedSequence(context.seed).spawn(n_iters),
            initial_state,
        )
        if not context.request.counts:
            return None
        return np.asarray(snapshots, dtype=int).reshape(
            (len(snapshots), context.n_clbits)
        )

    def _run_shot_batch(
        self,
        payload: _PythonExecution,
        context: _RunContext,
        seed_sequences: Sequence[np.random.SeedSequence],
        initial_state: EvolutionState[QuantumDataT],
    ) -> list[tuple[int, ...]]:
        """Initialize each trajectory within the caller's active runtime scope.

        Local and process entries share this lifecycle. The workflow borrows the
        current evolution container and may replace it; the engine adopts the result.
        """
        snapshots = []
        for seed_sequence in seed_sequences:
            self.initialize(
                context.system_dims,
                context.n_clbits,
                initial_state=initial_state,
            )
            assert self._evolution_state is not None
            self._evolution_state = _run_one_shot(
                payload,
                self._evolution_state,
                context,
                np.random.default_rng(seed_sequence),
            )
            clbits = self._evolution_state.classical.clbits
            snapshots.append(
                tuple(clbits) if clbits is not None else (0,) * context.n_clbits
            )
        return snapshots

    def apply(self, step: ApplyMatrixStep) -> None:
        """Apply one gate directly; normal execution binds functions once per plan."""
        execution = self._bind_kernels_to_python_workflow((step,))
        self._state = execution.apply_matrix(self.state, step, self._dims)

    def apply_channel(self, step: ApplyChannelStep, rng: np.random.Generator) -> None:
        """Apply one channel directly using the representation's numerical function."""
        execution = self._bind_kernels_to_python_workflow((step,))
        self._state = execution.apply_channel(self.state, step, self._dims, rng)

    def _expectation_values(
        self,
        state: QuantumDataT,
        observables: Sequence[tuple[tuple[float, tuple[tuple[int, str], ...]], ...]],
        *,
        policy: ExecutionPolicy,
    ) -> tuple[float, ...]:
        """Evaluate Pauli sums on native state without modifying it.

        Factors use engine subsystem indices. Return host scalars and keep
        numerical work within the runtime's execution scope.
        """
        raise NotImplementedError

    def export_state(self) -> QuantumDataT:
        """Return state data in runtime-native storage for further processing.

        The default borrows internal data without copying or host transfer.
        Overrides convert the representation only when needed. Callers copy or
        transfer the result when they need independent storage or host data.
        """
        return self.state
