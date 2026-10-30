"""Numba matrix-family engines.

`NumbaSVEngine` (statevector) and `NumbaDMEngine` (density matrix) reuse
shared payload preparation and Python execution from ``MatrixEngine`` and
ndarray initialization and expectations from ``_NdarrayEngine``,
while replacing numeric kernels with Numba-jitted loops. `NumbaUnitaryEngine`
and `NumbaSuperopEngine` then reuse *those* kernels for the operator
representations, with separate preparation for their compiled workflows. All four are reachable via
``Simulator(method=..., runtime="numba")``.

Kernel selection is key-driven: an `ApplyMatrixStep` carries the canonical
identity of the implementation that produced its matrix (``kernel_key``), and
each engine resolves which specialized kernel a step gets once per plan - by
declared identity or one `_classify_matrix` content scan, never per application
or per shot. `_KERNEL_SPECS` is the statevector key-to-structure table; the
density matrix derives its table from the same declarations because the
Kronecker product preserves structure (see below).

- `NumbaSVEngine` selects Numba gate, measurement, and reset functions
  and retains the channel matrix fallback (`_apply_unclassified_matrix`).
  Reset uses direct digit shifts shared with the compiled trajectory path.
- `NumbaDMEngine` selects Numba gate and measurement functions; channels
  share the gate superoperator application. It reuses the *same* coset-walk
  kernels as the statevector path by viewing ``rho`` (shape ``(size, size)``) as a flat
  vector over a doubled ``2n``-subsystem system: ``n`` bra subsystems (strides
  ``prod(dims[:q])``) then ``n`` ket subsystems (strides ``size *
  prod(dims[:q])``). The sandwich ``M rho M^dagger`` is the single
  super-operator ``kron(M, conj(M))`` on ``vec(rho)``, and a channel is
  ``sum_i kron(K_i, conj(K_i))`` - each one coset walk over the combined
  ket+bra super-target. The Kronecker product preserves structure, so a gate's
  declared identity transfers to its super-operator: a declared-dense key skips
  the scan, everything else takes one scan of the super-operator, cached per
  step. ``4^n`` is memory-bound and each gate is one pass, so DM parallelizes
  later than the statevector path (`_MIN_SIZE_TO_THREAD_DM`). Reset stays the
  NumPy partial-trace numerical function.

`NumbaFixedPhotonEngine` is `NumpyFixedPhotonEngine` with Numba gate and
measurement functions that walk the rows one at a time; `LOSimulator` alone
selects it.

Kraus arithmetic lives in ``numerics.numba_kernels._channel``. Channel/readout
packing and classical readout sequencing live in ``workflows.numba_workflow``.
Numerical modules do not import engines or workflows; compiled callers import
their numerical functions directly at module scope.

The compiled multi-shot statevector channel path weighs branches the way Aer's trajectory
sampler does, not the way the NumPy reference does: instead of materializing
every ``K_i|psi>`` and norming it, it forms the targets' reduced density matrix
``rho_T`` once (`_reduced_density`) and reads each branch probability off it as
``Tr(K_i^dagger K_i rho_T)``, then applies only the chosen operator (see
`_apply_channel_step`). A mathematically equal but numerically distinct estimator, so
it stays seed-reproducible and agrees with NumPy in distribution, not per-seed
bit-for-bit.

The operator engines do not apply gates one at a time. An operator buffer is
``(rows, columns)`` and every step acts on the row index alone, so columns are
independent and a whole plan runs inside one parallel region split over column
blocks (``workflows.numba_workflow.operators``). Those loops keep the per-step
accumulation order, so they are bit-identical to the per-gate coset kernels.
When ``fusion=True`` is requested for a supported method, a sufficiently large
plan may merge adjacent operator payloads into wider ones. That rewrite is
equal to the unfused plan only to floating-point tolerance.

Shot threading uses the compiled loop in ``workflows.numba_workflow.trajectories``, while kernel
threading partitions state amplitudes or cosets, or operator columns. Serial
inner kernels use one Numba thread unless the compiled shot loop owns the outer
pool. Process-backed replay uses a one-thread child policy, so process and Numba
thread pools never nest. The compiled multi-shot path is independent of the
optional fusion rewrite.

The RNG draw stays in NumPy - a ``np.random.Generator`` cannot cross into
nopython code - so uniforms are drawn with ``rng`` and the inverse-CDF search
runs in Numba. This consumes the stream identically to ``rng.choice``, so
counts stay reproducible per engine (float-summation differences from NumPy
mean they are not bit-identical across engines - the documented contract,
see `numpy_engine.py`).

Numba compiles kernels lazily on first call. This module is never imported from
``fatqat.simulator._engine``'s ``__init__``; import it explicitly.

Conventions match `numpy_engine.py`: private little-endian engine indexing (engine
subsystem ``q`` has place value ``prod(dims[:q])``) and a local gate matrix
whose most-significant index digit is ``targets[0]``. The simulator maps public
subsystems before lowering; reversing an exported buffer here would apply the
mapping twice and allocate an unnecessary full-result copy.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from math import prod
from dataclasses import replace
from functools import partial

import numpy as np

from ..._execution_contract import (
    RawResult,
    _SimulationConfig as SimulationConfig,
    _RunContext,
    _KernelCapabilities,
    _EngineCapabilities,
    _QuantumCapabilities,
    _TrajectoryCapabilities,
)
from ...._backends.steps import (
    ApplyChannelStep,
    ApplyMatrixStep,
    ApplyParametricMatrixStep,
    MeasurementStep,
    ResetStep,
    ResolvedStep,
)
from ....result import reduce_to_counts
from ....errors import BackendValidationError
from .._execution_policy import (
    _ExecutionPolicy as ExecutionPolicy,
    _resolve_kernel_scheduling,
    _plan_python_execution,
    _resolve_python_workflow_scheduling,
    _select_execution_path,
    _validate_shot_parallelism,
    _explicit_thread_worker_ceiling,
    _adaptive_thread_worker_ceiling,
)
from .._numba_scope import _kernel_thread_workers, _thread_scope
from ..numerics.numba_kernels._fusion import (
    _MIN_SIZE_TO_FUSE,
    _fuse_gate_channels,
    _fuse_operator_payloads,
)
from ..numerics.numba_kernels._measurement import (
    _sv_probabilities_kernel,
    _dm_probabilities_kernel,
    _project_statevector,
    _project_density_matrix,
    _sample_basis_indices,
    _reset_statevector,
)
from ..numerics.numba_kernels._matrix_cache import (
    _MatrixStructureCache,
    _SuperoperatorCache,
)
from ..numerics.numba_kernels._layouts import (
    _MAX_THREADS,
    _MIN_SIZE_TO_THREAD,
    _ApplyLayoutCache,
    _build_unitary_apply_layout,
    _build_superoperator_apply_layout,
)
from ..numerics.numba_kernels._apply_matrix_to_state import (
    _apply_matrix_statevector,
    _apply_superoperator_to_state,
)
from ..numerics.numba_kernels._channel import (
    _apply_channel_statevector,
    _apply_channel_density_matrix,
)
from ..numerics.numba_kernels._fixed_photon import (
    _apply_matrix_fixed_photon_numba,
    _fixed_photon_levels_numba,
    _project_fixed_photon_numba,
)
from ..numerics.numpy_kernels._fixed_photon import _fixed_photon_probabilities
from ..numerics.numpy_kernels._reset import _reset_density_matrix, _reset_superoperator
from ..binding import _bind_parameter_batch
from ..state import EvolutionState
from .ndarray_engine import _NdarrayEngine
from .numpy_engine import NumpyFixedPhotonEngine
from ..numerics.numpy_kernels._state import (
    _allocate_statevector,
    _allocate_density_matrix,
    _allocate_unitary,
    _allocate_superoperator,
    _export_superoperator,
)
from ..numerics.numpy_kernels._channel import _ChannelRouteCache
from ..numerics.numpy_kernels._reset import _ResetChannelCache
from ..workflows.python_workflow import _MeasurementKernels, _PythonExecution
from ..workflows.numba_workflow.operators import (
    _operator_chunks,
    _pack_operator_plan,
    _NumbaOperatorExecution,
    _execute_operator_plan,
)
from ..workflows.numba_workflow.single_pass import (
    _NumbaSinglePassExecution,
    _evolve_sv_batch,
    _pack_sv_batch,
    _placeholder_plan,
)
from ..workflows.numba_workflow.trajectories import (
    _NumbaTrajectoryExecution,
    _compile_dynamic_plan,
    _execute_sv_trajectories,
    _patch_gate_matrices,
    _plan_compilable,
)

# Bytes of evolved statevectors held at once by a compiled single-pass batch.
_BATCH_STATE_BYTES = 256 * 2**20


class NumbaSVEngine(_NdarrayEngine):
    """State-vector engine with Numba-jitted numeric kernels."""

    _capabilities = _EngineCapabilities(
        quantum=_QuantumCapabilities(
            "statevector", supports_nonunitary=True, nonunitary_is_stochastic=True
        ),
        trajectory=_TrajectoryCapabilities(classical_register=True, occupancy=True),
        kernels=_KernelCapabilities(
            supports_kernel_threads=True,
            thread_capacity=_MAX_THREADS,
            supports_fusion=False,
        ),
        supports_process_shots=True,
    )

    def _select_policy(
        self,
        plan: Sequence[ResolvedStep],
        *,
        context: _RunContext,
        simulation_config: SimulationConfig,
        initial_state: EvolutionState[np.ndarray],
    ) -> ExecutionPolicy:
        """Choose a compiled workflow or Python sequencing with Numba kernels.

        Eligible repeated trajectories use the compiled loop with serial inner
        kernels and serial or threaded shots. A parameter batch on the
        single-pass path, too small for threaded kernels, whose parametric
        gates all have compiled formulas, evolves its entries in a compiled
        call and is sampled by the Python workflow; packing costs more than it
        saves for a single plan. Other requests use shared Python
        scheduling, including process replay and threading within kernels.
        ``plan`` may hold parametric steps; it is judged as its placeholder plan.
        """
        # Only exactly-registered built-in gates have compiled formulas.
        parametric = [
            step for step in plan if isinstance(step, ApplyParametricMatrixStep)
        ]
        compiled_batch = bool(parametric) and all(
            step.implementation._batch_builder(step.operation) is not None
            for step in parametric
        )
        plan = _placeholder_plan(plan, context.system_dims)
        execution_path = _select_execution_path(
            plan,
            quantum_capabilities=self.capabilities.quantum,
            initial_state=initial_state,
            classical_requested=context.request.final_classical_state,
        )
        _validate_shot_parallelism(
            simulation_config,
            execution_path=execution_path,
            counts_requested=context.request.counts,
            state_requested=(
                context.request.final_state or context.request.final_classical_state
            ),
            shots=context.shots,
        )
        shot_request = simulation_config.shot_parallelism
        kernel_request = simulation_config.kernel_parallelism
        # Explicit processes and kernel threads use Python replay. With serial
        # shots, preserve adaptive kernel execution unless kernels are also serial.
        use_python = (
            shot_request == "processes"
            or kernel_request == "threads"
            or (shot_request == "serial" and kernel_request == "auto")
            or context.shots <= 1
            or execution_path
            != "per_shot"  # Plan not supported by compiled trajectories.
            or not context.request.counts
            or context.request.final_state
            or context.request.final_classical_state
            or initial_state.classical.occupied is not None
            or not _plan_compilable(plan)
        )
        if use_python:
            policy = _resolve_python_workflow_scheduling(
                simulation_config,
                execution_path=execution_path,
                counts_requested=context.request.counts,
                state_requested=(
                    context.request.final_state or context.request.final_classical_state
                ),
                capabilities=self.capabilities.kernels,
                supports_process_shots=self.capabilities.supports_process_shots,
                shots=context.shots,
            )
            if (
                compiled_batch
                and execution_path == "single_pass"
                and policy.shot_strategy != "processes"
                and policy.kernel_strategy != "threads"
                and prod(context.system_dims) < _MIN_SIZE_TO_THREAD
                and _plan_compilable(plan)
            ):
                return replace(policy, workflow="numba_single_pass")
            return policy

        # This compiled workflow owns the shot loop and calls serial kernels.
        # Auto prefers this loop even when the worker ceiling makes it serial.
        kernels = self.capabilities.kernels
        if shot_request == "threads":
            workers = _explicit_thread_worker_ceiling(
                simulation_config.max_workers, kernels
            )
            if workers < 2:
                raise BackendValidationError(
                    "shot_parallelism='threads' requires an effective thread "
                    "capacity of at least two"
                )
            shot_strategy = "threads"
        elif shot_request == "serial":
            shot_strategy, workers = "serial", 1
        else:
            workers = _adaptive_thread_worker_ceiling(
                simulation_config.max_workers, kernels
            )
            if kernels.thread_capacity >= 2 and (workers is None or workers > 1):
                shot_strategy = "threads"
            else:
                shot_strategy, workers = "serial", 1
        return ExecutionPolicy(
            execution_path=execution_path,
            shot_strategy=shot_strategy,
            kernel_strategy="serial",
            worker_limit=workers,
            fusion=simulation_config.fusion,
            workflow="numba_trajectories",
        )

    _allocate = staticmethod(_allocate_statevector)

    def __init__(self, name: str = "numba-sv"):
        super().__init__(name)
        self._channel_routes = _ChannelRouteCache()
        self._matrix_structures = _MatrixStructureCache()
        self._layout_cache = _ApplyLayoutCache(_build_unitary_apply_layout)

    def _bind_kernels_to_python_workflow(
        self,
        plan: tuple[ResolvedStep, ...],
        measurements: tuple[tuple[int, int], ...] = (),
        *,
        policy: ExecutionPolicy | None = None,
    ) -> _PythonExecution:
        # Each execution owns its mapping, while unchanged numerical layouts
        # are reused across repeated runs and sweep rows.
        operator = self._state_field == "unitary"
        targets = {
            tuple(step.target_indices)
            for step in plan
            if isinstance(step, (ApplyMatrixStep, ApplyChannelStep))
        }
        workers = _kernel_thread_workers(policy) if policy else _MAX_THREADS
        layouts = self._layout_cache.for_targets(
            self._dims,
            targets,
            batch_columns=operator,
            max_workers=workers,
            force_parallel=policy is not None and policy.kernel_strategy == "threads",
        )
        # pylint: disable-next=fixme
        # TODO Those None if operator indicate that the operator path does not support
        # measurements or resets. Should be implemented in the operator engines.
        return _PythonExecution(
            plan=plan,
            deferred_measurements=measurements,
            apply_matrix=partial(
                _apply_matrix_statevector,
                structures=self._matrix_structures,
                layouts=layouts,
            ),
            apply_channel=(
                None
                if operator
                else partial(
                    _apply_channel_statevector,
                    routes=self._channel_routes,
                    layouts=layouts,
                )
            ),
            measurement=(
                None
                if operator
                else _MeasurementKernels(
                    _sv_probabilities_kernel,
                    _project_statevector,
                    _sample_basis_indices,
                )
            ),
            reset=None if operator else _reset_statevector,
        )

    def _retain_step_caches(self, plan: tuple[ResolvedStep, ...]) -> None:
        self._channel_routes.retain(plan)
        self._matrix_structures.retain(plan)

    def _execution_scope(self, policy: ExecutionPolicy):
        return _thread_scope(policy)

    def prepare_payload(
        self,
        plan: tuple[ResolvedStep, ...],
        *,
        system_dims: tuple[int, ...],
        n_clbits: int,
        policy: ExecutionPolicy,
    ) -> _PythonExecution | _NumbaTrajectoryExecution | _NumbaSinglePassExecution:
        if policy.workflow == "python":
            return super().prepare_payload(
                plan, system_dims=system_dims, n_clbits=n_clbits, policy=policy
            )
        if policy.workflow == "numba_single_pass":
            template = _placeholder_plan(plan, system_dims)
            # Prepare sampling first, so packing leaves the gate caches in place.
            sampling = super().prepare_payload(
                tuple(step for step in template if isinstance(step, MeasurementStep)),
                system_dims=system_dims,
                n_clbits=n_clbits,
                policy=policy,
            )
            self._retain_step_caches(template)
            batch = _pack_sv_batch(
                plan,
                template,
                self._dims,
                self._build_apply_layout,
                self._matrix_structures.resolve,
            )
            return _NumbaSinglePassExecution(batch, sampling)
        self.configure_system(system_dims, n_clbits)
        self._retain_step_caches(plan)
        return _compile_dynamic_plan(
            list(plan),
            self._dims,
            self._build_apply_layout,
            self._matrix_structures.resolve,
        )

    def execute_payload(
        self,
        context: _RunContext,
        payload: _PythonExecution | _NumbaTrajectoryExecution,
        policy: ExecutionPolicy,
        *,
        initial_state: EvolutionState[np.ndarray],
    ) -> RawResult[np.ndarray]:
        if policy.workflow == "python":
            return super().execute_payload(
                context, payload, policy, initial_state=initial_state
            )
        self.configure_system(context.system_dims, context.n_clbits)
        with self._execution_scope(policy):
            rows = _execute_sv_trajectories(
                context, payload, initial_state=initial_state
            )
            outcome_keys, outcome_counts = reduce_to_counts(rows)
            return RawResult(outcome_keys=outcome_keys, outcome_counts=outcome_counts)

    def execute(
        self,
        plan: tuple[ResolvedStep, ...],
        *,
        context: _RunContext,
        simulation_config: SimulationConfig,
        initial_state: EvolutionState[np.ndarray],
    ) -> Iterator[RawResult[np.ndarray]]:
        """Execute a parameter batch, packing compiled workflows only once.

        Batch entries differ only in their parametric gate matrices, so the
        policy is planned once from a placeholder plan. A single-pass batch
        too small for threaded kernels, whose parametric gates all have
        compiled formulas, binds and evolves its entries inside one compiled
        call per chunk; the Python single-pass workflow samples each state.
        On the compiled trajectory workflow the first entry is packed as
        usual and later entries replace only their gate matrices. Other
        batches execute entry by entry. Results are yielded in batch order,
        as in `MatrixEngine.execute`.
        """
        positions = tuple(
            index
            for index, step in enumerate(plan)
            if isinstance(step, ApplyParametricMatrixStep)
        )
        policy = self._select_policy(
            plan,
            context=context,
            simulation_config=simulation_config,
            initial_state=initial_state,
        )
        # Compiled single pass: evolve blocks of entries in one compiled call,
        # then sample and yield each entry's state; a block is freed once all
        # of its entries are consumed.
        if policy.workflow == "numba_single_pass":
            payload = self.prepare_payload(
                plan,
                system_dims=context.system_dims,
                n_clbits=context.n_clbits,
                policy=policy,
            )
            n_entries = payload.batch.values.shape[0]
            chunk = max(1, _BATCH_STATE_BYTES // (16 * prod(context.system_dims)))
            for start in range(0, n_entries, chunk):
                with self._execution_scope(policy):
                    states = _evolve_sv_batch(
                        payload.batch,
                        slice(start, start + chunk),
                        context,
                        initial_state=initial_state,
                    )
                for state in states:
                    yield super().execute_payload(
                        context,
                        payload.sampling,
                        policy,
                        initial_state=EvolutionState(state, initial_state.classical),
                    )
            return

        # Every other workflow binds and runs one entry at a time, yielding each
        # result before the next entry starts. Compiled trajectories pack the
        # first entry and patch later entries' gate matrices; the Python
        # workflow executes each bound plan like an ordinary run.
        packed: _NumbaTrajectoryExecution | None = None
        for bound in _bind_parameter_batch(plan, context.system_dims):
            if policy.workflow != "numba_trajectories":
                yield self._dispatch_by_shot_strategy(
                    bound, context, policy, initial_state
                )
                continue
            if packed is None:
                payload = packed = self.prepare_payload(
                    bound,
                    system_dims=context.system_dims,
                    n_clbits=context.n_clbits,
                    policy=policy,
                )
            else:
                # Release the previous entry's gate steps before resolving these.
                self._retain_step_caches(bound)
                payload = _patch_gate_matrices(
                    packed, bound, positions, self._matrix_structures.resolve
                )
            yield self.execute_payload(
                context, payload, policy, initial_state=initial_state
            )

    def _build_apply_layout(
        self,
        targets: tuple[int, ...],
        *,
        max_workers: int = _MAX_THREADS,
        force_parallel: bool = False,
    ) -> tuple:
        """Strided-block kernel layout for ``targets`` over the physical dims."""
        return _build_unitary_apply_layout(
            self._dims, targets, max_workers=max_workers, force_parallel=force_parallel
        )


# --- density-matrix kernels ---


class NumbaDMEngine(_NdarrayEngine):
    """Density-matrix engine with Numba-jitted, key-driven numeric kernels.

    Binds Numba numerical functions to the shared Python workflow. Execution
    runs within the selected thread scope; density-matrix fusion is unsupported.
    Reset keeps the NumPy partial-trace numerical function (see
    the module docstring).

    Both a gate and a channel apply as one super-operator pass over
    ``vec(rho)`` (module docstring). The superoperator cache is the density-matrix
    analog of the statevector matrix cache: built and classified once
    per plan step, key-aware for gates, content-scanned for channels.
    """

    _capabilities = _EngineCapabilities(
        quantum=_QuantumCapabilities(
            "density_matrix", supports_nonunitary=True, nonunitary_is_stochastic=False
        ),
        trajectory=_TrajectoryCapabilities(classical_register=True, occupancy=True),
        kernels=_KernelCapabilities(
            supports_kernel_threads=True,
            thread_capacity=_MAX_THREADS,
            supports_fusion=False,
        ),
        supports_process_shots=True,
    )

    _allocate = staticmethod(_allocate_density_matrix)

    def _select_policy(
        self,
        plan: Sequence[ResolvedStep],
        *,
        context: _RunContext,
        simulation_config: SimulationConfig,
        initial_state: EvolutionState[np.ndarray],
    ) -> ExecutionPolicy:
        return _plan_python_execution(
            plan,
            context=context,
            simulation_config=simulation_config,
            initial_state=initial_state,
            capabilities=self.capabilities,
        )

    def __init__(self, name: str = "numba-dm"):
        super().__init__(name)
        self._superoperators = _SuperoperatorCache()
        self._layout_cache = _ApplyLayoutCache(_build_superoperator_apply_layout)

    def _bind_kernels_to_python_workflow(
        self,
        plan: tuple[ResolvedStep, ...],
        measurements: tuple[tuple[int, int], ...] = (),
        *,
        policy: ExecutionPolicy | None = None,
    ) -> _PythonExecution:
        operator = self._state_field == "superop"
        targets = {
            tuple(step.target_indices)
            for step in plan
            if isinstance(step, (ApplyMatrixStep, ApplyChannelStep))
        }
        workers = _kernel_thread_workers(policy) if policy else _MAX_THREADS
        layouts = self._layout_cache.for_targets(
            self._dims,
            targets,
            batch_columns=operator,
            max_workers=workers,
            force_parallel=policy is not None and policy.kernel_strategy == "threads",
        )
        return _PythonExecution(
            plan=plan,
            deferred_measurements=measurements,
            apply_matrix=partial(
                _apply_superoperator_to_state,
                superoperators=self._superoperators,
                layouts=layouts,
            ),
            apply_channel=partial(
                _apply_channel_density_matrix,
                superoperators=self._superoperators,
                layouts=layouts,
            ),
            measurement=(
                None
                if operator
                else _MeasurementKernels(
                    _dm_probabilities_kernel,
                    _project_density_matrix,
                    _sample_basis_indices,
                )
            ),
            reset=(
                partial(_reset_superoperator, reset_channels=self._reset_channels)
                if operator
                else _reset_density_matrix
            ),
        )

    def _retain_step_caches(self, plan: tuple[ResolvedStep, ...]) -> None:
        self._superoperators.retain(plan)

    def _execution_scope(self, policy: ExecutionPolicy):
        return _thread_scope(policy)


# --- operator engines ---
#
# An operator is its state twin evolved on many columns at once. A run takes the
# compiled whole-plan path; a single application falls back to the per-gate coset
# kernels over a column-batched apply plan, which is a `columns`-sized
# never-targeted subsystem prepended to the dims.


class _NumbaOperatorRunMixin:
    """Compiled whole-plan execution for the Numba operator engines.

    Leaves declare their capabilities and supply `_operator_row_dims` and
    `_operator_payloads`.
    """

    def _select_policy(
        self,
        plan: Sequence[ResolvedStep],
        *,
        context: _RunContext,
        simulation_config: SimulationConfig,
        initial_state: EvolutionState[np.ndarray],
    ) -> ExecutionPolicy:
        """Choose compiled operator sequencing with serial or threaded kernels.

        Operator evolution has no independent shots. Shared scheduling resolves
        the kernel controls and worker limit for the whole-plan operator loop.
        """
        execution_path = "operator"
        _validate_shot_parallelism(
            simulation_config,
            execution_path=execution_path,
            counts_requested=context.request.counts,
            state_requested=context.request.final_state,
            shots=context.shots,
        )
        kernel_strategy, workers = _resolve_kernel_scheduling(
            simulation_config.kernel_parallelism,
            max_workers=simulation_config.max_workers,
            capabilities=self.capabilities.kernels,
        )
        return ExecutionPolicy(
            execution_path=execution_path,
            shot_strategy="none",
            kernel_strategy=kernel_strategy,
            worker_limit=workers,
            fusion=simulation_config.fusion,
            workflow="numba_operator",
        )

    def prepare_payload(
        self,
        plan: tuple[ResolvedStep, ...],
        *,
        system_dims: tuple[int, ...],
        n_clbits: int,
        policy: ExecutionPolicy,
    ) -> _NumbaOperatorExecution:
        """Resolve, optionally fuse, and pack the operator plan once."""
        assert policy.workflow == "numba_operator"
        self.configure_system(system_dims, n_clbits)
        execution_plan = self._operator_execution_plan(plan, policy)
        self._retain_step_caches(execution_plan)
        payloads = self._operator_payloads(list(execution_plan))
        row_dims = self._operator_row_dims()
        n_columns = prod(row_dims) if row_dims else 1
        flat_size = n_columns * n_columns
        if policy.fusion and flat_size >= _MIN_SIZE_TO_FUSE:
            payloads = _fuse_operator_payloads(payloads, row_dims)
        if not payloads:
            return _NumbaOperatorExecution(None, 0, n_columns, 1)
        packed, scratch_rows = _pack_operator_plan(payloads, row_dims, n_columns)
        n_chunks = _operator_chunks(len(payloads), flat_size, n_columns, policy)
        return _NumbaOperatorExecution(packed, scratch_rows, n_columns, n_chunks)

    def _operator_execution_plan(
        self, plan: tuple[ResolvedStep, ...], policy: ExecutionPolicy
    ) -> tuple[ResolvedStep, ...]:
        del policy
        return plan

    def execute_payload(
        self,
        context: _RunContext,
        payload: _NumbaOperatorExecution,
        policy: ExecutionPolicy,
        *,
        initial_state: EvolutionState[np.ndarray],
    ) -> RawResult[np.ndarray]:
        """Execute one already-packed operator payload without replanning."""
        assert policy.workflow == "numba_operator"
        assert policy.shot_strategy == "none"
        assert policy.execution_path == "operator"
        self.configure_system(context.system_dims, context.n_clbits)
        with self._execution_scope(policy):
            self.initialize(
                self._dims,
                self._n_clbits,
                initial_state=initial_state,
            )
            self._state = _execute_operator_plan(self.state, payload)
            state = self.export_state() if context.request.final_state else None
            return RawResult(state=state)

    def _operator_row_dims(self) -> tuple[int, ...]:
        """Subsystem dimensions the operator's row index decomposes into."""
        raise NotImplementedError

    def _operator_payloads(self, plan: list) -> list[tuple]:
        """Resolve an already-prepared ``plan`` into numeric payloads."""
        raise NotImplementedError


class NumbaUnitaryEngine(_NumbaOperatorRunMixin, NumbaSVEngine):
    """Unitary engine with Numba-jitted numeric kernels.

    ``U`` is ``size`` statevector columns over a row index that decomposes into
    the plain system dims; every step is a gate.
    """

    _capabilities = _EngineCapabilities(
        quantum=_QuantumCapabilities(
            # Retain the SV branch classification; non-unitary plans are rejected.
            "unitary",
            supports_nonunitary=False,
            nonunitary_is_stochastic=True,
        ),
        trajectory=_TrajectoryCapabilities(),
        kernels=_KernelCapabilities(
            supports_kernel_threads=True,
            thread_capacity=_MAX_THREADS,
            supports_fusion=True,
        ),
        supports_process_shots=False,
    )

    _allocate = staticmethod(_allocate_unitary)

    def __init__(self, name: str = "numba-unitary"):
        super().__init__(name)

    def _operator_row_dims(self) -> tuple[int, ...]:
        return self._dims

    def _operator_payloads(self, plan: list) -> list[tuple]:
        # A unitary plan holds only gates, so it reaches this translator as lowered.
        payloads = []
        for step in plan:
            assert isinstance(
                step, ApplyMatrixStep
            ), "unitary execution accepts only matrix steps"
            code, columns, values = self._matrix_structures.resolve(step)
            payloads.append(
                (step.matrix, code, columns, values, None, tuple(step.target_indices))
            )
        return payloads


class NumbaSuperopEngine(_NumbaOperatorRunMixin, NumbaDMEngine):
    """Super-operator engine with Numba-jitted numeric kernels.

    ``S`` is ``size**2`` density-matrix columns over a row index that decomposes
    into the doubled ``bra + ket`` dims. Every step - gate, channel, or reset -
    resolves to one super-operator on the combined ket+bra super-target.
    """

    _capabilities = _EngineCapabilities(
        quantum=_QuantumCapabilities(
            "superop", supports_nonunitary=True, nonunitary_is_stochastic=False
        ),
        trajectory=_TrajectoryCapabilities(),
        kernels=_KernelCapabilities(
            supports_kernel_threads=True,
            thread_capacity=_MAX_THREADS,
            supports_fusion=True,
        ),
        supports_process_shots=False,
    )

    _allocate = staticmethod(_allocate_superoperator)

    def __init__(self, name: str = "numba-superop"):
        super().__init__(name)
        self._reset_channels = _ResetChannelCache()

    def export_state(self) -> np.ndarray:
        """Convert superoperator data to public column stacking."""
        return _export_superoperator(self.state, self._dims)

    def _retain_step_caches(self, plan: tuple[ResolvedStep, ...]) -> None:
        # Reset payloads resolve synthesized channels, not the ResetSteps themselves.
        reset_steps = tuple(
            self._reset_channels.resolve(index, self._dims[index])
            for step in plan
            if isinstance(step, ResetStep)
            for index in step.reset_indices
        )
        super()._retain_step_caches(plan + reset_steps)

    def _operator_row_dims(self) -> tuple[int, ...]:
        return self._dims + self._dims

    def _operator_payloads(self, plan: list) -> list[tuple]:
        n = len(self._dims)
        payloads = []
        for step in plan:
            # A reset is the Kraus channel sum_k |0><k|.
            resolved_steps = (
                [
                    self._reset_channels.resolve(index, self._dims[index])
                    for index in step.reset_indices
                ]
                if isinstance(step, ResetStep)
                else [step]
            )
            for resolved in resolved_steps:
                superop, code, columns, values, sparse = self._superoperators.resolve(
                    resolved
                )
                targets = tuple(resolved.target_indices)
                # Ket group first, so the local index is ``ket * D + bra``.
                row_targets = tuple(n + t for t in targets) + targets
                payloads.append((superop, code, columns, values, sparse, row_targets))
        return payloads

    def _operator_execution_plan(
        self, plan: tuple[ResolvedStep, ...], policy: ExecutionPolicy
    ) -> tuple[ResolvedStep, ...]:
        if not policy.fusion:
            return plan
        return tuple(_fuse_gate_channels(list(plan)))


# --- fixed-photon engine ---


class NumbaFixedPhotonEngine(NumpyFixedPhotonEngine):
    """Fixed-photon engine with Numba-jitted gate and measurement functions.

    Gates and projections split the rows across kernel threads, like the
    statevector coset walks.
    """

    _capabilities = replace(
        NumpyFixedPhotonEngine._capabilities,
        kernels=_KernelCapabilities(
            supports_kernel_threads=True,
            thread_capacity=_MAX_THREADS,
            supports_fusion=False,
        ),
    )

    def __init__(self, name: str = "numba-fixed-photon"):
        super().__init__(name)

    def _execution_scope(self, policy: ExecutionPolicy):
        return _thread_scope(policy)

    def _bind_kernels_to_python_workflow(
        self,
        plan: tuple[ResolvedStep, ...],
        measurements: tuple[tuple[int, int], ...] = (),
        *,
        policy: ExecutionPolicy | None = None,
    ) -> _PythonExecution:
        threads = {
            "workers": _kernel_thread_workers(policy) if policy else _MAX_THREADS,
            "force_parallel": policy is not None
            and policy.kernel_strategy == "threads",
        }
        return _PythonExecution(
            plan=plan,
            deferred_measurements=measurements,
            apply_matrix=partial(_apply_matrix_fixed_photon_numba, **threads),
            apply_channel=None,
            measurement=_MeasurementKernels(
                _fixed_photon_probabilities,
                partial(_project_fixed_photon_numba, **threads),
                _sample_basis_indices,
                _fixed_photon_levels_numba,
            ),
            reset=None,
        )
