"""NumPy engines for the matrix backend family.

`NumpySVEngine` (statevector) and `NumpyDMEngine` (density matrix) are the
two *state* `MatrixEngine` implementations. `MatrixEngine` owns shared payload
preparation and Python execution; `_NdarrayEngine` supplies NumPy
initialization and expectations. Measurement sequencing uses explicit state
and runtime-selected numerical functions. Each leaf contributes its numerical
operations and explicit capabilities. Engine-owned selection determines the
semantic execution shape.

`NumpyUnitaryEngine` and `NumpySuperopEngine` are the two *operator* engines:
they compute the program's map rather than a state under it. Each is its state
twin evolved on many columns at once - a unitary is ``size`` statevector
columns (column ``j`` is ``U|j>``), a super-operator is ``size**2``
density-matrix columns (column ``b`` is the image of basis matrix ``E_b``).
Their capabilities reject sampling; shared execution selects the operator pass.

Conventions:

- private little-endian engine order: engine subsystem ``q`` has place value
  ``prod(dims[:q])``. The simulator maps public subsystems into this order, so
  the native flat buffer is already public-order; exporting it with another
  subsystem reversal would be wrong and would add a full-result copy.
- an ``ApplyMatrixStep``'s ``target_indices`` map to the matrix's local index
  with ``target_indices[0]`` as the most-significant digit.
- a density matrix reshaped to ``reversed(dims) + reversed(dims)`` exposes ket
  axes ``[0, n)`` and bra axes ``[n, 2n)``; subsystem ``q``'s ket axis is
  ``n - 1 - q`` and its bra axis is ``2n - 1 - q``.

Semantics differences:

- statevector evolution is ``|psi'> = U_T |psi>``; density-matrix evolution is
  ``rho' = U_T rho U_T^dagger``. Neither materializes the global ``U_T``.
- measurement is trajectory-style on both: Born sampling, projective collapse,
  normalization.
- reset samples a branch on the statevector (one rng draw), so any reset forces
  the per-shot path; on the density matrix it is the deterministic
  partial-trace channel (no rng draw), so an unconditional reset stays on the
  fast path. Per-shot rng streams are therefore not draw-for-draw aligned across
  the two, so counts for a fixed seed are reproducible per engine, not across
  engines.
- a channel (``ApplyChannelStep``) follows the same split as reset: the
  statevector samples one Kraus branch per occurrence (quantum-jump
  unravelling, one rng draw), forcing the per-shot path; the density matrix
  applies the exact Kraus sum ``sum_i K_i rho K_i^dagger`` (no rng draw), so
  an unconditional channel stays on the fast path.
- a channel of scaled unitaries (every Pauli channel; also `Depolarizing` and
  `PhaseDamping` in any dimension) has state-independent branch probabilities,
  so the statevector draws its branch from that fixed vector and applies only
  the operator it drew. One rng draw either way, so which sampler a channel
  takes affects neither the stream nor path classification.
- measurement reporting is identical on both: the collapse keeps the physical
  outcome, maps it to a reported digit, then optionally resamples that digit
  through ``MeasurementStep.confusions``. It never affects path
  classification. On the per-shot path feedforward conditions read the
  (possibly confused) reported value.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

import numpy as np

from ..._execution_contract import (
    _RunContext,
    _SimulationConfig as SimulationConfig,
    _KernelCapabilities,
    _EngineCapabilities,
    _QuantumCapabilities,
    _TrajectoryCapabilities,
)
from ...._backends.steps import (
    ResolvedStep,
)
from .._execution_policy import (
    _ExecutionPolicy as ExecutionPolicy,
    _plan_python_execution,
)
from ..state import EvolutionState
from .ndarray_engine import _NdarrayEngine
from ..numerics.numpy_kernels._measurement import (
    _statevector_probabilities,
    _density_matrix_probabilities,
    _project_statevector,
    _project_density_matrix,
    _sample_basis_indices,
)
from ..numerics.numpy_kernels._channel import (
    _apply_channel_statevector,
    _apply_channel_density_matrix,
    _apply_channel_superoperator,
    _ChannelRouteCache,
)
from ..numerics.numpy_kernels._reset import (
    _ResetChannelCache,
    _reset_density_matrix,
    _reset_statevector,
    _reset_superoperator,
)
from ..numerics.numpy_kernels._state import (
    _allocate_statevector,
    _allocate_density_matrix,
    _allocate_unitary,
    _allocate_superoperator,
    _export_superoperator,
)
from ..numerics.numpy_kernels._apply import (
    _apply_matrix_statevector,
    _apply_matrix_density_matrix,
    _apply_matrix_unitary,
    _apply_matrix_superoperator,
)
from ..workflows.python_workflow import _MeasurementKernels, _PythonExecution

# --- statevector engine ---


class NumpySVEngine(_NdarrayEngine):
    """State-vector engine: evolves ``|psi>`` as a flat little-endian array."""

    _capabilities = _EngineCapabilities(
        quantum=_QuantumCapabilities(
            "statevector", supports_nonunitary=True, nonunitary_is_stochastic=True
        ),
        trajectory=_TrajectoryCapabilities(classical_register=True, occupancy=True),
        kernels=_KernelCapabilities(
            supports_kernel_threads=False, thread_capacity=1, supports_fusion=False
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
        return _plan_python_execution(
            plan,
            context=context,
            simulation_config=simulation_config,
            initial_state=initial_state,
            capabilities=self.capabilities,
        )

    def __init__(self, name: str = "numpy-sv"):
        super().__init__(name)
        self._channel_routes = _ChannelRouteCache()

    def _retain_step_caches(self, plan: tuple[ResolvedStep, ...]) -> None:
        self._channel_routes.retain(plan)

    _allocate = staticmethod(_allocate_statevector)

    def _bind_kernels_to_python_workflow(
        self,
        plan: tuple[ResolvedStep, ...],
        measurements: tuple[tuple[int, int], ...] = (),
        *,
        policy: ExecutionPolicy | None = None,
    ) -> _PythonExecution:
        return _PythonExecution(
            plan=plan,
            deferred_measurements=measurements,
            apply_matrix=_apply_matrix_statevector,
            apply_channel=partial(
                _apply_channel_statevector, routes=self._channel_routes
            ),
            measurement=_MeasurementKernels(
                _statevector_probabilities, _project_statevector, _sample_basis_indices
            ),
            reset=_reset_statevector,
        )


# --- density-matrix engine ---


class NumpyDMEngine(_NdarrayEngine):
    """Density-matrix engine: evolves ``rho`` as a ``(size, size)`` matrix."""

    _capabilities = _EngineCapabilities(
        quantum=_QuantumCapabilities(
            "density_matrix", supports_nonunitary=True, nonunitary_is_stochastic=False
        ),
        trajectory=_TrajectoryCapabilities(classical_register=True, occupancy=True),
        kernels=_KernelCapabilities(
            supports_kernel_threads=False, thread_capacity=1, supports_fusion=False
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
        return _plan_python_execution(
            plan,
            context=context,
            simulation_config=simulation_config,
            initial_state=initial_state,
            capabilities=self.capabilities,
        )

    def __init__(self, name: str = "numpy-dm"):
        super().__init__(name)

    _allocate = staticmethod(_allocate_density_matrix)

    def _bind_kernels_to_python_workflow(
        self,
        plan: tuple[ResolvedStep, ...],
        measurements: tuple[tuple[int, int], ...] = (),
        *,
        policy: ExecutionPolicy | None = None,
    ) -> _PythonExecution:
        return _PythonExecution(
            plan=plan,
            deferred_measurements=measurements,
            apply_matrix=_apply_matrix_density_matrix,
            apply_channel=_apply_channel_density_matrix,
            measurement=_MeasurementKernels(
                _density_matrix_probabilities,
                _project_density_matrix,
                _sample_basis_indices,
            ),
            reset=_reset_density_matrix,
        )


# --- operator engines ---


class NumpyUnitaryEngine(NumpySVEngine):
    """Unitary engine: evolves ``U`` as a ``(size, size)`` operator matrix.

    Column ``j`` of ``U`` is the statevector ``U|j>``, so the whole operator is
    the statevector kernel run on ``size`` columns at once.
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
            supports_kernel_threads=False, thread_capacity=1, supports_fusion=False
        ),
        supports_process_shots=False,
    )

    def __init__(self, name: str = "numpy-unitary"):
        super().__init__(name)

    _allocate = staticmethod(_allocate_unitary)

    def _bind_kernels_to_python_workflow(
        self,
        plan: tuple[ResolvedStep, ...],
        measurements: tuple[tuple[int, int], ...] = (),
        *,
        policy: ExecutionPolicy | None = None,
    ) -> _PythonExecution:
        return _PythonExecution(
            plan=plan,
            deferred_measurements=measurements,
            apply_matrix=_apply_matrix_unitary,
            apply_channel=None,
            measurement=None,
            reset=None,
        )


class NumpySuperopEngine(NumpyDMEngine):
    """Super-operator engine: evolves ``S`` as a ``(size**2, size**2)`` matrix.

    Internally, column ``b`` of ``S`` is the row-stacked image of basis matrix
    ``E_b``, so the whole channel is the density-matrix kernel run on
    ``size**2`` columns at once (``S = kron(M, conj(M))``). :meth:`export_state`
    converts the completed matrix once to the public column-stacking
    convention.
    """

    _capabilities = _EngineCapabilities(
        quantum=_QuantumCapabilities(
            "superop", supports_nonunitary=True, nonunitary_is_stochastic=False
        ),
        trajectory=_TrajectoryCapabilities(),
        kernels=_KernelCapabilities(
            supports_kernel_threads=False, thread_capacity=1, supports_fusion=False
        ),
        supports_process_shots=False,
    )

    def __init__(self, name: str = "numpy-superop"):
        super().__init__(name)
        self._reset_channels = _ResetChannelCache()

    _allocate = staticmethod(_allocate_superoperator)

    def _bind_kernels_to_python_workflow(
        self,
        plan: tuple[ResolvedStep, ...],
        measurements: tuple[tuple[int, int], ...] = (),
        *,
        policy: ExecutionPolicy | None = None,
    ) -> _PythonExecution:
        return _PythonExecution(
            plan=plan,
            deferred_measurements=measurements,
            apply_matrix=_apply_matrix_superoperator,
            apply_channel=_apply_channel_superoperator,
            measurement=None,
            reset=partial(_reset_superoperator, reset_channels=self._reset_channels),
        )

    def export_state(self) -> np.ndarray:
        """Convert superoperator data to public column stacking."""
        return _export_superoperator(self.state, self._dims)
