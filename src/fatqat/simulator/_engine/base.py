from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Generic, Literal

import numpy as np

from .._execution_contract import (
    RawResult,
    _InitialClassicalState as InitialClassicalState,
    _InitialEvolutionState as InitialEvolutionState,
    _SimulationConfig as SimulationConfig,
    _EngineCapabilities,
    _KernelCapabilities,
    _QuantumCapabilities,
    _TrajectoryCapabilities,
    _ExecutionContext as ExecutionContext,
)
from ..._backends.steps import ApplyMatrixStep, ResolvedStep
from ._execution_policy import (
    _ExecutionPolicy as ExecutionPolicy,
    _resolve_execution_policy,
    _should_probe_compiled_multi_shot,
)
from .state import EvolutionState, QuantumDataT, QuantumState


def _shot_seed_sequences(
    seed: int | None, n_iters: int
) -> list[np.random.SeedSequence]:
    """Spawn stable, ordered child streams for sampled shots."""
    return np.random.SeedSequence(seed).spawn(n_iters)


class MatrixEngine(ABC, Generic[QuantumDataT]):
    """
    Abstract base class and interface contract for all engines.
    """

    # Concrete engines declare all support; numeric/execution helpers may be shared.
    _quantum_capabilities: _QuantumCapabilities
    _trajectory_capabilities: _TrajectoryCapabilities
    _kernel_capabilities: _KernelCapabilities
    _supports_process_shots: bool = False

    def __init__(self, name: str):
        self.name = name

        self._evolution_state: EvolutionState[QuantumDataT] | None = None
        self._dims: tuple[int, ...] = ()
        self._reversed_dims: tuple[int, ...] = ()
        self._n_clbits = 0

    @property
    def _state(self) -> QuantumDataT | None:
        """Forward existing kernel access to the owned quantum data."""
        if self._evolution_state is None:
            return None
        return self._evolution_state.quantum.data

    @_state.setter
    def _state(self, value: QuantumDataT | None) -> None:
        """Replace the quantum data without losing current classical state."""
        if value is None:
            self._evolution_state = None
        elif self._evolution_state is None:
            self._evolution_state = EvolutionState(
                QuantumState(value, self._state_field)
            )
        else:
            self._evolution_state.quantum.data = value

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
        return self._quantum_capabilities.representation

    @property
    def capabilities(self) -> _EngineCapabilities:
        """Return supported quantum, classical, and numerical execution."""
        return _EngineCapabilities(
            quantum=self._quantum_capabilities,
            trajectory=self._trajectory_capabilities,
            kernels=self._kernel_capabilities,
            supports_process_shots=self._supports_process_shots,
        )

    def compiled_multi_shot_compatible(self, plan: Sequence[ResolvedStep]) -> bool:
        """Whether this engine can own the complete per-shot outer loop."""
        return False

    def resolve_execution_policy(
        self,
        plan: Sequence[ResolvedStep],
        simulation: SimulationConfig,
        *,
        counts_requested: bool,
        state_requested: bool,
        shots: int,
        initial_state: InitialEvolutionState[QuantumDataT],
    ) -> ExecutionPolicy:
        """Choose execution paths for a plan using this engine's support."""
        execution_path = self._select_execution_path(plan, initial_state=initial_state)
        compiled_multi_shot_compatible = False
        if _should_probe_compiled_multi_shot(
            simulation,
            execution_path=execution_path,
            counts_requested=counts_requested,
            state_requested=state_requested,
            initial_state=initial_state,
        ):
            compiled_multi_shot_compatible = self.compiled_multi_shot_compatible(plan)
        capabilities = self.capabilities
        return _resolve_execution_policy(
            simulation,
            execution_path=execution_path,
            counts_requested=counts_requested,
            state_requested=state_requested,
            capabilities=capabilities.kernels,
            supports_process_shots=capabilities.supports_process_shots,
            compiled_multi_shot_compatible=compiled_multi_shot_compatible,
            shots=shots,
            initial_state=initial_state,
            plan_is_empty=not plan,
        )

    @abstractmethod
    def _select_execution_path(
        self,
        plan: Sequence[ResolvedStep],
        *,
        initial_state: InitialEvolutionState[QuantumDataT],
    ) -> Literal["operator", "single_pass", "per_shot"]:
        """Select a supported execution path."""

    def configure_system(self, system_dims: Sequence[int], n_clbits: int = 0) -> None:
        """Configure dimensions without allocating an evolving state."""
        self._set_dims(system_dims)
        self._n_clbits = int(n_clbits)
        self._state = None

    @abstractmethod
    def initialize(
        self,
        system_dims: Sequence[int],
        n_clbits: int = 0,
        *,
        initial_state: InitialEvolutionState[QuantumDataT] | None = None,
    ) -> None:
        """Initialize owned quantum and classical storage from validated input."""

    def _set_dims(self, system_dims: Sequence[int]) -> None:
        """Set ``_dims`` and its cached reverse together, so they never drift apart."""
        self._dims = tuple(int(d) for d in system_dims)
        self._reversed_dims = tuple(reversed(self._dims))

    @abstractmethod
    def materialize_execution(
        self,
        plan: tuple[ResolvedStep, ...],
        *,
        system_dims: tuple[int, ...],
        n_clbits: int,
        policy: ExecutionPolicy,
    ) -> Any:
        """Build the engine-owned immutable payload outside execution scope.

        Retain preparation resources while pending work or execution uses them,
        and establish readiness before execution consumes prepared data. The
        later execution scope does not retroactively cover preparation.
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

    @abstractmethod
    def execute(
        self,
        plan: tuple[ResolvedStep, ...],
        *,
        context: ExecutionContext,
        policy: ExecutionPolicy,
        initial_state: InitialEvolutionState[QuantumDataT],
    ) -> RawResult[QuantumDataT]:
        """Materialize and execute a plan under the resolved execution policy."""

    @abstractmethod
    def execute_local(
        self,
        context: ExecutionContext,
        payload: Any,
        policy: ExecutionPolicy,
        *,
        initial_state: InitialEvolutionState[QuantumDataT],
    ) -> RawResult[QuantumDataT]:
        """Execute a materialized payload locally without dispatching."""

    def execute_shot_batch(
        self,
        context: ExecutionContext,
        payload: Any,
        seed_batch: list[np.random.SeedSequence],
        policy: ExecutionPolicy,
        *,
        initial_state: InitialEvolutionState[QuantumDataT],
    ) -> list[tuple[int, ...]]:
        """Execute one ordered shot batch on engines that support it."""
        raise NotImplementedError

    @abstractmethod
    def measure_subsystems(
        self, indices: Sequence[int], rng: np.random.Generator
    ) -> tuple[int, ...]: ...

    def measure_subsystem(self, index: int, rng: np.random.Generator) -> int:
        """
        Measure a single subsystem and return the result.
        """
        return self.measure_subsystems([index], rng)[0]

    @abstractmethod
    def reset_subsystems(
        self, indices: Sequence[int], rng: np.random.Generator
    ) -> None: ...

    def reset_subsystem(self, index: int, rng: np.random.Generator) -> None:
        """
        Reset a single subsystem to the |0> state.
        """
        self.reset_subsystems([index], rng)

    @abstractmethod
    def probabilities(self) -> np.ndarray:
        """Return the computational-basis probability distribution of the state."""

    @abstractmethod
    def collapse(
        self, measured_subsystems: Sequence[int], rng: np.random.Generator
    ) -> int:
        """Sample one outcome, project the internal state, return the flat index."""

    @abstractmethod
    def apply(self, step: ApplyMatrixStep) -> None:
        """Apply a single matrix step to the internal state in place."""

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
        return self._export_state_data(self.state)

    def _export_state_data(self, data: QuantumDataT) -> QuantumDataT:
        """Convert representation when required, retaining runtime storage."""
        return data

    def _export_evolution_state(self) -> InitialEvolutionState[QuantumDataT]:
        """Export completed local evolution as reusable initialization.

        Quantum data is borrowed in runtime-native storage. Classical values
        are frozen so later execution cannot change the saved initialization.
        Each consumer initializes its own mutable state before evolving it.
        """
        if self._evolution_state is None:
            raise RuntimeError("MatrixEngine state has not been initialized.")
        classical = self._evolution_state.classical
        return InitialEvolutionState(
            quantum=self.export_state(),
            classical=InitialClassicalState(
                clbits=None if classical.clbits is None else tuple(classical.clbits),
                occupied=(
                    None
                    if classical.occupied is None
                    else frozenset(classical.occupied)
                ),
            ),
        )

    @abstractmethod
    def sample_indices(self, shots: int, rng: np.random.Generator) -> np.ndarray:
        """Return sampled flat basis-state indices as a NumPy array."""
