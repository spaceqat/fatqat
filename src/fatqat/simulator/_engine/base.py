from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

import numpy as np

from ..._backends.engine_contract import (
    _DensityMatrixResultRequest as DensityMatrixResultRequest,
    _EngineConfig as EngineConfig,
    _StateVectorResultRequest as StateVectorResultRequest,
    RawResult,
)
from ..._backends.steps import ResolvedStep, ApplyMatrixStep

ResultRequest = DensityMatrixResultRequest | StateVectorResultRequest


class MatrixEngine(ABC):
    """
    Abstract base class and interface contract for all engines.
    """

    def __init__(
        self,
        name: str,
        config: EngineConfig | None = None,
        *,
        state_semantics: Literal["sv", "dm"],
    ):
        self.name = name
        self.config = config or EngineConfig()
        self.state_semantics = state_semantics

        self._state: np.ndarray = None  # type: ignore[assignment]
        self._initial_state: np.ndarray | None = None
        self._dims: tuple[int, ...] = ()
        self._reversed_dims: tuple[int, ...] = ()
        self._n_clbits = 0

    @property
    def state(self) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("MatrixEngine state has not been initialized.")
        return self._state

    @state.setter
    def state(self, value: np.ndarray) -> None:
        self._state = value

    @property
    def initial_state(self) -> np.ndarray | None:
        """State every shot starts from, or ``None`` for the all-zero state.

        Held on the engine rather than passed to `initialize` because
        `initialize` is also how a dynamic run returns to the start of the next
        shot: a per-shot reset must land on the state this run began with, not
        on the computational zero. Standard paths read it through `_allocate`;
        the compiled multi-shot path uses it as a read-only template and can
        initialize the default zero-state buffers directly without one.

        Every evolving buffer owns its storage, so a caller's array is never
        evolved in place.
        """
        return self._initial_state

    @initial_state.setter
    def initial_state(self, value: np.ndarray | None) -> None:
        self._initial_state = value

    def _prepare_execution_plan(
        self, plan: list[ResolvedStep], config: EngineConfig
    ) -> list[ResolvedStep]:
        """Return the plan this engine will actually execute.

        Every engine passes a plan through here before executing it, and any
        rewrite an engine makes to a plan lives here and nowhere else. The base
        engine rewrites nothing.

        The single point exists because the alternative had already failed:
        fusion was applied at whichever sites happened to need it, so adding a
        switch meant finding them all, and one was missed - leaving a setting
        that appeared to work. A rewrite added here reaches every path by
        construction; one added at a call site reaches only that call.

        ``config`` is the *effective* config for this run, not the engine's
        construction default, since a per-run option has to be able to change
        what the plan becomes.
        """
        return plan

    @property
    def n_subsystems(self) -> int:
        return len(self._dims)

    @abstractmethod
    def initialize(self, system_dims: Sequence[int], n_clbits: int = 0) -> None:
        """Configure dimensions and reset to the all-zero computational state."""

    def _set_dims(self, system_dims: Sequence[int]) -> None:
        """Set ``_dims`` and its cached reverse together, so they never drift apart."""
        self._dims = tuple(int(d) for d in system_dims)
        self._reversed_dims = tuple(reversed(self._dims))

    @abstractmethod
    def run(
        self,
        plan: list[ResolvedStep],
        shots: int,
        seed: int | None,
        request: ResultRequest,
        *,
        config: EngineConfig | None = None,
    ) -> RawResult: ...

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

    def export_state(self) -> np.ndarray:
        """
        Export the current state of the engine as a numpy array.
        """
        return self.state.copy()

    def sample_indices(self, shots: int, rng: np.random.Generator) -> np.ndarray:
        """
        Sample flat basis-state indices from the current state.
        """
        return rng.choice(self.state.shape[0], size=shots, p=self.probabilities())
