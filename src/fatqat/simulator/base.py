from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

import numpy as np

from ..backends.engine_contract import (
    _DensityMatrixResultRequest as DensityMatrixResultRequest,
    _EngineConfig as EngineConfig,
    _StateVectorResultRequest as StateVectorResultRequest,
    RawResult,
)
from ..backends.steps import ResolvedStep, ApplyMatrixStep

ResultRequest = DensityMatrixResultRequest | StateVectorResultRequest


class Simulator(ABC):
    """
    Abstract base class and interface contract for all simulators.
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
        self._dims: tuple[int, ...] = ()
        self._reversed_dims: tuple[int, ...] = ()
        self._n_clbits = 0

    @property
    def state(self) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("Simulator state has not been initialized.")
        return self._state

    @state.setter
    def state(self, value: np.ndarray) -> None:
        self._state = value

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
        Export the current state of the simulator as a numpy array.
        """
        return self.state.copy()

    def sample_indices(self, shots: int, rng: np.random.Generator) -> np.ndarray:
        """
        Sample flat basis-state indices from the current state.
        """
        return rng.choice(self.state.shape[0], size=shots, p=self.probabilities())
