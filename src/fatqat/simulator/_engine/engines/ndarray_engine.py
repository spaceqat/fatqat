"""Shared engine lifecycle for NumPy-backed state buffers.

NumPy and Numba share storage and expectation arithmetic. Concrete engines
select workflows, allocate their representation, and bind numerical functions.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from math import prod

import numpy as np

from ...._expectation import expectation_density_matrix, expectation_statevector
from .._execution_policy import _ExecutionPolicy as ExecutionPolicy
from ..state import EvolutionState
from .base import MatrixEngine


class _NdarrayEngine(MatrixEngine[np.ndarray]):
    """Initialization and expectation evaluation on owned ndarray state."""

    def initialize(
        self,
        system_dims: Sequence[int],
        n_clbits: int = 0,
        *,
        initial_state: EvolutionState[np.ndarray] | None = None,
    ) -> None:
        self.configure_system(system_dims, n_clbits)
        initial = initial_state if initial_state is not None else EvolutionState()
        quantum = self._allocate(prod(self._dims) if self._dims else 1, initial.quantum)
        classical = deepcopy(initial.classical)
        self._evolution_state = EvolutionState(quantum, classical)

    def _expectation_values(
        self,
        state: np.ndarray,
        observables: Sequence[tuple[tuple[float, tuple[tuple[int, str], ...]], ...]],
        *,
        policy: ExecutionPolicy,
    ) -> tuple[float, ...]:
        kernel = (
            expectation_statevector
            if self._state_field == "statevector"
            else expectation_density_matrix
        )
        values = []
        with self._execution_scope(policy):
            for terms in observables:
                kernel_terms = tuple(
                    (
                        coefficient,
                        tuple(
                            (self.n_subsystems - 1 - index, letter)
                            for index, letter in factors
                        ),
                    )
                    for coefficient, factors in terms
                )
                values.append(kernel(state, kernel_terms))
        return tuple(values)

    @abstractmethod
    def _allocate(self, size: int, initial_state: np.ndarray | None) -> np.ndarray:
        """Return a fresh owned state over ``size`` basis states."""
