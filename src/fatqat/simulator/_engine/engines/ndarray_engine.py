"""Shared engine lifecycle for NumPy-backed state buffers.

NumPy and Numba share storage and expectation arithmetic. Concrete engines
select workflows, allocate their representation, and bind numerical functions.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from copy import deepcopy

import numpy as np

from ...._expectation import expectation_density_matrix, expectation_statevector
from .._execution_policy import _ExecutionPolicy as ExecutionPolicy
from ..state import EvolutionState, _LoweredBasisState
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
        operator = self.capabilities.quantum.is_operator
        if initial_state is None:
            # Direct callers get the same start Simulator would pass.
            ground = None if operator else _LoweredBasisState((0,) * len(self._dims))
            initial_state = EvolutionState(ground)
        # Operator allocators check their own side: no initial state.
        assert (
            operator or initial_state.quantum is not None
        ), "state methods start from a lowered basis state or an array"
        quantum = self._allocate(self._dims, initial_state.quantum)
        classical = deepcopy(initial_state.classical)
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
    def _allocate(
        self, dims: Sequence[int], start: np.ndarray | _LoweredBasisState | None
    ) -> np.ndarray:
        """Return a fresh owned state over ``dims`` from the engine's start."""
