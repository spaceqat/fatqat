"""Shared engine lifecycle for NumPy-backed state buffers.

NumPy and Numba share storage. Concrete engines select workflows, allocate
their representation, and bind their own numerical functions, expectation
kernels included.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Sequence
from copy import deepcopy

import numpy as np

from ....observable import _Factors
from .._execution_policy import _ExecutionPolicy as ExecutionPolicy
from ..state import EvolutionState, _LoweredBasisState
from ..numerics.numpy_kernels._measurement import _strides
from .base import MatrixEngine


class _NdarrayEngine(MatrixEngine[np.ndarray]):
    """Initialization and expectation evaluation on owned ndarray state."""

    # ``(state, terms, dims, strides) -> float``; each concrete engine binds the
    # statevector or density-matrix kernel of its own runtime.
    _expectation_kernel: Callable[..., float]

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
        observables: Sequence[tuple[tuple[float, _Factors], ...]],
        *,
        policy: ExecutionPolicy,
    ) -> tuple[float, ...]:
        kernel = self._expectation_kernel
        dims = tuple(self._dims)
        strides = _strides(dims)  # the engine is little-endian
        values = []
        with self._execution_scope(policy):
            for terms in observables:
                values.append(kernel(state, terms, dims, strides))
        return tuple(values)

    @abstractmethod
    def _allocate(
        self, dims: Sequence[int], start: np.ndarray | _LoweredBasisState | None
    ) -> np.ndarray:
        """Return a fresh owned state over ``dims`` from the engine's start."""
