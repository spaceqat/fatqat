"""Implementations of user-defined circuit operations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..operations import CustomOperation

if TYPE_CHECKING:
    from ..simulator import EvolutionState, ExecutionContext


class CustomOperationImplementation(ABC):
    """Base class for custom-operation implementations.

    Define ``execute`` as a static method and register an instance with
    ``MatrixImplementationMap.add``.
    """

    @staticmethod
    @abstractmethod
    def execute(
        operation: CustomOperation,
        state: EvolutionState,
        context: ExecutionContext,
    ) -> EvolutionState:
        """Apply an operation to the current shot's quantum and classical state.

        Use ``context.rng`` for random draws to respect the simulation seed.

        Args:
            operation: Operation instance, including its parameters.
            state: This shot's quantum data and classical data.
            context: Targets, dimensions, and sampling information; see
                ``ExecutionContext``.

        Returns:
            The mutated ``state`` or a replacement ``EvolutionState``. Preserve
            the quantum shape and dtype, and the classical-register size.
        """
