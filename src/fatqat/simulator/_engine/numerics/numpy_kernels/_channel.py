"""NumPy Kraus application, branch sampling, and channel route reuse."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial

import numpy as np

from ....._backends.steps import ApplyChannelStep, ResolvedStep
from .....noise.base import _sampled_unitary_branches
from ._apply import _apply_statevector, _apply_density_matrix, _apply_superoperator

# What `_sampled_unitary_branches` resolves a channel step to: branch
# probabilities, unit-norm operators, identity flags - or None for a channel
# that must be weighed against the state.
_Branches = tuple[np.ndarray, tuple[np.ndarray, ...], tuple[bool, ...]] | None


class _ChannelRouteCache:
    """Prepare state-independent channel branches across shot initialization.

    Entries pin their source steps to prevent recycled identities from aliasing.
    Routes depend only on frozen Kraus operators, not dimensions or live state.
    Preparing a new payload retains shared steps and releases replaced ones.
    """

    def __init__(self):
        self._entries: dict[int, tuple[ApplyChannelStep, _Branches]] = {}

    def retain(self, plan: tuple[ResolvedStep, ...]) -> None:
        """Keep only resolutions used by the next effective plan."""
        active_ids = {id(step) for step in plan}
        self._entries = {
            step_id: entry
            for step_id, entry in self._entries.items()
            if step_id in active_ids
        }

    def resolve(self, step: ApplyChannelStep) -> _Branches:
        """Resolve a channel step's branch-weighing route once, then cache it."""
        cached = self._entries.get(id(step))
        if cached is not None and cached[0] is step:
            return cached[1]
        branches = _sampled_unitary_branches(step.kraus_ops)
        self._entries[id(step)] = (step, branches)
        return branches


def _apply_kraus_jump(
    source: np.ndarray,
    kraus_ops: tuple[np.ndarray, ...],
    targets: Sequence[int],
    rng: np.random.Generator,
    apply_local: Callable[[np.ndarray, np.ndarray, Sequence[int]], np.ndarray],
) -> np.ndarray:
    """Sample and normalize one Kraus branch, consuming one RNG draw.

    Each branch starts from a copy so an in-place local application cannot
    corrupt the source used by the remaining branches.
    """
    branches = [apply_local(source.copy(), kraus, targets) for kraus in kraus_ops]
    norms = np.array([np.real(np.vdot(b, b)) for b in branches])
    # CPTP guarantees the norms sum to <psi|psi> = 1; renormalize anyway so
    # rng.choice never rejects the distribution over float round-off.
    chosen = int(rng.choice(len(branches), p=norms / norms.sum()))
    return branches[chosen] / np.sqrt(norms[chosen])


def _apply_kraus_sum(
    rho: np.ndarray,
    kraus_ops: tuple[np.ndarray, ...],
    targets: Sequence[int],
    apply_local: Callable[[np.ndarray, np.ndarray, Sequence[int]], np.ndarray],
) -> np.ndarray:
    """Sum local Kraus sandwiches for a density matrix or superoperator."""
    new = np.zeros_like(rho)
    for kraus in kraus_ops:
        new += apply_local(rho, kraus, targets)
    return new


def _sample_unitary(
    branches: tuple[np.ndarray, tuple[np.ndarray, ...], tuple[bool, ...]],
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Draw one scaled-unitary branch, returning None for an identity.

    Probabilities are state-independent and consume exactly one RNG draw.
    Returned operators are already unit-norm; identity branches need no
    state access, application, or defensive copy.
    """
    probabilities, unitaries, identities = branches
    chosen = int(rng.choice(len(probabilities), p=probabilities / probabilities.sum()))
    if identities[chosen]:
        return None
    return unitaries[chosen]


# --- Python-facing channel operations ---


def _apply_channel_statevector(
    state: np.ndarray,
    step: ApplyChannelStep,
    dims: Sequence[int],
    rng: np.random.Generator,
    *,
    routes: _ChannelRouteCache,
) -> np.ndarray:
    """Sample one branch, using fixed probabilities for scaled unitaries.

    Both routes consume one draw. General Kraus branches start from independent
    state copies so in-place numerical implementations cannot corrupt the source.
    """
    apply_matrix = partial(
        _apply_statevector, dims=dims, reversed_dims=tuple(reversed(dims))
    )
    branches = routes.resolve(step)
    if branches is None:
        return _apply_kraus_jump(
            state, step.kraus_ops, step.target_indices, rng, apply_matrix
        )
    matrix = _sample_unitary(branches, rng)
    return state if matrix is None else apply_matrix(state, matrix, step.target_indices)


def _apply_channel_density_matrix(
    state: np.ndarray,
    step: ApplyChannelStep,
    dims: Sequence[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply the exact Kraus sum without consuming randomness."""
    apply_matrix = partial(
        _apply_density_matrix, dims=dims, reversed_dims=tuple(reversed(dims))
    )
    return _apply_kraus_sum(state, step.kraus_ops, step.target_indices, apply_matrix)


def _apply_channel_superoperator(
    state: np.ndarray,
    step: ApplyChannelStep,
    dims: Sequence[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply the exact Kraus sum to every accumulated operator column."""
    apply_matrix = partial(
        _apply_superoperator, dims=dims, reversed_dims=tuple(reversed(dims))
    )
    return _apply_kraus_sum(state, step.kraus_ops, step.target_indices, apply_matrix)
