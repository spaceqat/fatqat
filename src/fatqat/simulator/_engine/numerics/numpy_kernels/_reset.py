"""NumPy state reset and reusable operator reset channels."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import numpy as np

from ....._backends.steps import ApplyChannelStep
from .....implementation.matrices import shift_matrix
from ._apply import _apply_statevector, _apply_superoperator
from ._measurement import _project_statevector, _statevector_probabilities

# This cache has one operation: resolve an immutable reset channel on demand.


class _ResetChannelCache:  # pylint: disable=too-few-public-methods
    """Reuse reset channels by subsystem and dimension across executions."""

    def __init__(self):
        self._entries: dict[tuple[int, int], ApplyChannelStep] = {}

    def resolve(self, index: int, dim: int) -> ApplyChannelStep:
        """Return the single-subsystem reset channel, building it once."""
        step = self._entries.get((index, dim))
        if step is None:
            kraus = []
            for outcome in range(dim):
                operator = np.zeros((dim, dim), dtype=complex)
                operator[0, outcome] = 1.0
                kraus.append(operator)
            step = ApplyChannelStep(kraus_ops=tuple(kraus), target_indices=(index,))
            self._entries[(index, dim)] = step
        return step


def _digit(flat: int, index: int, dims: Sequence[int]) -> int:
    """Little-endian digit of subsystem ``index`` in flat basis state ``flat``."""
    stride = 1
    for q in range(index):
        stride *= dims[q]
    return (flat // stride) % dims[index]


def _reset_density_matrix(
    rho: np.ndarray,
    indices: Sequence[int],
    dims: Sequence[int],
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Trace out targets and reprepare zero without consuming the supplied RNG.

    ``rho' = |0..0><0..0|_targets (x) Tr_targets(rho)`` is trace-preserving;
    no renormalization is needed. The generator is accepted for interface parity.
    """
    assert len(indices) >= 1, "reset_subsystems requires at least one index"
    reversed_dims = tuple(reversed(dims))
    n = len(dims)
    targets = tuple(indices)
    local_dims = tuple(dims[t] for t in targets)
    dt = prod(local_dims)
    rest = rho.shape[0] // dt

    tensor = rho.reshape(reversed_dims * 2)
    ket_axes = [n - 1 - q for q in targets]
    bra_axes = [2 * n - 1 - q for q in targets]
    moved = ket_axes + bra_axes
    # `remaining` stays ascending, so rest-ket axes precede rest-bra axes and
    # the (dt, dt, rest, rest) regroup below is valid.
    remaining = [ax for ax in range(2 * n) if ax not in moved]
    block = np.transpose(tensor, moved + remaining).reshape(dt, dt, rest, rest)

    rho_rest = np.trace(block, axis1=0, axis2=1)
    post = np.zeros_like(block)
    post[0, 0] = rho_rest

    rest_shape = tuple((reversed_dims * 2)[ax] for ax in remaining)
    post = post.reshape(local_dims + local_dims + rest_shape)
    inverse_perm = np.argsort(moved + remaining)
    return np.transpose(post, inverse_perm).reshape(rho.shape)


def _reset_statevector(
    state: np.ndarray,
    indices: Sequence[int],
    dims: Sequence[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one grouped branch, project, then shift its targets to zero.

    Consumes one RNG draw. Unmeasured subsystems retain the state conditioned
    on that outcome, including any remaining superposition.
    """
    assert len(indices) >= 1, "reset_subsystems requires at least one index"
    outcome_index = int(rng.choice(len(state), p=_statevector_probabilities(state)))
    state = _project_statevector(state, outcome_index, indices, dims)
    reversed_dims = tuple(reversed(dims))
    for index in indices:
        outcome = _digit(outcome_index, index, dims)
        if outcome != 0:
            inv = shift_matrix(dims[index], -outcome)
            state = _apply_statevector(state, inv, (index,), dims, reversed_dims)
    return state


def _reset_superoperator(
    state: np.ndarray,
    indices: Sequence[int],
    dims: Sequence[int],
    rng: np.random.Generator | None = None,
    *,
    reset_channels: _ResetChannelCache,
) -> np.ndarray:
    """Apply cached reset channels to every operator column, without RNG draws.

    Each target uses ``sum_k |0><k| . |k><0|``, the operator representation of
    partial trace followed by preparation of zero. Cached Kraus matrices are
    reused across executions; the evolving operator is supplied explicitly.
    """
    assert len(indices) >= 1, "reset_subsystems requires at least one index"
    reversed_dims = tuple(reversed(dims))
    for index in indices:
        channel = reset_channels.resolve(index, dims[index])
        result = np.zeros_like(state)
        for kraus in channel.kraus_ops:
            result += _apply_superoperator(
                state, kraus, channel.target_indices, dims, reversed_dims
            )
        state = result
    return state
