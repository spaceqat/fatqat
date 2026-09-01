"""Numba kernels for noise math: channel Kraus payloads and readout confusion.

The Kraus payload of an ``ApplyChannelStep`` needs two things done to it that
gate matrices never need, and this module is where both live as Numba kernels:

- **statevector**: quantum-jump unravelling - weigh the candidate branches
  ``K_i |psi>`` by their squared norms, draw one, renormalize
  (`_jump_branch_kernel`).
- **density matrix**: collapse the whole Kraus sum into the single
  super-operator ``sum_i kron(K_i, conj(K_i))`` acting on ``vec(rho)``
  (`_kraus_superop_kernel`).

Classical readout confusion is the other noise source with in-kernel math, and it
is not a channel: the collapse keeps the true outcome and only the *reported*
digit is resampled through a column-stochastic confusion matrix
(`_report_digit_kernel`).

Plus `_compile_channel_table` and `_compile_readout_table`, which flatten a
plan's channel occurrences and per-measurement confusion matrices into the
typed arrays the compiled multi-shot kernel in ``simulator._engine.nb`` walks.

Division of labour with ``simulator._engine.nb``, and why it runs this way: *applying*
a local matrix is representation machinery and stays there (one coset-walk
kernel family, shared by gates, reset shifts, and Kraus operators alike), so
`_jump_branch_kernel` takes the branch stack *already produced* by that
primitive and only does the channel-specific part. This keeps the dependency
one-directional - ``simulator._engine.nb`` imports this module, never the reverse.
That direction is forced, not stylistic: a Numba function resolves the njit
callees in its body as module globals at compile time, so the compiled shots
kernel cannot reach a channel kernel it does not import at module scope, and
importing back would cycle.

Numba compiles kernels lazily on first call. This module is never imported from
``fatqat.noise``'s package ``__init__``; import it explicitly
(``from fatqat.noise.nb import ...``), exactly as
``fatqat.simulator._engine.nb`` is treated.

Conventions match ``simulator._engine.np`` / ``simulator._engine.nb``: little-endian flat
indexing, and a local Kraus matrix whose most-significant index digit is
``target_indices[0]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

import numpy as np
from numba import njit

from .base import _sampled_unitary_branches


def _kraus_stack(kraus_ops: Sequence[np.ndarray]) -> np.ndarray:
    """Stack a channel's Kraus tuple into one contiguous ``(num, d, d)`` array.

    The kernels below want a single typed buffer rather than a tuple of frozen
    arrays; ``np.ascontiguousarray`` copies, so the step's read-only operators
    are never aliased into kernel-writable memory.
    """
    return np.ascontiguousarray(kraus_ops, dtype=np.complex128)


@njit(cache=True)
def _inverse_cdf_pick(
    probabilities: np.ndarray, u: float
) -> int:  # pragma: no cover - compiled by Numba
    """Pick an index from ``probabilities`` by inverse CDF at uniform ``u``.

    Accumulates the cdf, normalizes it by its last entry, and returns the
    first entry strictly greater than ``u`` - the exact two steps
    ``rng.choice(n, p=...)`` performs internally, so one ``rng.random()`` fed
    here consumes the caller's stream identically. A zero-probability entry
    has a zero-width interval and is therefore never returned. Callers pass
    the same ``p`` their NumPy counterpart passes to ``rng.choice``, already
    normalized or not, so the arithmetic matches term for term.

    The normalization makes ``cdf[-1]`` exactly 1.0 and ``u < 1.0``, so the
    search cannot run off the end; the clamp holds that invariant explicitly
    because callers use the result to index memory.

    The cdf is never materialized: it is accumulated twice instead, once for
    its total and once to find the crossing. Same additions and same division
    per entry, and a cdf is non-decreasing, so this is bit-identical to
    bisecting the array.

    ``simulator._engine.nb`` has the same two steps for basis-index sampling. It is
    duplicated rather than imported: this module must not import from
    ``simulator._engine.nb`` (see the module docstring on the forced direction).
    """
    n = probabilities.shape[0]
    total = 0.0
    for i in range(n):
        total += probabilities[i]

    running = 0.0
    for i in range(n):
        running += probabilities[i]
        if running / total > u:
            return i
    return n - 1


@njit(cache=True)
def _jump_branch_kernel(
    branches: np.ndarray, u: float
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Draw one quantum-jump branch given uniform ``u`` and return it normalized.

    ``branches[i]`` is the flat state ``K_i |psi>``, already applied by the
    caller's local-apply primitive. Its squared norm is the branch
    probability, which CPTP makes sum to ``<psi|psi> = 1``; the weights are
    divided by their total anyway - matching the NumPy twin's
    ``rng.choice(num, p=norms / norms.sum())`` - so round-off can never leave
    ``u`` past the end.
    """
    num = branches.shape[0]
    size = branches.shape[1]

    weights = np.empty(num, dtype=np.float64)
    total = 0.0
    for i in range(num):
        norm_sq = 0.0
        for j in range(size):
            amplitude = branches[i, j]
            norm_sq += amplitude.real * amplitude.real + amplitude.imag * amplitude.imag
        weights[i] = norm_sq
        total += norm_sq

    probabilities = np.empty(num, dtype=np.float64)
    for i in range(num):
        probabilities[i] = weights[i] / total
    chosen = _inverse_cdf_pick(probabilities, u)

    norm = sqrt(weights[chosen])
    out = np.empty(size, dtype=np.complex128)
    for j in range(size):
        out[j] = branches[chosen, j] / norm
    return out


@njit(cache=True)
def _report_digit_kernel(
    conf_flat: np.ndarray, ptr: int, dim: int, true_digit: int, u: float
) -> int:  # pragma: no cover - compiled by Numba
    """Resample one reported readout digit through a confusion matrix.

    ``conf_flat[ptr:]`` holds a row-major ``(dim, dim)`` column-stochastic
    matrix ``C[i, j] = P(report i | true j)``, so column ``true_digit`` is the
    reporting distribution for this outcome and is already normalized - it is
    handed to `_inverse_cdf_pick` unscaled, exactly as the NumPy twin hands
    ``confusion[:, true_digit]`` to ``rng.choice``.

    Only the reported classical value is affected; the caller's collapsed
    state keeps the true outcome.
    """
    column = np.empty(dim, dtype=np.float64)
    for i in range(dim):
        column[i] = conf_flat[ptr + i * dim + true_digit]
    return _inverse_cdf_pick(column, u)


@njit(cache=True)
def _kraus_superop_kernel(
    stack: np.ndarray,
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Build ``sum_i kron(K_i, conj(K_i))`` from a ``(num, d, d)`` Kraus stack.

    The one super-operator that reproduces the exact channel
    ``rho' = sum_i K_i rho K_i^dagger`` when applied to ``vec(rho)`` with the
    ket group most-significant (see `NumbaDMEngine`). Entries accumulate in
    Kraus order, so this is ``sum(np.kron(k, k.conj()) for k in kraus_ops)``
    term for term - up to last-bit round-off, since Numba may contract each
    multiply-accumulate into an FMA where NumPy rounds the product first.
    """
    num = stack.shape[0]
    d = stack.shape[1]
    out = np.zeros((d * d, d * d), dtype=np.complex128)
    for i in range(num):
        for a in range(d):
            for b in range(d):
                ket = stack[i, a, b]
                for p in range(d):
                    for q in range(d):
                        out[a * d + p, b * d + q] += ket * stack[i, p, q].conjugate()
    return out


def _compile_channel_table(entries: Sequence[tuple]) -> tuple:
    """Flatten a plan's channel occurrences for the compiled multi-shot kernel.

    Each entry is ``(kraus_ops, offsets, comp_strides, comp_dims)``: the
    step's resolved Kraus tuple plus the coset layout its target tuple got
    from the caller's apply-plan cache. Channels keep their own flat pools
    (they are the only steps carrying a stack of matrices rather than one), so
    the returned arrays are self-contained and independent of the gate table.

    Alongside each Kraus operator this precomputes ``M_i = K_i^dagger K_i``
    (same ``(num, d, d)`` layout, indexed by the same per-channel pointer),
    the local operator the compiled kernel needs for the branch weight: the
    quantum-jump probability ``<psi|K_i^dagger K_i|psi>`` is ``Tr(M_i rho_T)``
    over the target subsystems' reduced density matrix, so the kernel weighs
    every branch from one ``O(size)`` reduced density matrix and these ``d x d``
    operators, never by materializing and norming ``num`` full branches.

    A channel of scaled unitaries (`_sampled_unitary_branches`) needs neither.
    Its branch probabilities go into ``cdf_flat`` already accumulated and
    normalized, its operators into ``kra_flat`` already divided by their own
    scale, and ``ident_flat`` marks the branches that are the identity - so
    the kernel's per-occurrence work is a search, and usually nothing else.
    ``cdf_ptr`` is the channel's offset into ``cdf_flat`` and ``ident_flat``
    alike, or ``-1`` for a channel weighed against the state. ``mmat_diag``
    marks a channel whose ``M_i`` are all diagonal, weighable from the target
    marginal alone.

    ``mmat_flat`` is taken of whatever ``kra_flat`` holds, so
    ``M_i = K_i^dagger K_i`` is true of the stored operators either way.

    Returns the kernel's positional channel arrays: per-channel pointers and
    flags, then the pools themselves (the Kraus stack, the ``M_i`` stack, and
    the offset / complement / cdf / identity backings).
    """
    kra_ptr: list[int] = []
    num_kraus: list[int] = []
    local_dim: list[int] = []
    off_ptr: list[int] = []
    comp_ptr: list[int] = []
    comp_len: list[int] = []
    cdf_ptr: list[int] = []
    mmat_diag: list[int] = []
    kra_flat: list[complex] = []
    mmat_flat: list[complex] = []
    off_flat: list[int] = []
    comp_stride_flat: list[int] = []
    comp_dim_flat: list[int] = []
    cdf_flat: list[float] = []
    ident_flat: list[int] = []

    for kraus_ops, offsets, comp_strides, comp_dims in entries:
        branches = _sampled_unitary_branches(tuple(kraus_ops))
        if branches is None:
            cdf_ptr.append(-1)
            stack = _kraus_stack(kraus_ops)
        else:
            probabilities, unitaries, identities = branches
            cdf_ptr.append(len(cdf_flat))
            # Same running sums and division `_inverse_cdf_pick` would do per
            # draw, so a search over this finds the index it would have.
            cumulative = np.cumsum(probabilities)
            cdf_flat.extend(float(c) for c in cumulative / cumulative[-1])
            ident_flat.extend(int(flag) for flag in identities)
            stack = _kraus_stack(unitaries)
        assert stack.shape[1] == offsets.shape[0], (
            "Kraus dimension must equal the coset layout's local dimension"
        )
        kra_ptr.append(len(kra_flat))
        num_kraus.append(stack.shape[0])
        local_dim.append(stack.shape[1])
        kra_flat.extend(stack.ravel().tolist())
        diagonal_only = True
        for kraus in stack:
            mmat = kraus.conj().T @ kraus
            mmat_flat.extend(mmat.ravel().tolist())
            # Exact zeros, like `_classify_matrix`: a merely tiny off-diagonal
            # entry is a real one and must take the general path.
            diagonal_only = diagonal_only and not np.any(
                mmat - np.diag(np.diagonal(mmat))
            )
        mmat_diag.append(int(diagonal_only))
        off_ptr.append(len(off_flat))
        off_flat.extend(int(o) for o in offsets)
        comp_ptr.append(len(comp_stride_flat))
        comp_len.append(comp_strides.shape[0])
        comp_stride_flat.extend(int(s) for s in comp_strides)
        comp_dim_flat.extend(int(d) for d in comp_dims)

    def i64(values):
        return np.asarray(values, dtype=np.int64)

    return (
        i64(kra_ptr),
        i64(num_kraus),
        i64(local_dim),
        i64(off_ptr),
        i64(comp_ptr),
        i64(comp_len),
        i64(cdf_ptr),
        i64(mmat_diag),
        np.asarray(kra_flat, dtype=np.complex128),
        np.asarray(mmat_flat, dtype=np.complex128),
        i64(off_flat),
        i64(comp_stride_flat),
        i64(comp_dim_flat),
        np.asarray(cdf_flat, dtype=np.float64),
        i64(ident_flat),
    )


def _compile_readout_table(entries: Sequence[tuple]) -> tuple:
    """Flatten readout confusion matrices for the compiled multi-shot kernel.

    Each entry is one measurement step's ``(num_subsystems, confusions)``,
    where ``confusions`` is the step's per-subsystem tuple (entries may be
    ``None``) or ``None`` for an error-free measurement. Entries must arrive in
    the order the caller flattened its measured subsystems, because the
    returned pointers are indexed by that same flat subsystem position.

    A confusion's side length is not stored: lowering validates each matrix
    against its measured subsystem's dimension, so the kernel reads it from
    the measurement table's ``me_dim``.

    Returns ``(conf_ptr, conf_flat)``: ``conf_ptr[k]`` is where subsystem
    ``k``'s row-major matrix starts in ``conf_flat``, or ``-1`` when that
    subsystem reports without error (the common case, which stores nothing).
    """
    conf_ptr: list[int] = []
    conf_flat: list[float] = []
    for num_subsystems, confusions in entries:
        if confusions is None:
            conf_ptr.extend([-1] * num_subsystems)
            continue
        assert len(confusions) == num_subsystems, (
            "confusions must align with the step's measured subsystems"
        )
        for confusion in confusions:
            if confusion is None:
                conf_ptr.append(-1)
                continue
            matrix = np.ascontiguousarray(confusion, dtype=np.float64)
            conf_ptr.append(len(conf_flat))
            conf_flat.extend(matrix.ravel().tolist())
    return (
        np.asarray(conf_ptr, dtype=np.int64),
        np.asarray(conf_flat, dtype=np.float64),
    )
