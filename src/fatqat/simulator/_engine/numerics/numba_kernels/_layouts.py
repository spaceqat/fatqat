"""Numba target layouts, execution chunks, and layout reuse.

Statevectors use physical subsystem dimensions; density matrices use doubled
bra/ket dimensions. Operator buffers add an independent column dimension.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from math import prod

import numpy as np
from numba import config as numba_config

# Numba fixes the launch pool's capacity at import time. ``set_num_threads``
# changes only the active mask, so policy ceilings clamp to this configured
# capacity rather than to whichever smaller mask the caller currently uses.
_MAX_THREADS = int(numba_config.NUMBA_NUM_THREADS)
# A coset walk goes parallel only once each worker thread would get at least
# this many amplitudes of work; below that the parallel-region launch/sync cost
# outweighs the memory-bandwidth-bound work it saves. Expressed per-thread so
# the absolute floor scales with the core count (the dominant machine variable)
# as ``_MAX_THREADS * grain``, not a fixed size. The grains reproduce the
# previously hand-tuned 2^15 / 2^18 floors at ~16 threads; the density matrix's
# 4^n pass is more bandwidth-bound, so it takes a larger grain. The floor
# affects speed only - chunking is bit-identical - so a coarse grain suffices.
_GRAIN_TO_THREAD = 1 << 11  # statevector: min amplitudes per thread
_GRAIN_TO_THREAD_DM = 1 << 14  # density matrix: min amplitudes per thread
_MIN_SIZE_TO_THREAD = _MAX_THREADS * _GRAIN_TO_THREAD
_MIN_SIZE_TO_THREAD_DM = _MAX_THREADS * _GRAIN_TO_THREAD_DM


def _choose_chunk_count(
    num_cosets: int,
    size: int,
    min_parallel_size: int = _MIN_SIZE_TO_THREAD,
    *,
    max_workers: int = _MAX_THREADS,
    force_parallel: bool = False,
) -> int:
    """Parallel chunk count for a gate on a ``size``-amplitude state.

    Returns 1 (the serial kernel, no parallel-region overhead) for states below
    ``min_parallel_size``; otherwise splits the cosets across all worker threads.
    """
    if max_workers <= 1:
        return 1
    if not force_parallel and size < min_parallel_size:
        return 1
    return max(1, min(max_workers, num_cosets))


def _build_apply_layout(
    dims: Sequence[int],
    targets: Sequence[int],
    min_parallel_size: int = _MIN_SIZE_TO_THREAD,
    *,
    max_workers: int = _MAX_THREADS,
    force_parallel: bool = False,
) -> tuple:
    """Precompute the strided-block kernel layout for a target tuple.

    Returns ``offsets[c]`` (the flat offset of local index ``c``, with
    ``targets[0]`` most-significant), the flat strides/dimensions of the
    complement (non-target) subsystems the kernel odometers over, the coset
    count, and the parallel chunk count for this many cosets.

    Depends only on ``dims`` and ``targets``, so the statevector path calls it
    with the physical system dims while the density-matrix path calls it with
    the doubled ``bra + ket`` dims (see the module docstring).
    """
    n = len(dims)
    local_dims = [dims[t] for t in targets]
    local_dim = prod(local_dims)

    strides = [prod(dims[:q]) for q in range(n)]
    target_strides = [strides[t] for t in targets]
    # Mixed-radix place values with targets[0] most-significant.
    local_places = [1] * len(targets)
    for j in range(len(targets) - 2, -1, -1):
        local_places[j] = local_places[j + 1] * local_dims[j + 1]

    local_index = np.arange(local_dim, dtype=np.int64)
    offsets = np.zeros(local_dim, dtype=np.int64)
    for j, stride in enumerate(target_strides):
        offsets += ((local_index // local_places[j]) % local_dims[j]) * stride

    target_set = set(targets)
    complement = [q for q in range(n) if q not in target_set]
    comp_strides = np.array([strides[q] for q in complement], dtype=np.int64)
    comp_dims = np.array([dims[q] for q in complement], dtype=np.int64)
    size = prod(dims)
    num_cosets = size // local_dim
    n_chunks = _choose_chunk_count(
        num_cosets,
        size,
        min_parallel_size,
        max_workers=max_workers,
        force_parallel=force_parallel,
    )
    return offsets, comp_strides, comp_dims, num_cosets, n_chunks


def _build_measurement_layout(
    dims: Sequence[int], subsystems: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Flat strides and dimensions of the measured subsystems for the kernels."""
    strides = np.array([prod(dims[:q]) for q in subsystems], dtype=np.int64)
    measured_dims = np.array([dims[q] for q in subsystems], dtype=np.int64)
    return strides, measured_dims


def _build_unitary_apply_layout(
    dims: Sequence[int],
    targets: Sequence[int],
    *,
    batch_columns: bool = False,
    max_workers: int = _MAX_THREADS,
    force_parallel: bool = False,
) -> tuple:
    """Build a matrix-application layout over the physical dimensions.

    ``batch_columns`` adds a never-targeted column dimension for a unitary
    buffer, so the matrix acts independently on every column.
    """
    if batch_columns:
        dims = (prod(dims),) + tuple(dims)
        targets = tuple(1 + target for target in targets)
    return _build_apply_layout(
        dims, targets, max_workers=max_workers, force_parallel=force_parallel
    )


def _build_superoperator_apply_layout(
    dims: Sequence[int],
    targets: Sequence[int],
    *,
    batch_columns: bool = False,
    max_workers: int = _MAX_THREADS,
    force_parallel: bool = False,
) -> tuple:
    """Build the ket-first local layout over doubled bra/ket dimensions.

    Ket targets precede bra targets so local indices match kron(M, conj(M)).
    ``batch_columns`` adds a never-targeted column dimension for an accumulated
    superoperator buffer.
    """
    dims = tuple(dims)
    n = len(dims)
    shift = int(batch_columns)
    row_dims = dims + dims
    if batch_columns:
        row_dims = (prod(dims) ** 2,) + row_dims
    super_targets = tuple(shift + n + t for t in targets) + tuple(
        shift + t for t in targets
    )
    return _build_apply_layout(
        row_dims,
        super_targets,
        _MIN_SIZE_TO_THREAD_DM,
        max_workers=max_workers,
        force_parallel=force_parallel,
    )


class _ApplyLayoutCache:
    """Reuse numerical layouts across Python executions with matching inputs.

    Retain only the latest binding's layouts, without retaining steps or states.
    Each binding gets a separate target mapping; its layout arrays are shared
    and treated as immutable by the numerical functions. Pruning this cache
    cannot change a previously bound execution, and execution never restores
    or updates the cache.
    """

    def __init__(self, build_layout: Callable[..., tuple]):
        self._build_layout = build_layout
        self._entries: dict[tuple, tuple] = {}

    def for_targets(
        self,
        dims: tuple[int, ...],
        targets: Iterable[tuple[int, ...]],
        *,
        batch_columns: bool,
        max_workers: int,
        force_parallel: bool,
    ) -> dict[tuple[int, ...], tuple]:
        """Reuse matching layouts and build the missing ones for this binding."""
        entries = {}
        layouts = {}
        for target in targets:
            key = (dims, target, batch_columns, max_workers, force_parallel)
            layout = self._entries.get(key)
            if layout is None:
                layout = self._build_layout(
                    dims,
                    target,
                    batch_columns=batch_columns,
                    max_workers=max_workers,
                    force_parallel=force_parallel,
                )
            entries[key] = layout
            layouts[target] = layout
        self._entries = entries
        return layouts
