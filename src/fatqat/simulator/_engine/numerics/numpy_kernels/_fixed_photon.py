"""NumPy numerics for states of one fixed total photon number.

Passive optics never changes the total photon number ``N``, so a state only
needs one amplitude per occupation that sums to ``N``: ``C(M + N - 1, N)``
amplitudes for ``M`` modes instead of ``(N + 1)**M``. Rows ascend like the
engine's flat index restricted to that sector, so the amplitudes are a
statevector's in-sector entries in the same order. In public order (the last
engine subsystem first) that is lexicographic order of the occupations.

No table of occupations is stored: a row's occupation and an occupation's row
are computed from ``C(m + n - 1, n)`` counts, so memory beyond the amplitudes
is one chunk of rows. Each mode may hold every photon, so dims must be at
least ``N + 1``.

A gate whose matrix conserves the photon number of its targets maps each
group of rows that agree outside the targets onto itself. The rows of a
group with local photon number ``s`` hold every local configuration of ``s``
photons once, so each group is one small block of the gate matrix. Groups
are disjoint, so gates and measurements update the amplitudes in place, like
the Numba statevector kernels.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import comb
from typing import NamedTuple

import numpy as np

from ....._backends.steps import ApplyMatrixStep
from .....errors import BackendValidationError
from ...state import _LoweredBasisState

# Entries this small in a conserving gate's off-sector blocks are round-off.
_LEAK_TOLERANCE = 1e-10
# Rows decoded at once; bounds the scratch memory of a pass over the state.
_CHUNK_ROWS = 1 << 15


class _FixedPhotonSector(NamedTuple):
    """Occupations of ``num_modes`` modes with ``photons`` photons in total.

    ``counts[m, n]`` is the number of occupations of ``m`` modes with ``n``
    photons, ``C(m + n - 1, n)``; ``len(counts) - 1`` modes at most.
    """

    num_modes: int
    photons: int
    counts: np.ndarray

    @property
    def size(self) -> int:
        """Return the number of occupations, the length of the amplitudes."""
        return int(self.counts[self.num_modes, self.photons])


class _FixedPhotonState(NamedTuple):
    """Amplitudes over the occupations of a `_FixedPhotonSector`."""

    amplitudes: np.ndarray
    sector: _FixedPhotonSector


def _sector_counts(num_modes: int, photons: int) -> np.ndarray:
    """Return ``counts[m, n] = C(m + n - 1, n)`` for ``m <= num_modes``."""
    if comb(num_modes + photons - 1, photons) > np.iinfo(np.int64).max:
        raise BackendValidationError(
            f"{num_modes} modes with {photons} photons have too many occupations "
            "to index"
        )
    counts = np.zeros((num_modes + 1, photons + 1), dtype=np.int64)
    counts[0, 0] = 1  # zero modes hold only zero photons
    for modes in range(1, num_modes + 1):
        for n in range(photons + 1):
            counts[modes, n] = comb(modes + n - 1, n)
    return counts


def _fixed_photon_sector(num_modes: int, photons: int) -> _FixedPhotonSector:
    """Return the sector of ``photons`` photons in ``num_modes`` modes."""
    return _FixedPhotonSector(num_modes, photons, _sector_counts(num_modes, photons))


def _rank_span(
    counts: np.ndarray,
    num_modes: int,
    first: int,
    span: np.ndarray,
    remaining: np.ndarray,
) -> np.ndarray:
    """Return the part of a row that columns ``first`` on contribute.

    An occupation's row counts the occupations with the same total that are
    lexicographically smaller. Those with a smaller value ``x`` at column
    ``c`` number ``counts[M - c, r] - counts[M - c, r - x]``, for the ``r``
    photons left from ``c`` on. ``span[..., j]`` holds column ``first + j``
    and ``remaining`` the photons left at ``first``. The last column adds
    nothing, since one mode holds its photons one way.
    """
    rows = np.zeros(span.shape[:-1], dtype=np.int64)
    for j in range(span.shape[-1]):
        level = span[..., j]
        within = counts[num_modes - first - j]  # 1-D lookups are cheaper
        rows += within[remaining] - within[remaining - level]
        remaining = remaining - level
    return rows


def _rank(counts: np.ndarray, occupations: np.ndarray) -> np.ndarray:
    """Return the rows of public-order ``occupations``, shape ``(..., M)``."""
    num_modes = occupations.shape[-1]
    return _rank_span(counts, num_modes, 0, occupations, occupations.sum(axis=-1))


def _unrank(
    sector: _FixedPhotonSector, rows: np.ndarray, num_columns: int | None = None
) -> np.ndarray:
    """Return the public-order occupations of ``rows``, shape ``(len, M)``.

    With ``num_columns``, only those leading columns are decoded and returned.
    """
    counts = sector.counts
    num_modes = sector.num_modes
    num_columns = num_modes if num_columns is None else num_columns
    rows = np.array(rows, dtype=np.int64)
    occupations = np.zeros((len(rows), num_columns), dtype=np.int64)
    remaining = np.full(len(rows), sector.photons, dtype=np.int64)
    # Rows stay below the sector size, so `remaining * stride + row` orders
    # rows by remaining photons first; one search then serves every r.
    stride = sector.size + 1
    r, x = np.tril_indices(sector.photons + 1, -1)
    x += 1  # (r, x) for 1 <= x <= r, ordered by r then x
    starts = np.searchsorted(r, np.arange(sector.photons + 1))
    for column in range(min(num_columns, num_modes - 1)):
        within = counts[num_modes - column]
        # Occupations smaller at this column with value x, counted as in
        # `_rank_span`; they grow with x, so the level is how many fit below.
        below = within[r] - within[r - x]
        found = np.searchsorted(r * stride + below, remaining * stride + rows, "right")
        level = found - starts[remaining]
        rows -= within[remaining] - within[remaining - level]
        remaining -= level
        occupations[:, column] = level
    if num_columns == num_modes and num_modes:
        occupations[:, -1] = remaining
    return occupations


def _chunks(size: int) -> range:
    return range(0, size, _CHUNK_ROWS)


def _allocate_fixed_photon(
    dims: Sequence[int], start: _LoweredBasisState
) -> _FixedPhotonState:
    """Return the basis state ``start`` in its photon-number sector."""
    assert isinstance(start, _LoweredBasisState), "fixed-photon runs start from levels"
    photons = sum(start.levels)
    assert min(dims, default=photons + 1) > photons, "every mode holds every photon"
    sector = _fixed_photon_sector(len(dims), photons)
    amplitudes = np.zeros(sector.size, dtype=complex)
    amplitudes[_rank(sector.counts, np.array(start.levels[::-1]))] = 1.0
    return _FixedPhotonState(amplitudes, sector)


class _GateBlocks(NamedTuple):
    """A gate's photon-number blocks, by local photon number ``s``.

    ``levels[offsets[s]:offsets[s + 1]]`` are the local occupations of ``s``
    photons, one column per target, and ``matrices[s]`` the gate's block
    between them. The targets sit at public ``columns``; they span columns
    ``first`` to ``last``, at ``columns - first`` within that span.
    """

    columns: np.ndarray
    first: int
    last: int
    offsets: np.ndarray
    levels: np.ndarray
    matrices: tuple[np.ndarray, ...]


def _gate_blocks(
    matrix: np.ndarray, targets: Sequence[int], dims: Sequence[int], photons: int
) -> _GateBlocks:
    """Check that ``matrix`` conserves photon number and list its blocks."""
    local_dims = tuple(dims[t] for t in targets)
    levels = np.indices(local_dims).reshape(len(targets), -1).T
    local_photons = levels.sum(axis=1)
    # A matrix entry linking different local photon numbers would move
    # amplitude out of the sector this state stores.
    leaks = local_photons[:, None] != local_photons[None, :]
    if np.abs(matrix[leaks]).max(initial=0.0) > _LEAK_TOLERANCE:
        raise BackendValidationError(
            "method='fixed_photon_state' needs gates that conserve the photon number; "
            "a gate matrix changes it"
        )
    # Local index order within each s; levels above the photon number never occur.
    local = np.flatnonzero(local_photons <= photons)
    local = local[np.argsort(local_photons[local], kind="stable")]
    offsets = np.searchsorted(local_photons[local], np.arange(photons + 2))
    matrices = tuple(
        np.ascontiguousarray(matrix[np.ix_(block, block)], dtype=complex)
        for block in np.split(local, offsets[1:-1])
    )
    num_modes = len(dims)
    columns = np.array([num_modes - 1 - t for t in targets], dtype=np.int64)
    return _GateBlocks(
        columns,
        int(columns.min()),
        int(columns.max()),
        offsets.astype(np.int64),
        levels[local],
        matrices,
    )


def _apply_matrix_fixed_photon(
    state: _FixedPhotonState, step: ApplyMatrixStep, dims: Sequence[int]
) -> _FixedPhotonState:
    """Apply a photon-number-conserving local matrix block by block, in place."""
    sector = state.sector
    blocks = _gate_blocks(step.matrix, step.target_indices, dims, sector.photons)
    amplitudes = state.amplitudes
    span_columns = blocks.columns - blocks.first
    for start in _chunks(sector.size):
        rows = np.arange(start, min(start + _CHUNK_ROWS, sector.size))
        # Columns after the targets do not decide groups or member rows.
        prefix = _unrank(sector, rows, blocks.last + 1)
        local = prefix[:, blocks.columns]
        local_photons = local.sum(axis=1)
        for s in np.unique(local_photons):
            first, stop = blocks.offsets[s], blocks.offsets[s + 1]
            # Each group is handled once, from its row holding the block's
            # first configuration.
            leaders = (local_photons == s) & (local == blocks.levels[first]).all(axis=1)
            span = prefix[leaders, blocks.first :]
            remaining = sector.photons - prefix[leaders, : blocks.first].sum(axis=1)
            # Members differ from their leader only from `first` to `last`,
            # and hold the same photons there, so only that span's part of
            # the row changes.
            leader_part = _rank_span(
                sector.counts, sector.num_modes, blocks.first, span, remaining
            )
            spans = np.repeat(span[:, None, :], stop - first, axis=1)
            spans[:, :, span_columns] = blocks.levels[first:stop]
            members = (rows[leaders] - leader_part)[:, None] + _rank_span(
                sector.counts, sector.num_modes, blocks.first, spans, remaining[:, None]
            )
            amplitudes[members] = amplitudes[members] @ blocks.matrices[s].T
    return state


def _fixed_photon_probabilities(state: _FixedPhotonState) -> np.ndarray:
    """Return normalized occupation probabilities, in one new array."""
    probabilities = np.abs(state.amplitudes)
    probabilities *= probabilities
    total = probabilities.sum()
    if total > 0:
        probabilities /= total
    return probabilities


def _normalize(amplitudes: np.ndarray) -> None:
    """Scale ``amplitudes`` to unit norm in place, if it is not zero."""
    norm = np.sqrt(np.vdot(amplitudes, amplitudes).real)
    if norm > 0:
        amplitudes /= norm


def _project_fixed_photon(
    state: _FixedPhotonState,
    row: int,
    measured_subsystems: Sequence[int],
    dims: Sequence[int],
) -> _FixedPhotonState:
    """Keep the rows that match ``row`` on the measured subsystems, in place."""
    del dims  # rows are not flat indices; the sector decodes them
    sector = state.sector
    columns = [sector.num_modes - 1 - q for q in measured_subsystems]
    num_columns = max(columns) + 1
    outcome = _unrank(sector, [row], num_columns)[0, columns]
    amplitudes = state.amplitudes
    for start in _chunks(sector.size):
        rows = np.arange(start, min(start + _CHUNK_ROWS, sector.size))
        levels = _unrank(sector, rows, num_columns)[:, columns]
        amplitudes[rows[(levels != outcome).any(axis=1)]] = 0.0
    _normalize(amplitudes)
    return state


def _fixed_photon_levels(
    state: _FixedPhotonState, rows: np.ndarray, subsystems: Sequence[int]
) -> np.ndarray:
    """Return the measured subsystems' levels for sampled rows."""
    sector = state.sector
    columns = [sector.num_modes - 1 - q for q in subsystems]
    return _unrank(sector, np.asarray(rows, dtype=np.int64))[:, columns]
