"""Runtime-neutral gate-matrix facts shared by kernels and plan passes.

Structure codes, the declared structure of each builtin kernel key, and the
dense embedding of a local matrix onto wider targets. This module MUST NOT
import Numba: NumPy engines and plan passes read it without the extra.

Numba's on-disk cache freezes the imported structure codes as constants.
After changing a code value, delete the Numba cache files.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import numpy as np

from ...._backends.steps import BuiltinKernelKey

# Gate-matrix structure codes. A diagonal or (phased) permutation matrix has at
# most one nonzero per row, so its application is a single multiply per amplitude
# instead of the D-term dot product a dense row needs - a machine-independent
# constant-factor win that covers most common gates (Z/S/T/RZ/CPhase/CZ are
# diagonal; X/CX/CCX/SWAP/iSwap are permutations).
_DENSE = 0
_DIAGONAL = 1
_PERMUTATION = 2


# Declared structure per canonical gate identity: the key-to-kernel table.
# The numerical runtime owns these declarations; the key carries only identity.
#
# How kernel selection flows (one classifier, one kernel family):
#   standard path   _apply_matrix_statevector calls _MatrixStructureCache.resolve.
#                   A key declared _DENSE skips the scan; everything else takes ONE
#                   `_classify_matrix` scan at plan preparation, cached per
#                   step - never per application, never per shot.
#   fallback path   _apply_unclassified_matrix handles bare Kraus matrices:
#                   the same single
#                   `_classify_matrix` scan, then the same resolved kernels.
#   compiled path   _compile_dynamic_plan bakes each gate's resolved code into
#                   the plan arrays, so the shots kernel never classifies.
#
# An entry may claim _DIAGONAL or _PERMUTATION only if that structure holds
# for EVERY parameter value of the gate (RZ is diagonal for all theta); a
# parametric gate whose structure varies claims _DENSE, which is always safe
# (RX at theta=pi happens to be a permutation, but _DENSE stays correct).
# The spec-vs-content test in test_kernel_dispatch.py enforces agreement
# with `_classify_matrix` over the whole default map.
_K = BuiltinKernelKey
_KERNEL_SPECS: dict[BuiltinKernelKey, int] = {
    # diagonal for every parameter value
    _K.I: _DIAGONAL,
    _K.Z: _DIAGONAL,
    _K.S: _DIAGONAL,
    _K.SDG: _DIAGONAL,
    _K.T: _DIAGONAL,
    _K.TDG: _DIAGONAL,
    _K.CZ: _DIAGONAL,
    _K.CS: _DIAGONAL,
    _K.RZ: _DIAGONAL,
    _K.PHASE: _DIAGONAL,
    _K.U1: _DIAGONAL,
    _K.CPHASE: _DIAGONAL,
    _K.CLOCK: _DIAGONAL,
    _K.CCLOCK: _DIAGONAL,
    _K.SUBSPACE_RZ: _DIAGONAL,
    _K.PHASE_SHIFTER: _DIAGONAL,
    # exactly one nonzero per row, for every parameter value
    _K.X: _PERMUTATION,
    _K.Y: _PERMUTATION,
    _K.CX: _PERMUTATION,
    _K.CY: _PERMUTATION,
    _K.SWAP: _PERMUTATION,
    _K.ISWAP: _PERMUTATION,
    _K.CCX: _PERMUTATION,
    _K.CSWAP: _PERMUTATION,
    _K.SHIFT: _PERMUTATION,
    _K.SUM: _PERMUTATION,
    _K.SWAP_LEVELS: _PERMUTATION,
    # generically dense
    _K.H: _DENSE,
    _K.SX: _DENSE,
    _K.RX: _DENSE,
    _K.RY: _DENSE,
    _K.U: _DENSE,
    _K.U2: _DENSE,
    _K.U3: _DENSE,
    _K.FOURIER: _DENSE,
    _K.FOURIERDG: _DENSE,
    _K.SUBSPACE_RX: _DENSE,
    _K.SUBSPACE_RY: _DENSE,
    _K.BEAM_SPLITTER: _DENSE,
}


def _local_places(dims: Sequence[int]) -> list[int]:
    """Mixed-radix place values with element 0 most significant."""
    places = [1] * len(dims)
    for i in range(len(dims) - 2, -1, -1):
        places[i] = places[i + 1] * dims[i + 1]
    return places


def _embedding(
    from_targets: tuple[int, ...], to_targets: tuple[int, ...], dims: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index maps embedding a ``from_targets`` operator into ``to_targets``.

    Returns ``(local, base, offsets)``: ``local[x]`` is the narrow row wide
    index ``x`` selects, ``base[x]`` is ``x`` with the narrow digits cleared,
    and ``offsets[r]`` is the wide offset of narrow index ``r``. Both index
    spaces are most-significant-first in their own target order.
    """
    to_dims = [dims[t] for t in to_targets]
    from_dims = [dims[t] for t in from_targets]
    to_places = _local_places(to_dims)
    from_places = _local_places(from_dims)
    positions = [to_targets.index(t) for t in from_targets]

    wide = np.arange(prod(to_dims), dtype=np.int64)
    narrow = np.arange(prod(from_dims), dtype=np.int64)
    local = np.zeros_like(wide)
    offsets = np.zeros_like(narrow)
    for i, position in enumerate(positions):
        local += ((wide // to_places[position]) % to_dims[position]) * from_places[i]
        offsets += ((narrow // from_places[i]) % from_dims[i]) * to_places[position]
    return local, wide - offsets[local], offsets


def _embed_dense(
    matrix: np.ndarray,
    from_targets: tuple[int, ...],
    to_targets: tuple[int, ...],
    dims: Sequence[int],
) -> np.ndarray:
    """Widen a dense matrix onto ``to_targets``, tensoring identity onto the rest."""
    local, base, offsets = _embedding(from_targets, to_targets, dims)
    size = len(local)
    out = np.zeros((size, size), dtype=np.complex128)
    rows = np.arange(size, dtype=np.int64)
    for column, offset in enumerate(offsets):
        out[rows, base + offset] = matrix[local, column]
    return out
