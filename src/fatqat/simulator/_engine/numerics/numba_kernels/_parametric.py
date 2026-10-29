"""Compiled matrices of the built-in parametric gates.

A compiled workflow fills a parametric gate's matrix from parameter values
inside its kernel. Each formula mirrors the gate's scalar rule in
`implementation.matrices`; `_FORMULAS` names the fields it reads.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from ....._backends.steps import BuiltinKernelKey

_K = BuiltinKernelKey
# Formula id and the operation fields it reads, per built-in kernel key.
_FORMULAS: dict[BuiltinKernelKey, tuple[int, tuple[str, ...]]] = {
    _K.RX: (0, ("theta",)),
    _K.RY: (1, ("theta",)),
    _K.RZ: (2, ("theta",)),
    _K.PHASE: (3, ("theta",)),
    _K.U: (4, ("theta", "phi", "lam")),
    _K.U1: (3, ("lam",)),
    _K.U2: (5, ("phi", "lam")),
    _K.U3: (4, ("theta", "phi", "lam")),
    _K.CPHASE: (6, ("theta",)),
    _K.SUBSPACE_RX: (7, ("theta",)),
    _K.SUBSPACE_RY: (8, ("theta",)),
    _K.SUBSPACE_RZ: (9, ("theta",)),
}


@njit(cache=True)
def _fill_parametric_matrix(
    formula, a0, a1, a2, j, k, out
) -> None:  # pragma: no cover - compiled by Numba
    """Write one gate's ``(d, d)`` matrix into ``out`` from its field values.

    ``a0``-``a2`` are the formula's fields in `_FORMULAS` order; ``j`` and
    ``k`` are the levels of a subspace rotation.
    """
    out[:, :] = 0.0
    if formula == 3:  # Phase, U1
        out[0, 0] = 1.0
        out[1, 1] = np.exp(1j * a0)
    elif formula == 6:  # CPhase
        for r in range(3):
            out[r, r] = 1.0
        out[3, 3] = np.exp(1j * a0)
    elif 4 <= formula <= 5:  # U, U3; U2 has theta = pi / 2
        theta, phi, lam = (a0, a1, a2) if formula == 4 else (np.pi / 2, a0, a1)
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        out[0, 0] = c
        out[0, 1] = -np.exp(1j * lam) * s
        out[1, 0] = np.exp(1j * phi) * s
        out[1, 1] = np.exp(1j * (phi + lam)) * c
    else:  # RX, RY, RZ (0-2) and their subspace forms (7-9) on levels j, k
        for r in range(out.shape[0]):
            out[r, r] = 1.0
        axis = formula if formula < 3 else formula - 7
        if formula < 3:
            j, k = 0, 1
        if axis == 2:
            out[j, j] = np.exp(-1j * a0 / 2)
            out[k, k] = np.exp(1j * a0 / 2)
        else:
            c, s = np.cos(a0 / 2), np.sin(a0 / 2)
            out[j, j] = c
            out[k, k] = c
            out[j, k] = -1j * s if axis == 0 else -s
            out[k, j] = -1j * s if axis == 0 else s
