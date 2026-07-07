"""Concrete matrix builders: fixed-gate constants, rotation/phase gates, and
dimension-generic (qudit) gate families.

See `implementation.base` for the local-matrix convention every builder here
follows (target/control operand ordering, MSB/LSB layout).

Adding a new gate's rule (registered in `implementation.registry`):
    - Gate reads its own fields (e.g. `Shift.power`, `RX.theta`): write a bare
      ``def _foo_rule(op: ops.Foo, targets: tuple[RegisterRef, ...]) -> np.ndarray``
      and register it directly. The parameter must be named exactly
      ``targets`` (or the rule must accept ``**kwargs``) - that literal name
      is how `_wrap_rule`/`_callable_wants_targets` detect that the rule wants
      it; a differently-named parameter is silently called as ``rule(op)``
      instead and only fails the first time the rule runs.
    - Gate's matrix depends only on target dimensions, not on any of its own
      fields (e.g. `Fourier`): write ``def _foo_rule(dims: tuple[int, ...]) ->
      np.ndarray`` and register it wrapped as ``_DimMatrix(_foo_rule)``.
"""

from __future__ import annotations

import numpy as np

from .. import operations as ops
from ..registers import RegisterRef


def shift_matrix(dim: int, power: int) -> np.ndarray:
    """Generalized-Pauli shift: ``|k> -> |(k + power) mod dim>``.

    Returns a ``dim x dim`` permutation matrix. ``power`` is reduced modulo
    ``dim``, so ``shift_matrix(3, 5)`` equals ``shift_matrix(3, 2)``.
    """
    power %= dim
    m = np.zeros((dim, dim), dtype=complex)
    for k in range(dim):
        m[(k + power) % dim, k] = 1.0
    return m


def clock_matrix(dim: int, power: int) -> np.ndarray:
    """Generalized-Pauli clock: diag(omega^(k*power)), omega = e^{2πi/dim}."""
    power %= dim
    omega = np.exp(2j * np.pi / dim)
    return np.diag([omega ** ((k * power) % dim) for k in range(dim)]).astype(complex)


def sum_matrix(dims: tuple[int, ...]) -> np.ndarray:
    """Controlled mod-d add on two equal-dimension subsystems.

    Local index is ``i*d + j`` with operand 0 (control ``i``) the MSB. Maps
    ``|i, j> -> |i, (i + j) mod d>``.
    """
    if len(dims) != 2 or dims[0] != dims[1]:
        raise ValueError(
            f"default Sum requires two equal-dimension targets, got {dims}"
        )
    d = dims[0]
    m = np.zeros((d * d, d * d), dtype=complex)
    for i in range(d):
        for j in range(d):
            m[i * d + (i + j) % d, i * d + j] = 1.0
    return m


def _shift_rule(op: "ops.Shift", targets: tuple[RegisterRef, ...]) -> np.ndarray:
    return shift_matrix(targets[0].register.dim, op.power)


def _clock_rule(op: "ops.Clock", targets: tuple[RegisterRef, ...]) -> np.ndarray:
    return clock_matrix(targets[0].register.dim, op.power)


def swap_levels_matrix(dim: int, j: int, k: int) -> np.ndarray:
    """Level-transposition permutation: swaps basis levels j and k, identity
    elsewhere. Returns a dim x dim matrix.
    """
    m = np.eye(dim, dtype=complex)
    m[[j, k]] = m[[k, j]]
    return m


def _swap_levels_rule(op: "ops.SwapLevels", targets: tuple[RegisterRef, ...]) -> np.ndarray:
    return swap_levels_matrix(targets[0].register.dim, op.j, op.k)


def fourier_matrix(dim: int) -> np.ndarray:
    """Single-qudit discrete Fourier transform: F[m,n] = omega^(m*n)/sqrt(d)."""
    omega = np.exp(2j * np.pi / dim)
    idx = np.arange(dim)
    return (omega ** np.outer(idx, idx)) / np.sqrt(dim)


def fourierdg_matrix(dim: int) -> np.ndarray:
    """Conjugate transpose of fourier_matrix(dim)."""
    return fourier_matrix(dim).conj().T


def _fourier_rule(dims: tuple[int, ...]) -> np.ndarray:
    return fourier_matrix(dims[0])


def _fourierdg_rule(dims: tuple[int, ...]) -> np.ndarray:
    return fourierdg_matrix(dims[0])


def subspace_rx_matrix(dim: int, subspace: tuple[int, int], theta: float) -> np.ndarray:
    """RX(theta) embedded in the 2-level subspace (j, k), identity elsewhere."""
    j, k = subspace
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    m = np.eye(dim, dtype=complex)
    m[j, j] = c
    m[k, k] = c
    m[j, k] = -1j * s
    m[k, j] = -1j * s
    return m


def subspace_ry_matrix(dim: int, subspace: tuple[int, int], theta: float) -> np.ndarray:
    """RY(theta) embedded in the 2-level subspace (j, k), identity elsewhere."""
    j, k = subspace
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    m = np.eye(dim, dtype=complex)
    m[j, j] = c
    m[k, k] = c
    m[j, k] = -s
    m[k, j] = s
    return m


def subspace_rz_matrix(dim: int, subspace: tuple[int, int], theta: float) -> np.ndarray:
    """RZ(theta) embedded in the 2-level subspace (j, k), identity elsewhere."""
    j, k = subspace
    m = np.eye(dim, dtype=complex)
    m[j, j] = np.exp(-1j * theta / 2)
    m[k, k] = np.exp(1j * theta / 2)
    return m


def _subspace_rx_rule(op: "ops.SubspaceRX", targets: tuple[RegisterRef, ...]) -> np.ndarray:
    return subspace_rx_matrix(targets[0].register.dim, op.subspace, op.theta)


def _subspace_ry_rule(op: "ops.SubspaceRY", targets: tuple[RegisterRef, ...]) -> np.ndarray:
    return subspace_ry_matrix(targets[0].register.dim, op.subspace, op.theta)


def _subspace_rz_rule(op: "ops.SubspaceRZ", targets: tuple[RegisterRef, ...]) -> np.ndarray:
    return subspace_rz_matrix(targets[0].register.dim, op.subspace, op.theta)


def cclock_matrix(dims: tuple[int, int], power: int) -> np.ndarray:
    """Controlled-Clock: |i, k> -> omega_t^((i*k*power) mod d_t) |i, k>,
    omega_t = exp(2*pi*i/d_t). Local index i*d_t + k (control i is the MSB).
    dims need not be equal.
    """
    d_c, d_t = dims
    power %= d_t
    omega_t = np.exp(2j * np.pi / d_t)
    diag_vals = [
        omega_t ** ((i * k * power) % d_t) for i in range(d_c) for k in range(d_t)
    ]
    return np.diag(diag_vals).astype(complex)


def _cclock_rule(op: "ops.CClock", targets: tuple[RegisterRef, ...]) -> np.ndarray:
    dims = (targets[0].register.dim, targets[1].register.dim)
    return cclock_matrix(dims, op.power)


# Module-level constant matrices (reused; do not rebuild per call).
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_I = np.eye(2, dtype=complex)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_SDG = np.array([[1, 0], [0, -1j]], dtype=complex)
_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
_TDG = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)
# 2-qubit fixed gates (see module docstring for the control/target convention).
_CX = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)
_CZ = np.diag([1, 1, 1, -1]).astype(complex)
_SWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
)
_CY = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]], dtype=complex
)
_CS = np.diag([1, 1, 1, 1j]).astype(complex)
_ISWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex
)
# 3-qubit fixed gates. Basis index bits are (operand0, operand1, operand2)
# from MSB to LSB, matching the control-first convention above.
_CCX = np.eye(8, dtype=complex)
_CCX[[6, 7]] = _CCX[[7, 6]]  # swap |110> <-> |111>: flip target iff both controls=1

_CSWAP = np.eye(8, dtype=complex)
_CSWAP[[5, 6]] = _CSWAP[[6, 5]]  # swap |101> <-> |110>: exchange targets iff control=1


def _rx(op: ops.RX) -> np.ndarray:
    """Build the RX matrix from the operation's angle."""
    theta = op.theta
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(op: ops.RY) -> np.ndarray:
    """Build the RY matrix from the operation's angle."""
    theta = op.theta
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(op: ops.RZ) -> np.ndarray:
    """Build the RZ matrix from the operation's angle."""
    theta = op.theta
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex
    )


def _phase(op: ops.Phase) -> np.ndarray:
    """Build the Phase matrix from the operation's angle."""
    theta = op.theta
    return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)


def _cphase(op: ops.CPhase) -> np.ndarray:
    """Build the CPhase matrix from the operation's angle."""
    theta = op.theta
    return np.diag([1, 1, 1, np.exp(1j * theta)]).astype(complex)
