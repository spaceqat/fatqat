"""Concrete matrix builders: fixed-gate constants, rotation/phase gates, and
dimension-generic (qudit) gate families.

See `implementation.matrix_map` for the local-matrix convention every builder here
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


def _swap_levels_rule(
    op: "ops.SwapLevels", targets: tuple[RegisterRef, ...]
) -> np.ndarray:
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


def _subspace_rx_rule(
    op: "ops.SubspaceRX", targets: tuple[RegisterRef, ...]
) -> np.ndarray:
    return subspace_rx_matrix(targets[0].register.dim, op.subspace, op.theta)


def _subspace_ry_rule(
    op: "ops.SubspaceRY", targets: tuple[RegisterRef, ...]
) -> np.ndarray:
    return subspace_ry_matrix(targets[0].register.dim, op.subspace, op.theta)


def _subspace_rz_rule(
    op: "ops.SubspaceRZ", targets: tuple[RegisterRef, ...]
) -> np.ndarray:
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
_SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex)
_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
_TDG = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)
# 2-qubit fixed gates (see module docstring for the control/target convention).
_CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
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


def u_matrix(theta: float, phi: float, lam: float) -> np.ndarray:
    """Build the Qiskit-compatible U(theta, phi, lam) matrix."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [
            [c, -np.exp(1j * lam) * s],
            [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
        ],
        dtype=complex,
    )


def _u(op: ops.U) -> np.ndarray:
    return u_matrix(op.theta, op.phi, op.lam)


def _u1(op: ops.U1) -> np.ndarray:
    return _phase(ops.Phase(op.lam))


def _u2(op: ops.U2) -> np.ndarray:
    return u_matrix(np.pi / 2, op.phi, op.lam)


def _u3(op: ops.U3) -> np.ndarray:
    return u_matrix(op.theta, op.phi, op.lam)


def _cphase(op: ops.CPhase) -> np.ndarray:
    """Build the CPhase matrix from the operation's angle."""
    theta = op.theta
    return np.diag([1, 1, 1, np.exp(1j * theta)]).astype(complex)


# Batched twins of the parametric built-in rules, for sweep replay. Each takes
# its angle fields as scalars or equal-length 1-D arrays and returns an
# ``(N, d, d)`` stack whose row ``i`` is the scalar builder's matrix for row
# ``i``'s values. The arithmetic mirrors the scalar builder term by term, so
# the scalar builder stays the single source of truth for the matrix.


def _matrix_stack(dim: int, *fields: object, identity: bool = False) -> np.ndarray:
    """Allocate the ``(N, dim, dim)`` output for the broadcast field rows."""
    (rows,) = np.broadcast_shapes(*(np.shape(field) for field in fields))
    stack = np.zeros((rows, dim, dim), dtype=complex)
    if identity:
        stack[:] = np.eye(dim, dtype=complex)
    return stack


def _rx_batch(theta: np.ndarray) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    m = _matrix_stack(2, theta)
    m[:, 0, 0] = c
    m[:, 0, 1] = -1j * s
    m[:, 1, 0] = -1j * s
    m[:, 1, 1] = c
    return m


def _ry_batch(theta: np.ndarray) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    m = _matrix_stack(2, theta)
    m[:, 0, 0] = c
    m[:, 0, 1] = -s
    m[:, 1, 0] = s
    m[:, 1, 1] = c
    return m


def _rz_batch(theta: np.ndarray) -> np.ndarray:
    m = _matrix_stack(2, theta)
    m[:, 0, 0] = np.exp(-1j * theta / 2)
    m[:, 1, 1] = np.exp(1j * theta / 2)
    return m


def _phase_batch(theta: np.ndarray) -> np.ndarray:
    m = _matrix_stack(2, theta)
    m[:, 0, 0] = 1
    m[:, 1, 1] = np.exp(1j * theta)
    return m


def _u_matrix_batch(theta: object, phi: object, lam: object) -> np.ndarray:
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    m = _matrix_stack(2, theta, phi, lam)
    m[:, 0, 0] = c
    m[:, 0, 1] = -np.exp(1j * lam) * s
    m[:, 1, 0] = np.exp(1j * phi) * s
    m[:, 1, 1] = np.exp(1j * (phi + lam)) * c
    return m


def _cphase_batch(theta: np.ndarray) -> np.ndarray:
    m = _matrix_stack(4, theta, identity=True)
    m[:, 3, 3] = np.exp(1j * theta)
    return m


def _subspace_rx_batch(
    dim: int, subspace: tuple[int, int], theta: np.ndarray
) -> np.ndarray:
    j, k = subspace
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    m = _matrix_stack(dim, theta, identity=True)
    m[:, j, j] = c
    m[:, k, k] = c
    m[:, j, k] = -1j * s
    m[:, k, j] = -1j * s
    return m


def _subspace_ry_batch(
    dim: int, subspace: tuple[int, int], theta: np.ndarray
) -> np.ndarray:
    j, k = subspace
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    m = _matrix_stack(dim, theta, identity=True)
    m[:, j, j] = c
    m[:, k, k] = c
    m[:, j, k] = -s
    m[:, k, j] = s
    return m


def _subspace_rz_batch(
    dim: int, subspace: tuple[int, int], theta: np.ndarray
) -> np.ndarray:
    j, k = subspace
    m = _matrix_stack(dim, theta, identity=True)
    m[:, j, j] = np.exp(-1j * theta / 2)
    m[:, k, k] = np.exp(1j * theta / 2)
    return m


# Batched rules for sweep replay, registered beside the scalar rules in
# `registry`. Each receives a field accessor (a float64 column for a
# `Parameter` field, the declared value otherwise) and the program targets.
# They are module-level functions so implementation maps stay picklable.


def _rx_batch_rule(field, _targets) -> np.ndarray:
    return _rx_batch(field("theta"))


def _ry_batch_rule(field, _targets) -> np.ndarray:
    return _ry_batch(field("theta"))


def _rz_batch_rule(field, _targets) -> np.ndarray:
    return _rz_batch(field("theta"))


def _phase_batch_rule(field, _targets) -> np.ndarray:
    return _phase_batch(field("theta"))


def _u_batch_rule(field, _targets) -> np.ndarray:
    return _u_matrix_batch(field("theta"), field("phi"), field("lam"))


def _u1_batch_rule(field, _targets) -> np.ndarray:
    return _phase_batch(field("lam"))


def _u2_batch_rule(field, _targets) -> np.ndarray:
    return _u_matrix_batch(np.pi / 2, field("phi"), field("lam"))


def _cphase_batch_rule(field, _targets) -> np.ndarray:
    return _cphase_batch(field("theta"))


def _subspace_rx_batch_rule(field, targets) -> np.ndarray:
    return _subspace_rx_batch(
        targets[0].register.dim, field("subspace"), field("theta")
    )


def _subspace_ry_batch_rule(field, targets) -> np.ndarray:
    return _subspace_ry_batch(
        targets[0].register.dim, field("subspace"), field("theta")
    )


def _subspace_rz_batch_rule(field, targets) -> np.ndarray:
    return _subspace_rz_batch(
        targets[0].register.dim, field("subspace"), field("theta")
    )
