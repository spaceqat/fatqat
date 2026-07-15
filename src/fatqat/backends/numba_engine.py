"""Numba engine: drop-in JIT-optimized sibling of `NumpyEngine`.

`NumbaEngine` keeps `NumpyEngine`'s orchestration, seam, and semantics and
replaces the per-gate execution with Numba-compiled kernels plus two
engine-internal optimizations. Nothing outside this module changes: the
engine consumes the same lowered plan through the same
``initialize``/``run``/``apply`` surface.

Optimization layers (all engine-internal):

1. In-place specialized kernels. For qubit-only registers, gates execute
   in place on the flat state with bit-shift pair enumeration - no output
   buffer, no index-decode arithmetic. The generic fallback (mixed-radix
   qudits, dense multi-target gates) ping-pongs between the state and one
   scratch buffer allocated per ``initialize()``, so no per-gate allocation
   remains anywhere.
2. Exact matrix classification. Each ``ApplyMatrixStep`` matrix is
   classified structurally at apply time using EXACT comparisons (``== 0``,
   ``== 1`` - never a tolerance): exactly-diagonal matrices (RZ, Phase, CZ)
   run a multiply-only sweep; two-qubit controlled-U blocks (CX, CY, CRZ)
   touch only the control=1 half of the state. Because skipped terms are
   exact zeros and copied amplitudes are multiplied by exact ones, the
   specialized results are bit-identical to the generic kernel's.

Gate fusion was prototyped here and removed for now: fused dense blocks fell
back to the generic (out-of-place) kernel, which only pays off once the
state exceeds the CPU cache (n >= ~22) and slows cache-resident sizes down.
Reintroducing it requires an in-place fused-block kernel first.

Accuracy contract: complex128 everywhere, no ``fastmath``, no tolerances in
classification, no approximations. Execution is bit-identical to running
the same kernels gate by gate.

Density-matrix support reuses the same kernels: for ``rho[a, b]`` flattened
C-order (``a*N + b``), the ket side of subsystem ``q`` lives at bit
``n + q`` and the bra side at bit ``q``, so ``U rho U^dagger`` is two
in-place passes of the same qubit kernels (``U`` on ket bits, ``conj(U)``
on bra bits).

Requirements and caveats:

- ``numba`` must be installed; it is deliberately NOT a package dependency,
  so importing this module (and only this module) fails without it.
- First call per kernel signature pays JIT compilation; ``cache=True``
  persists compiled artifacts across processes.
- `parallel.py` workers construct `NumpyEngine` (unchanged by this module),
  so dynamic multi-shot parallel batches run NumPy kernels on the plan -
  correct, just not accelerated; prefer ``parallel_mode="serial"``
  for dynamic multi-shot runs on this engine.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numba import njit, prange

from ..implementation.matrices import shift_matrix
from .engine_contract import _EngineConfig
from .numpy_engine import (
    NumpyEngine,
    _digit,
    _strides,
)
from .steps import ApplyMatrixStep

# Below this flat-buffer size the prange thread fork/join costs more than it
# saves; measured crossover is a few thousand amplitudes on desktop CPUs.
_PARALLEL_MIN_SIZE = 1 << 13


class NumbaEngine(NumpyEngine):
    """`NumpyEngine` with JIT-compiled, specialized gate execution.

    Public surface and semantics are inherited unchanged. See the module
    docstring for the optimization layers and the accuracy contract.

    Args:
        config: Optional execution-strategy options (as `NumpyEngine`).
        state_semantics: ``"statevector"`` or ``"density_matrix"``.
    """

    def __init__(
        self,
        config: _EngineConfig | None = None,
        *,
        state_semantics: str,
    ) -> None:
        super().__init__(config, state_semantics=state_semantics)
        self._qubit_only = False
        self._scratch: np.ndarray | None = None
        self._plan_cache: dict = {}

    # --- state lifecycle -------------------------------------------------

    def initialize(self, system_dims: Sequence[int], n_clbits: int = 0) -> None:
        """Configure dimensions, reset the state, and prepare work buffers."""
        super().initialize(system_dims, n_clbits)
        self._qubit_only = all(d == 2 for d in self._dims)
        # One scratch buffer for the whole run: the generic kernel ping-pongs
        # between state and scratch instead of allocating per gate.
        self._scratch = np.empty(self._state.size, dtype=np.complex128)
        self._plan_cache = {}

    # --- per-gate dispatch -------------------------------------------------

    def apply(self, step: ApplyMatrixStep) -> None:
        """Evolve the state by one resolved matrix step (classified, in place)."""
        self._require_state()
        self._apply_matrix(step.matrix, step.target_indices)

    def _apply_matrix(self, matrix: np.ndarray, targets: tuple[int, ...]) -> None:
        """Classify the matrix exactly and route it to the cheapest kernel.

        Classification is O(dt^2) on a <=16-dim local matrix - noise next to
        any state sweep - so it runs per call and needs no caching.
        """
        m = np.ascontiguousarray(matrix, dtype=np.complex128)
        if self._qubit_only:
            n = len(self._dims)
            flat = self._state.reshape(-1)
            par = flat.size >= _PARALLEL_MIN_SIZE
            # (bit offset, operator) passes: one for a statevector; ket then
            # bra (conjugated) for a density matrix.
            if self._state_semantics == "statevector":
                passes = ((0, m),)
            else:
                passes = ((n, m), (0, m.conj()))

            diag = _exact_diagonal(m)
            if diag is not None:
                for off, mm in passes:
                    d = np.ascontiguousarray(np.diagonal(mm))
                    pos = np.array([off + q for q in targets], dtype=np.int64)
                    (_k_diag_parallel if par else _k_diag_serial)(flat, d, pos)
                return
            if len(targets) == 1:
                for off, mm in passes:
                    (_k_1q_parallel if par else _k_1q_serial)(
                        flat, mm, off + targets[0]
                    )
                return
            if len(targets) == 2 and _exact_controlled(m) is not None:
                # The conjugate of a controlled-V block is controlled-conj(V),
                # so each pass extracts V from its own (possibly conjugated) mm.
                for off, mm in passes:
                    vv = np.ascontiguousarray(mm[2:, 2:])
                    (_k_ctrl_1q_parallel if par else _k_ctrl_1q_serial)(
                        flat, vv, off + targets[0], off + targets[1]
                    )
                return
        self._apply_generic(m, targets)

    def _apply_generic(self, m: np.ndarray, targets: tuple[int, ...]) -> None:
        """Dense fallback: offset-planned kernel, ping-ponging with scratch."""
        size = self._state.size
        par = size >= _PARALLEL_MIN_SIZE
        kernel = _apply_compiled_parallel if par else _apply_compiled_serial
        if self._state_semantics == "statevector":
            plan = self._cached_plan("sv", targets)
            out = self._scratch
            kernel(self._state, out, m, *plan)
            self._scratch = self._state
            self._state = out
        else:
            # Two passes state -> scratch -> state: the state object (and its
            # (N, N) view) is preserved, scratch ends up unchanged too.
            flat = self._state.reshape(-1)
            ket = self._cached_plan("dm_ket", targets)
            bra = self._cached_plan("dm_bra", targets)
            kernel(flat, self._scratch, m, *ket)
            kernel(self._scratch, flat, m.conj(), *bra)

    def _cached_plan(self, kind: str, targets: tuple[int, ...]):
        """Offset plan for the generic kernel, cached per (kind, targets)."""
        key = (kind, targets)
        plan = self._plan_cache.get(key)
        if plan is None:
            size = int(np.prod(self._dims)) if self._dims else 1
            if kind == "sv":
                plan = _plan_local_apply(targets, self._dims)
            elif kind == "dm_ket":
                plan = _plan_local_apply(
                    targets, self._dims, scale=size, extra=((size, 1),)
                )
            else:  # dm_bra
                plan = _plan_local_apply(targets, self._dims, extra=((size, size),))
            self._plan_cache[key] = plan
        return plan

    # --- reset (statevector shift-back goes through the fast dispatch) ----

    def reset_subsystems(
        self, indices: Sequence[int], rng: np.random.Generator | None = None
    ) -> None:
        """Reprepare a group of subsystems in ``|0>`` (see `NumpyEngine`).

        Statevector semantics: identical branch sampling (exactly one rng
        draw, via the inherited NumPy collapse) with the shift-back routed
        through the classified in-place kernels. Density-matrix semantics:
        inherited deterministic partial-trace channel.
        """
        if self._state_semantics != "statevector":
            super().reset_subsystems(indices, rng)
            return
        self._require_state()
        if len(indices) < 1:
            raise ValueError("reset_subsystems requires at least one index")
        flat_idx = self.collapse(tuple(indices), rng)
        for index in indices:
            outcome = _digit(flat_idx, index, self._dims)
            if outcome != 0:
                inv = shift_matrix(self._dims[index], -outcome)
                self._apply_matrix(inv, (index,))


# --- exact structural classification (no tolerances - see accuracy contract)


def _exact_diagonal(m: np.ndarray) -> np.ndarray | None:
    """Return the diagonal if every off-diagonal entry is exactly zero."""
    if not m[~np.eye(m.shape[0], dtype=bool)].any():
        return np.diagonal(m)
    return None


def _exact_controlled(m: np.ndarray) -> np.ndarray | None:
    """Return V if a 4x4 matrix is exactly ``|0><0| (x) I + |1><1| (x) V``.

    The control is the local MSB, i.e. ``targets[0]`` under the step
    convention - which is how controlled gates lower (``CX = (control,
    target)``).
    """
    if m.shape != (4, 4):
        return None
    if (
        m[0, 0] == 1.0
        and m[1, 1] == 1.0
        and not m[0, 1]
        and not m[1, 0]
        and not np.any(m[:2, 2:])
        and not np.any(m[2:, :2])
    ):
        return m[2:, 2:]
    return None


# --- compiled kernels ------------------------------------------------------
# All kernels: complex128 in place, IEEE-strict (no fastmath), serial and
# prange twins selected by flat-buffer size.


@njit(cache=True)
def _k_1q_serial(state, m, pos):
    """In-place single-qubit dense apply at bit ``pos`` of a flat buffer."""
    half = state.size >> 1
    low_mask = (1 << pos) - 1
    bit = 1 << pos
    for k in range(half):
        i0 = ((k >> pos) << (pos + 1)) | (k & low_mask)
        i1 = i0 | bit
        a0 = state[i0]
        a1 = state[i1]
        state[i0] = m[0, 0] * a0 + m[0, 1] * a1
        state[i1] = m[1, 0] * a0 + m[1, 1] * a1


@njit(cache=True, parallel=True)
def _k_1q_parallel(state, m, pos):
    half = state.size >> 1
    low_mask = (1 << pos) - 1
    bit = 1 << pos
    for k in prange(half):
        i0 = ((k >> pos) << (pos + 1)) | (k & low_mask)
        i1 = i0 | bit
        a0 = state[i0]
        a1 = state[i1]
        state[i0] = m[0, 0] * a0 + m[0, 1] * a1
        state[i1] = m[1, 0] * a0 + m[1, 1] * a1


@njit(cache=True)
def _k_diag_serial(state, diag, positions):
    """In-place diagonal apply; ``positions[0]`` is the local MSB."""
    kbits = positions.size
    for i in range(state.size):
        idx = 0
        for t in range(kbits):
            idx = (idx << 1) | ((i >> positions[t]) & 1)
        state[i] = state[i] * diag[idx]


@njit(cache=True, parallel=True)
def _k_diag_parallel(state, diag, positions):
    kbits = positions.size
    for i in prange(state.size):
        idx = 0
        for t in range(kbits):
            idx = (idx << 1) | ((i >> positions[t]) & 1)
        state[i] = state[i] * diag[idx]


@njit(cache=True)
def _k_ctrl_1q_serial(state, v, cpos, tpos):
    """In-place controlled-V: touches only the control=1 half of the state."""
    p1 = cpos if cpos < tpos else tpos
    p2 = tpos if cpos < tpos else cpos
    quarter = state.size >> 2
    m1 = (1 << p1) - 1
    m2 = (1 << p2) - 1
    cbit = 1 << cpos
    tbit = 1 << tpos
    for k in range(quarter):
        i = ((k >> p1) << (p1 + 1)) | (k & m1)
        i = ((i >> p2) << (p2 + 1)) | (i & m2)
        i0 = i | cbit
        i1 = i0 | tbit
        a0 = state[i0]
        a1 = state[i1]
        state[i0] = v[0, 0] * a0 + v[0, 1] * a1
        state[i1] = v[1, 0] * a0 + v[1, 1] * a1


@njit(cache=True, parallel=True)
def _k_ctrl_1q_parallel(state, v, cpos, tpos):
    p1 = cpos if cpos < tpos else tpos
    p2 = tpos if cpos < tpos else cpos
    quarter = state.size >> 2
    m1 = (1 << p1) - 1
    m2 = (1 << p2) - 1
    cbit = 1 << cpos
    tbit = 1 << tpos
    for k in prange(quarter):
        i = ((k >> p1) << (p1 + 1)) | (k & m1)
        i = ((i >> p2) << (p2 + 1)) | (i & m2)
        i0 = i | cbit
        i1 = i0 | tbit
        a0 = state[i0]
        a1 = state[i1]
        state[i0] = v[0, 0] * a0 + v[0, 1] * a1
        state[i1] = v[1, 0] * a0 + v[1, 1] * a1


@njit(cache=True)
def _apply_compiled_serial(state, out, m, local_offsets, rest_dims, rest_strides):
    """Generic dense local apply (mixed radix, any target count), out of place.

    Every flat index splits uniquely into (rest digits, local digit), so each
    ``out`` entry is written exactly once and no zero-fill is needed.
    """
    n_rest = 1
    for q in range(rest_dims.size):
        n_rest *= rest_dims[q]
    dt = local_offsets.size
    for r in range(n_rest):
        # Decode r with a running divisor: parfors forbids mutating any
        # variable derived from the parallel index, so no `rr //= d` here.
        base = 0
        div = 1
        for q in range(rest_dims.size):
            base += ((r // div) % rest_dims[q]) * rest_strides[q]
            div *= rest_dims[q]
        for i in range(dt):
            acc = 0.0 + 0.0j
            for j in range(dt):
                acc += m[i, j] * state[base + local_offsets[j]]
            out[base + local_offsets[i]] = acc


@njit(cache=True, parallel=True)
def _apply_compiled_parallel(state, out, m, local_offsets, rest_dims, rest_strides):
    """Parallel twin of `_apply_compiled_serial` (independent rest blocks)."""
    n_rest = 1
    for q in range(rest_dims.size):
        n_rest *= rest_dims[q]
    dt = local_offsets.size
    for r in prange(n_rest):
        base = 0
        div = 1
        for q in range(rest_dims.size):
            base += ((r // div) % rest_dims[q]) * rest_strides[q]
            div *= rest_dims[q]
        for i in range(dt):
            acc = 0.0 + 0.0j
            for j in range(dt):
                acc += m[i, j] * state[base + local_offsets[j]]
            out[base + local_offsets[i]] = acc


def _plan_local_apply(
    targets: Sequence[int],
    dims: Sequence[int],
    scale: int = 1,
    extra: tuple[tuple[int, int], ...] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the flat-offset plan consumed by the generic compiled kernels.

    Args:
        targets: Flat subsystem indices the local matrix acts on;
            ``targets[0]`` is the matrix's most-significant local digit
            (same convention as the NumPy kernels).
        dims: Per-subsystem dimensions of the register.
        scale: Multiplier applied to every subsystem stride (``N`` for the
            density-matrix ket side, else 1).
        extra: ``(dim, stride)`` pseudo-subsystems appended to the rest
            enumeration (the opposite-side index of a density matrix).

    Returns:
        ``(local_offsets, rest_dims, rest_strides)`` int64 arrays:
        ``local_offsets[j]`` is the flat offset of local basis state ``j``,
        and the rest arrays enumerate every untouched digit.
    """
    strides = _strides(dims)
    dt = 1
    for t in targets:
        dt *= dims[t]

    local_offsets = np.zeros(dt, dtype=np.int64)
    for j in range(dt):
        rem = j
        for t in reversed(tuple(targets)):  # peel the least-significant local digit
            local_offsets[j] += (rem % dims[t]) * strides[t] * scale
            rem //= dims[t]

    target_set = set(targets)
    rest = [(dims[q], strides[q] * scale) for q in range(len(dims)) if q not in target_set]
    rest.extend(extra)
    if rest:
        rest_dims = np.array([d for d, _s in rest], dtype=np.int64)
        rest_strides = np.array([s for _d, s in rest], dtype=np.int64)
    else:
        rest_dims = np.ones(1, dtype=np.int64)
        rest_strides = np.zeros(1, dtype=np.int64)
    return local_offsets, rest_dims, rest_strides
