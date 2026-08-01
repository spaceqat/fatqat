"""Key-driven numba kernel dispatch: spec/content agreement and parity."""

import numpy as np
import pytest

import fatqat as fq
from fatqat.simulator import Simulator
from fatqat._backends.steps import ApplyChannelStep, ApplyMatrixStep, BuiltinKernelKey
from fatqat.implementation import default_matrix_implementation_map

numba = pytest.importorskip("numba")

# pylint: disable=wrong-import-position  # these need the importorskip above
from fatqat.simulator._engine.nb import (
    _classify_matrix,
    _DENSE,
    _fuse_gate_channels,
    _KERNEL_SPECS,
    _superop_csr,
    NumbaSVEngine,
)
from fatqat.simulator._engine.np import NumpySVEngine
from fatqat.noise import Depolarizing
from fatqat.noise.catalog import depolarizing_rule
from fatqat.noise.nb import _kraus_stack, _kraus_superop_kernel

# pylint: enable=wrong-import-position


# One representative applied instance per default-map gate, with generic
# parameter values (structure at special angles may be *more* special than
# the declared code - RX(pi) is a permutation - which is exactly what the
# spec-vs-content comparison below is careful about).
_QUBIT = fq.QuantumRegister(3)
_QUTRIT = fq.QuantumRegister(2, dim=3)
_GATE_CASES = [
    (fq.ops.X, (_QUBIT[0],)),
    (fq.ops.Y, (_QUBIT[0],)),
    (fq.ops.Z, (_QUBIT[0],)),
    (fq.ops.H, (_QUBIT[0],)),
    (fq.ops.I, (_QUBIT[0],)),
    (fq.ops.S, (_QUBIT[0],)),
    (fq.ops.Sdg, (_QUBIT[0],)),
    (fq.ops.SX, (_QUBIT[0],)),
    (fq.ops.T, (_QUBIT[0],)),
    (fq.ops.Tdg, (_QUBIT[0],)),
    (fq.ops.CX, (_QUBIT[0], _QUBIT[1])),
    (fq.ops.CZ, (_QUBIT[0], _QUBIT[1])),
    (fq.ops.Swap, (_QUBIT[0], _QUBIT[1])),
    (fq.ops.CY, (_QUBIT[0], _QUBIT[1])),
    (fq.ops.CS, (_QUBIT[0], _QUBIT[1])),
    (fq.ops.iSwap, (_QUBIT[0], _QUBIT[1])),
    (fq.ops.CCX, (_QUBIT[0], _QUBIT[1], _QUBIT[2])),
    (fq.ops.CSwap, (_QUBIT[0], _QUBIT[1], _QUBIT[2])),
    (fq.ops.RX(0.3), (_QUBIT[0],)),
    (fq.ops.RY(0.3), (_QUBIT[0],)),
    (fq.ops.RZ(0.3), (_QUBIT[0],)),
    (fq.ops.Phase(0.3), (_QUBIT[0],)),
    (fq.ops.CPhase(0.3), (_QUBIT[0], _QUBIT[1])),
    (fq.ops.Shift(1), (_QUTRIT[0],)),
    (fq.ops.Clock(1), (_QUTRIT[0],)),
    (fq.ops.Sum, (_QUTRIT[0], _QUTRIT[1])),
    (fq.ops.SwapLevels(0, 2), (_QUTRIT[0],)),
    (fq.ops.Fourier, (_QUTRIT[0],)),
    (fq.ops.InverseFourier, (_QUTRIT[0],)),
    (fq.ops.SubspaceRX(0.3, (0, 1)), (_QUTRIT[0],)),
    (fq.ops.SubspaceRY(0.3, (0, 1)), (_QUTRIT[0],)),
    (fq.ops.SubspaceRZ(0.3, (0, 1)), (_QUTRIT[0],)),
    (fq.ops.CClock(1), (_QUTRIT[0], _QUTRIT[1])),
]


def _content_code(matrix):
    """Classify through the single classification rule, `_classify_matrix`."""
    d = matrix.shape[0]
    columns = np.empty(d, dtype=np.int64)
    values = np.empty(d, dtype=np.complex128)
    contiguous = np.ascontiguousarray(matrix, dtype=np.complex128)
    return int(_classify_matrix(contiguous, columns, values))


def test_every_key_has_a_spec():
    assert set(_KERNEL_SPECS) == set(BuiltinKernelKey)


def test_kernel_specs_agree_with_content_classification():
    """The declared structure must match what a content scan finds.

    For generic parameters the two must agree exactly; a _DENSE declaration
    on a gate whose special parameter values are more structured is the one
    permitted (and safe) looseness, but at generic parameters a mismatch
    means either a wrong declaration or a missed specialization.
    """
    default_map = default_matrix_implementation_map()
    seen = set()
    for op, targets in _GATE_CASES:
        rule = default_map.implementation_for(type(op))
        key = rule._kernel_key(op, targets=targets)
        seen.add(key)
        matrix = np.asarray(rule(op, targets=targets), dtype=complex)
        assert _KERNEL_SPECS[key] == _content_code(matrix), type(op).__name__
    assert seen == set(BuiltinKernelKey)  # the case table covers every gate


def _counts(engine_cls, plan, dims, n_clbits, shots, seed, request):
    simulator = engine_cls()
    simulator.initialize(dims, n_clbits)
    raw = simulator.run(plan, shots, seed, request)
    return list(zip(raw.outcome_keys.tolist(), raw.outcome_counts.tolist()))


def _plan_and_request(program):
    backend = Simulator()
    plan, _ = backend._lower_program(program)
    return plan, backend._request_cls(counts=True, statevector=False)


def test_keyed_dispatch_matches_numpy_across_structure_classes():
    # Diagonal (RZ, CZ), permutation (X, CX, Swap), and dense (H, RX) gates
    # in one circuit; fast path, counts identical shot-for-shot.
    program = fq.Program(3, 3)
    program.add(fq.ops.H, 0)
    program.add(fq.ops.RZ(0.4), 0)
    program.add(fq.ops.CX, (0, 1))
    program.add(fq.ops.Swap, (1, 2))
    program.add(fq.ops.CZ, (0, 2))
    program.add(fq.ops.RX(1.1), 2)
    program.add(fq.ops.X, 1)
    program.measure((0, 1, 2), (0, 1, 2))
    plan, request = _plan_and_request(program)

    numpy_counts = _counts(NumpySVEngine, plan, (2, 2, 2), 3, 300, 11, request)
    numba_counts = _counts(NumbaSVEngine, plan, (2, 2, 2), 3, 300, 11, request)
    assert numpy_counts == numba_counts


def test_keyed_dispatch_matches_numpy_on_the_dynamic_path():
    program = fq.Program(2, 2)
    program.add(fq.ops.H, 0)
    program.measure(0, 0)
    program.add(fq.ops.X, 1, condition=(0, 1))
    program.add(fq.ops.RZ(0.7), 1)
    program.measure(1, 1)
    plan, request = _plan_and_request(program)

    numpy_counts = _counts(NumpySVEngine, plan, (2, 2), 2, 20, 13, request)
    numba_counts = _counts(NumbaSVEngine, plan, (2, 2), 2, 20, 13, request)
    assert numpy_counts == numba_counts


def test_unkeyed_step_content_scan_matches_keyed_dispatch():
    # The same X matrix keyed and un-keyed must produce identical states:
    # the key changes where identification happens, never the numbers.
    x_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
    keyed = ApplyMatrixStep(
        matrix=x_matrix, target_indices=(0,), kernel_key=BuiltinKernelKey.X
    )
    unkeyed = ApplyMatrixStep(matrix=x_matrix, target_indices=(0,))
    states = []
    for step in (keyed, unkeyed):
        simulator = NumbaSVEngine()
        simulator.initialize((2, 2), 0)
        simulator.apply(step)
        states.append(simulator.export_state())
    assert np.array_equal(states[0], states[1])


def test_dense_declaration_stays_correct_at_special_parameters():
    # RX is declared _DENSE; at theta=pi its matrix happens to be a
    # permutation. The declaration must stay numerically correct (the dense
    # kernel handles any matrix), just without the opportunistic speedup.
    program = fq.Program(1)
    program.add(fq.ops.RX(np.pi), 0)
    backend = Simulator()
    plan, _ = backend._lower_program(program)
    (step,) = plan
    assert _KERNEL_SPECS[step.kernel_key] == _DENSE

    simulator = NumbaSVEngine()
    simulator.initialize((2,), 0)
    simulator.apply(step)
    assert np.allclose(simulator.export_state(), [0.0, -1.0j])


def test_fallback_path_matches_the_standard_path():
    # apply(step) (standard, key-aware, cached) and _apply_local (matrix-only
    # fallback used by reset shifts and Kraus branches) must produce the same
    # state through the same resolved kernels.
    matrix = np.array([[0, 1], [1, 0]], dtype=complex)
    standard = NumbaSVEngine()
    standard.initialize((2, 2), 0)
    standard.apply(
        ApplyMatrixStep(
            matrix=matrix, target_indices=(1,), kernel_key=BuiltinKernelKey.X
        )
    )
    fallback = NumbaSVEngine()
    fallback.initialize((2, 2), 0)
    fallback._state = fallback._apply_local(fallback.state, matrix, (1,))

    assert np.array_equal(standard.export_state(), fallback.export_state())


def test_structure_cache_pins_steps_and_reuses_resolutions():
    simulator = NumbaSVEngine()
    simulator.initialize((2,), 0)
    step = ApplyMatrixStep(
        matrix=np.eye(2, dtype=complex),
        target_indices=(0,),
        kernel_key=BuiltinKernelKey.I,
    )
    first = simulator._resolve_structure(step)
    assert simulator._resolve_structure(step) is first  # cached, not re-resolved
    assert simulator._structure_cache[id(step)][0] is step  # pinned identity


def _reconstruct(csr, d):
    """Dense matrix from a ``(indptr, indices, data)`` CSR triple."""
    indptr, indices, data = csr
    matrix = np.zeros((d, d), dtype=complex)
    for r in range(d):
        for k in range(indptr[r], indptr[r + 1]):
            matrix[r, indices[k]] = data[k]
    return matrix


def test_superop_csr_skips_dense_and_round_trips_a_sparse_channel():
    # A fully dense matrix is left to the dense kernel (no CSR).
    assert _superop_csr(np.ones((4, 4), dtype=complex)) is None

    # A two-qubit depolarizing super-operator is far from full (~28 of 256
    # nonzero, plus a little ~1e-17 residue). Its CSR lists every nonzero in
    # column order, so the round-trip reproduces it bit for bit.
    qreg = fq.QuantumRegister(2)
    ops = depolarizing_rule(Depolarizing(p=0.2), targets=(qreg[0], qreg[1]))
    superop = _kraus_superop_kernel(_kraus_stack(ops))
    csr = _superop_csr(superop)
    assert csr is not None
    assert csr[1].size < superop.size // 2  # comfortably below the sparse floor
    assert np.array_equal(_reconstruct(csr, superop.shape[0]), superop)


def test_fuse_gate_channels_merges_an_adjacent_same_target_pair():
    # gate M then channel {K_i} on the same targets is the channel {K_i M}:
    # one super-operator pass instead of two, kraus_ops multiplied through.
    m = np.array([[0, 1], [1, 0]], dtype=complex)  # X
    k0 = np.sqrt(0.9) * np.eye(2, dtype=complex)
    k1 = np.sqrt(0.1) * np.array([[0, 1], [1, 0]], dtype=complex)
    gate = ApplyMatrixStep(matrix=m, target_indices=(0,))
    channel = ApplyChannelStep(kraus_ops=(k0, k1), target_indices=(0,))

    fused = _fuse_gate_channels([gate, channel])

    assert len(fused) == 1
    assert isinstance(fused[0], ApplyChannelStep)
    assert fused[0].target_indices == (0,)
    assert np.allclose(fused[0].kraus_ops[0], k0 @ m)
    assert np.allclose(fused[0].kraus_ops[1], k1 @ m)


def test_plan_chunks_floor_scales_with_thread_count():
    # The parallel floor is per-thread (`_MAX_THREADS * _GRAIN_TO_THREAD`), so a
    # state stays serial just below it and splits across threads at it - the
    # absolute crossover tracks the core count, not a fixed size.
    from fatqat.simulator._engine.nb import (
        _GRAIN_TO_THREAD,
        _MAX_THREADS,
        _MIN_SIZE_TO_THREAD,
        _plan_chunks,
    )

    assert _MIN_SIZE_TO_THREAD == _MAX_THREADS * _GRAIN_TO_THREAD
    cosets = _MIN_SIZE_TO_THREAD  # >> _MAX_THREADS, so the chunk count is thread-bound
    assert _plan_chunks(cosets, _MIN_SIZE_TO_THREAD - 1) == 1
    assert _plan_chunks(cosets, _MIN_SIZE_TO_THREAD) == min(_MAX_THREADS, cosets)


def test_coset_chunking_is_bit_identical_across_chunk_counts():
    # The floor only ever changes the chunk COUNT; disjoint cosets make the
    # result independent of it, which is why moving the floor never changes an
    # amplitude. A dense gate applied as one chunk equals it applied as many.
    from fatqat.simulator._engine.nb import (
        _apply_resolved_parallel,
        _apply_resolved_serial,
        _compute_apply_plan,
    )

    rng = np.random.default_rng(0)
    size = 1 << 6
    state = (rng.standard_normal(size) + 1j * rng.standard_normal(size)).astype(
        np.complex128
    )
    matrix = (rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))).astype(
        np.complex128
    )
    columns = np.empty(2, dtype=np.int64)
    values = np.empty(2, dtype=np.complex128)
    code = int(_classify_matrix(matrix, columns, values))
    offsets, comp_strides, comp_dims, num_cosets, _ = _compute_apply_plan(
        (2,) * 6, (0,)
    )

    args = (code, matrix, columns, values, offsets, comp_strides, comp_dims, num_cosets)
    serial = _apply_resolved_serial(state.copy(), *args)
    parallel = _apply_resolved_parallel(state.copy(), *args, 4)
    assert np.array_equal(serial, parallel)


def test_fuse_gate_channels_leaves_unfusable_pairs_untouched():
    # Only exact-adjacent, identically-targeted, unconditional pairs merge; a
    # differently-targeted or conditioned neighbor passes through by identity.
    m = np.eye(2, dtype=complex)
    kraus = (np.eye(2, dtype=complex),)
    gate = ApplyMatrixStep(matrix=m, target_indices=(0,))
    other_target = ApplyChannelStep(kraus_ops=kraus, target_indices=(1,))
    conditioned = ApplyChannelStep(
        kraus_ops=kraus, target_indices=(0,), condition=((0, 1),)
    )
    plan = [gate, other_target, gate, conditioned]

    fused = _fuse_gate_channels(plan)

    assert len(fused) == len(plan)
    assert all(f is p for f, p in zip(fused, plan))
