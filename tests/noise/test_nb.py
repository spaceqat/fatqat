"""Unit tests for the Numba channel kernels in ``fatqat.noise.nb``.

White-box, kernel-level: each test pins one kernel against the NumPy
expression it replaces. End-to-end channel execution through the numba
runtime lives in tests/backend/test_noise_execution.py.
"""

import numpy as np
import pytest

pytest.importorskip("numba")

# pylint: disable=wrong-import-position
import fatqat as fq
from fatqat.noise import AmplitudeDamping, Depolarizing, PhaseDamping
from fatqat.noise.catalog import (
    amplitude_damping_rule,
    depolarizing_rule,
    phase_damping_rule,
)
from fatqat.noise.nb import (
    _compile_channel_table,
    _compile_readout_table,
    _jump_branch_kernel,
    _kraus_stack,
    _kraus_superop_kernel,
    _report_digit_kernel,
)


def _kraus_for(channel, rule, dim):
    """Resolve a catalog channel at ``dim`` through its own rule."""
    return rule(channel, targets=(fq.QuantumRegister(1, dim=dim)[0],))


def _branch_stack(weights):
    """Branches whose squared norms are ``weights`` and that stay distinguishable.

    Branch ``i`` populates only amplitude ``i``, so the kernel's normalized
    output identifies which branch it picked.
    """
    branches = np.zeros((len(weights), len(weights)), dtype=np.complex128)
    for i, weight in enumerate(weights):
        branches[i, i] = np.sqrt(weight)
    return branches


# --- _kraus_stack / _kraus_superop_kernel ---


def test_kraus_stack_copies_and_normalizes_the_payload():
    kraus_ops = _kraus_for(AmplitudeDamping(p=(0.3,)), amplitude_damping_rule, 2)
    stack = _kraus_stack(kraus_ops)

    assert stack.shape == (2, 2, 2)
    assert stack.dtype == np.complex128
    assert stack.flags["C_CONTIGUOUS"]
    # The step's operators are frozen read-only; the stack must be a writable
    # copy the kernels can own, never a view onto them.
    assert stack.flags.writeable
    assert not np.shares_memory(stack, kraus_ops[0])
    assert np.array_equal(stack[0], kraus_ops[0])


@pytest.mark.parametrize(
    "channel, rule, dim",
    [
        (Depolarizing(p=0.3), depolarizing_rule, 2),
        (Depolarizing(p=0.15), depolarizing_rule, 3),
        (AmplitudeDamping(p=(0.2, 0.4)), amplitude_damping_rule, 3),
        (PhaseDamping(p=0.25), phase_damping_rule, 3),
    ],
)
def test_kraus_superop_kernel_equals_the_numpy_kronecker_sum(channel, rule, dim):
    # The kernel replaces `sum(np.kron(k, k.conj()) for k in kraus_ops)`; it
    # accumulates in the same Kraus order, so only last-bit round-off (Numba
    # may fuse the multiply-accumulate) separates the two.
    kraus_ops = _kraus_for(channel, rule, dim)
    expected = sum(np.kron(k, np.conjugate(k)) for k in kraus_ops)

    superop = _kraus_superop_kernel(_kraus_stack(kraus_ops))

    assert superop.shape == (dim**2, dim**2)
    assert np.allclose(superop, expected, rtol=0.0, atol=1e-15)


def test_kraus_superop_kernel_reproduces_the_sandwich_on_vectorized_rho():
    # The super-operator's defining property: applied to vec(rho) with the ket
    # group most-significant it equals sum_i K_i rho K_i^dagger.
    kraus_ops = _kraus_for(AmplitudeDamping(p=(0.3, 0.1)), amplitude_damping_rule, 3)
    rng = np.random.default_rng(0)
    ket = rng.normal(size=3) + 1j * rng.normal(size=3)
    rho = np.outer(ket, ket.conj())

    superop = _kraus_superop_kernel(_kraus_stack(kraus_ops))
    applied = (superop @ rho.reshape(-1)).reshape(3, 3)
    expected = sum(k @ rho @ k.conj().T for k in kraus_ops)

    assert np.allclose(applied, expected)


# --- _jump_branch_kernel ---


def test_jump_branch_kernel_returns_the_chosen_branch_normalized():
    branches = _branch_stack([0.25, 0.75])

    out = _jump_branch_kernel(branches, 0.9)  # in the second branch's interval

    assert np.isclose(np.linalg.norm(out), 1.0)
    assert np.allclose(out, [0.0, 1.0])


def test_jump_branch_kernel_selects_by_squared_norm_at_interval_edges():
    # Four equal branches of amplitude 0.5: the weights the kernel recovers
    # (0.5 * 0.5) and the cdf edges (0.25 / 0.5 / 0.75 / 1.0) are all exact in
    # binary, so the boundary behavior is not round-off dependent.
    branches = _branch_stack([0.25] * 4)

    for u, expected in [
        (0.0, 0),
        (0.25 - 1e-12, 0),
        (0.25, 1),  # right-side search: a u equal to the edge falls right
        (0.5, 2),
        (0.75, 3),
        (0.999, 3),
    ]:
        out = _jump_branch_kernel(branches, float(u))
        assert int(np.argmax(np.abs(out))) == expected, u


def test_jump_branch_kernel_consumes_the_rng_stream_like_rng_choice():
    # The reproducibility contract: one `rng.random()` fed through the kernel's
    # inverse CDF must land on the same branch `rng.choice(num, p=...)` would,
    # draw for draw.
    weights = np.array([0.2, 0.3, 0.5])
    branches = _branch_stack(weights)
    probabilities = weights / weights.sum()

    kernel_rng = np.random.default_rng(1234)
    choice_rng = np.random.default_rng(1234)
    for _ in range(50):
        out = _jump_branch_kernel(branches, float(kernel_rng.random()))
        assert int(np.argmax(np.abs(out))) == int(
            choice_rng.choice(len(weights), p=probabilities)
        )


def test_jump_branch_kernel_never_picks_a_zero_weight_branch():
    # A zero-probability Kraus branch has a zero-width cdf interval, so the
    # right-side search steps over it - the same way rng.choice does.
    branches = _branch_stack([0.5, 0.0, 0.5])

    for u in np.linspace(0.0, 0.999, 25):
        out = _jump_branch_kernel(branches, float(u))
        assert int(np.argmax(np.abs(out))) in (0, 2)


# --- _report_digit_kernel ---


def test_report_digit_kernel_reads_the_column_of_the_true_digit():
    # Column j is P(report | true j): column 0 always reports 1, column 1
    # always reports 0.
    flip = np.array([[0.0, 1.0], [1.0, 0.0]])
    conf_flat = flip.ravel()

    for true_digit, expected in [(0, 1), (1, 0)]:
        for u in (0.0, 0.5, 0.999):
            assert (
                _report_digit_kernel(conf_flat, 0, 2, true_digit, float(u)) == expected
            )


def test_report_digit_kernel_consumes_the_rng_stream_like_report_digit():
    # `_report_digit` on the NumPy path is rng.choice(dim, p=confusion[:, true]);
    # the kernel must land on the same digit for the same uniform, draw for draw,
    # or the fused kernel's counts diverge from the serial path's.
    confusion = np.array([[0.7, 0.1, 0.2], [0.2, 0.6, 0.1], [0.1, 0.3, 0.7]])
    conf_flat = np.ascontiguousarray(confusion).ravel()

    kernel_rng = np.random.default_rng(99)
    choice_rng = np.random.default_rng(99)
    for step in range(60):
        true_digit = step % 3
        assert _report_digit_kernel(
            conf_flat, 0, 3, true_digit, float(kernel_rng.random())
        ) == int(choice_rng.choice(3, p=confusion[:, true_digit]))


def test_report_digit_kernel_reads_its_matrix_at_the_given_offset():
    # Confusions live in one shared pool, so the kernel must honor ``ptr``.
    identity = np.eye(2)
    flip = np.array([[0.0, 1.0], [1.0, 0.0]])
    conf_flat = np.concatenate([identity.ravel(), flip.ravel()])

    assert _report_digit_kernel(conf_flat, 0, 2, 1, 0.5) == 1  # identity block
    assert _report_digit_kernel(conf_flat, 4, 2, 1, 0.5) == 0  # flip block


# --- _compile_readout_table ---


def test_compile_readout_table_marks_error_free_subsystems_with_minus_one():
    conf_ptr, conf_flat = _compile_readout_table([(2, None), (1, None)])

    assert conf_ptr.dtype == np.int64
    assert conf_ptr.tolist() == [-1, -1, -1]
    assert conf_flat.dtype == np.float64
    assert conf_flat.size == 0


def test_compile_readout_table_pools_only_the_attached_matrices():
    flip = np.array([[0.0, 1.0], [1.0, 0.0]])
    qutrit = np.eye(3)
    entries = [
        (2, (None, flip)),  # second subsystem of a paired measurement
        (1, None),  # error-free measurement
        (1, (qutrit,)),  # qutrit readout
    ]

    conf_ptr, conf_flat = _compile_readout_table(entries)

    # One pointer per measured subsystem, in measurement order.
    assert conf_ptr.tolist() == [-1, 0, -1, 4]
    assert conf_flat.size == 4 + 9
    assert np.array_equal(conf_flat[:4].reshape(2, 2), flip)
    assert np.array_equal(conf_flat[4:].reshape(3, 3), qutrit)


def test_compile_readout_table_rejects_confusions_that_do_not_align():
    # The tuple length and the step's measured-subsystem count are the same
    # fact arriving from two places; a mismatch is a caller bug.
    flip = np.array([[0.0, 1.0], [1.0, 0.0]])

    with pytest.raises(AssertionError):
        _compile_readout_table([(2, (flip,))])


# --- _compile_channel_table ---


def test_compile_channel_table_is_empty_but_typed_without_channels():
    # A channel-free plan still hands the kernel arrays; only their dtypes and
    # emptiness matter for Numba's typing.
    kra_ptr, num_kraus, local_dim, off_ptr, comp_ptr, comp_len, *pools = (
        _compile_channel_table([])
    )

    for array in (kra_ptr, num_kraus, local_dim, off_ptr, comp_ptr, comp_len):
        assert array.dtype == np.int64
        assert array.size == 0
    kra_flat, mmat_flat, off_flat, comp_stride_flat, comp_dim_flat = pools
    for array in (kra_flat, mmat_flat):
        assert array.dtype == np.complex128
        assert array.size == 0
    for array in (off_flat, comp_stride_flat, comp_dim_flat):
        assert array.dtype == np.int64
        assert array.size == 0


def test_compile_channel_table_lays_out_two_channels_back_to_back():
    qubit_ops = _kraus_for(Depolarizing(p=0.2), depolarizing_rule, 2)  # 4 of 2x2
    qutrit_ops = _kraus_for(PhaseDamping(p=0.2), phase_damping_rule, 3)  # 3 of 3x3
    entries = [
        (qubit_ops, np.array([0, 1]), np.array([2]), np.array([3])),
        (qutrit_ops, np.array([0, 2, 4]), np.array([1]), np.array([2])),
    ]

    (
        kra_ptr,
        num_kraus,
        local_dim,
        off_ptr,
        comp_ptr,
        comp_len,
        kra_flat,
        mmat_flat,
        off_flat,
        comp_stride_flat,
        comp_dim_flat,
    ) = _compile_channel_table(entries)

    assert num_kraus.tolist() == [4, 3]
    assert local_dim.tolist() == [2, 3]
    # Pointers are running offsets into the flat pools: 4 * 2 * 2 = 16 complex
    # entries for the first channel, then 3 * 3 * 3 = 27 for the second.
    assert kra_ptr.tolist() == [0, 16]
    assert kra_flat.size == 16 + 27
    assert off_ptr.tolist() == [0, 2]
    assert off_flat.tolist() == [0, 1, 0, 2, 4]
    assert comp_ptr.tolist() == [0, 1]
    assert comp_len.tolist() == [1, 1]
    assert comp_stride_flat.tolist() == [2, 1]
    assert comp_dim_flat.tolist() == [3, 2]
    # Row-major per operator, operators in Kraus order.
    assert np.array_equal(kra_flat[:4].reshape(2, 2), qubit_ops[0])
    assert np.array_equal(kra_flat[16:25].reshape(3, 3), qutrit_ops[0])
    # M_i = K_i^dagger K_i shares the Kraus pool's layout and its per-channel
    # pointer, one d x d Hermitian block per operator, in Kraus order.
    assert mmat_flat.size == 16 + 27
    k0 = qubit_ops[0]
    assert np.allclose(mmat_flat[:4].reshape(2, 2), k0.conj().T @ k0)


def test_compile_channel_table_rejects_a_layout_of_the_wrong_local_dimension():
    # The Kraus dimension and the coset layout's local dimension are the same
    # fact arriving from two places; a mismatch is a caller bug, not input.
    qubit_ops = _kraus_for(Depolarizing(p=0.2), depolarizing_rule, 2)
    entries = [(qubit_ops, np.array([0, 1, 2, 3]), np.array([]), np.array([]))]

    with pytest.raises(AssertionError):
        _compile_channel_table(entries)
