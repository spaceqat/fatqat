"""Compiled statevector trajectory sequencing with direct Numba numerical calls.

Each shot owns its quantum and classical state. The workflow interprets the
packed plan, applies conditions, records measurement reports, and advances the
random-draw cursor; numerical modules manipulate the quantum buffer.
"""

from collections.abc import Callable, Sequence
from math import prod
from typing import NamedTuple

import numpy as np
from numba import njit, prange

from ....._backends.steps import (
    ApplyChannelStep,
    ApplyMatrixStep,
    MeasurementStep,
    ResetStep,
    ResolvedStep,
)
from .....noise.base import _sampled_unitary_branches
from ._reporting import _report_digit_kernel
from ...._execution_contract import (
    _RunContext,
)
from ...numerics.numba_kernels._apply import _DENSE, _apply_resolved_range
from ...numerics.numba_kernels._channel import _apply_channel_step, _kraus_stack
from ...numerics.numba_kernels._measurement import (
    _sample_index_kernel,
    _sv_probabilities_kernel,
    _project_statevector_strided,
    _sv_reset_kernel,
)
from ...state import (
    EvolutionState,
    NumbaClassicalState,
    NumbaEvolutionState,
    _LoweredBasisState,
)
from ...numerics.numpy_kernels._state import _basis_index


class _NumbaTrajectoryExecution(NamedTuple):
    """Packed arrays and per-shot draw budget for the compiled SV workflow.

    This record stays on the host; the compiled entry receives the original
    positional arrays and scalar budget. Array references are retained as-is.
    """

    plan_arrays: tuple[np.ndarray, ...]
    max_draws: int


def _plan_compilable(plan: Sequence[ResolvedStep]) -> bool:
    """Whether the compiled multi-shot kernel understands every plan step.

    The workflow supports matrix, channel, measurement (including readout
    confusion), and reset steps. It does not yet encode non-identity
    physical-to-reported measurement maps. An unsupported
    step or map must not reach ``_compile_dynamic_plan``; the caller falls back
    to the Python per-shot workflow with the selected numerical runtime.
    """
    # pylint: disable-next=fixme
    # TODO: Compile non-identity ``reported_digit_maps`` into the measurement
    # table and apply them after physical collapse but before readout confusion.
    # Until then, keep routing such plans through the semantics-complete Python
    # per-shot workflow rather than silently reporting the physical digit.
    return all(
        isinstance(
            step, (ApplyMatrixStep, ApplyChannelStep, MeasurementStep, ResetStep)
        )
        and not (
            isinstance(step, MeasurementStep)
            and step.reported_digit_maps is not None
            and any(
                reported_map != tuple(range(len(reported_map)))
                for reported_map in step.reported_digit_maps
            )
        )
        for step in plan
    )


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
        assert (
            stack.shape[1] == offsets.shape[0]
        ), "Kraus dimension must equal the coset layout's local dimension"
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
        assert (
            len(confusions) == num_subsystems
        ), "confusions must align with the step's measured subsystems"
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


def _compile_dynamic_plan(
    plan: Sequence[ResolvedStep],
    dims: tuple[int, ...],
    build_apply_layout: Callable[[tuple[int, ...]], tuple],
    resolve_structure: Callable[[ApplyMatrixStep], tuple],
) -> _NumbaTrajectoryExecution:
    """Flatten a plan into typed arrays for `_run_sv_trajectories`.

    Uses the supplied numerical layout builder; measurement/reset carry their
    subsystem strides and dims for in-kernel collapse and reset-shift, while
    the two noise payloads are packed here: a channel hands its Kraus
    stack plus that same layout to `_compile_channel_table`, and each
    measurement hands its confusion tuple to `_compile_readout_table`, whose
    pointers are indexed by the measured-subsystem position built here.

    Returns a host record of positional plan arrays and the per-shot uniform-draw
    budget: one per measurement, one more per confusion-bearing measured
    subsystem, and one per reset and channel - the upper bound on RNG draws
    (a conditioned reset or channel that never fires simply draws less).
    """
    strides = [prod(dims[:q]) for q in range(len(dims))]

    step_kind: list[int] = []
    step_data: list[int] = []
    step_cond_ptr: list[int] = []
    step_cond_len: list[int] = []
    cond_clbit: list[int] = []
    cond_value: list[int] = []
    ap_mat_ptr, ap_dim, ap_off_ptr, ap_comp_ptr, ap_comp_len = [], [], [], [], []
    ap_code = []
    mat_flat, off_flat, comp_stride_flat, comp_dim_flat = [], [], [], []
    col_flat: list[int] = []
    val_flat: list[complex] = []
    me_ptr, me_len, me_classical, me_stride, me_dim = [], [], [], [], []
    rs_ptr, rs_len, rs_stride, rs_dim = [], [], [], []
    # Noise payloads, flattened in one pass below by the noise package:
    # (kraus_ops, offsets, comp_strides, comp_dims) per channel occurrence,
    # and (num_subsystems, confusions) per measurement - the latter in
    # measurement order, which is the order me_* is filled in.
    channel_entries: list[tuple] = []
    readout_entries: list[tuple] = []
    num_measurements = 0
    num_resets = 0

    def add_condition(condition):
        step_cond_ptr.append(len(cond_clbit))
        step_cond_len.append(0 if condition is None else len(condition))
        for clbit, value in condition or ():
            cond_clbit.append(clbit)
            cond_value.append(value)

    for step in plan:
        if isinstance(step, ApplyMatrixStep):
            offsets, comp_strides, comp_dims, _, _ = build_apply_layout(
                step.target_indices
            )
            step_kind.append(0)
            step_data.append(len(ap_dim))
            add_condition(step.condition)
            ap_dim.append(offsets.shape[0])
            ap_mat_ptr.append(len(mat_flat))
            matrix = np.ascontiguousarray(step.matrix, dtype=np.complex128)
            mat_flat.extend(matrix.ravel().tolist())
            # Structure resolved once at compile time (declared key or a
            # single content scan) instead of per gate per shot in-kernel.
            # col/val ride the same per-gate pointer as the offsets; a
            # dense gate stores zero padding the kernel never reads.
            code, columns, values = resolve_structure(step)
            ap_code.append(code)
            local_dim = offsets.shape[0]
            if code == _DENSE:
                col_flat.extend([0] * local_dim)
                val_flat.extend([0j] * local_dim)
            else:
                col_flat.extend(int(c) for c in columns)
                val_flat.extend(complex(v) for v in values)
            ap_off_ptr.append(len(off_flat))
            off_flat.extend(int(o) for o in offsets)
            ap_comp_ptr.append(len(comp_stride_flat))
            ap_comp_len.append(comp_strides.shape[0])
            comp_stride_flat.extend(int(s) for s in comp_strides)
            comp_dim_flat.extend(int(d) for d in comp_dims)
        elif isinstance(step, MeasurementStep):
            step_kind.append(1)
            step_data.append(len(me_len))
            add_condition(None)
            me_ptr.append(len(me_stride))
            me_len.append(len(step.measured_indices))
            for q in step.measured_indices:
                me_stride.append(strides[q])
                me_dim.append(dims[q])
            me_classical.extend(step.classical_indices)
            readout_entries.append((len(step.measured_indices), step.confusions))
            num_measurements += 1
        elif isinstance(step, ApplyChannelStep):
            offsets, comp_strides, comp_dims, _, _ = build_apply_layout(
                step.target_indices
            )
            step_kind.append(3)
            step_data.append(len(channel_entries))
            add_condition(step.condition)
            channel_entries.append((step.kraus_ops, offsets, comp_strides, comp_dims))
        else:  # ResetStep
            step_kind.append(2)
            step_data.append(len(rs_len))
            add_condition(step.condition)
            rs_ptr.append(len(rs_stride))
            rs_len.append(len(step.reset_indices))
            for q in step.reset_indices:
                rs_stride.append(strides[q])
                rs_dim.append(dims[q])
            num_resets += 1

    def i64(values):
        return np.asarray(values, dtype=np.int64)

    me_conf_ptr, conf_flat = _compile_readout_table(readout_entries)
    plan_arrays = (
        i64(step_kind),
        i64(step_data),
        i64(step_cond_ptr),
        i64(step_cond_len),
        i64(cond_clbit),
        i64(cond_value),
        i64(ap_mat_ptr),
        i64(ap_dim),
        i64(ap_off_ptr),
        i64(ap_comp_ptr),
        i64(ap_comp_len),
        i64(ap_code),
        np.asarray(mat_flat, dtype=np.complex128),
        i64(off_flat),
        i64(col_flat),
        np.asarray(val_flat, dtype=np.complex128),
        i64(comp_stride_flat),
        i64(comp_dim_flat),
        i64(me_ptr),
        i64(me_len),
        i64(me_classical),
        i64(me_stride),
        i64(me_dim),
        me_conf_ptr,
        conf_flat,
        i64(rs_ptr),
        i64(rs_len),
        i64(rs_stride),
        i64(rs_dim),
        *_compile_channel_table(channel_entries),
    )
    num_confusions = int(np.count_nonzero(me_conf_ptr >= 0))
    max_draws = num_measurements + num_confusions + num_resets + len(channel_entries)
    return _NumbaTrajectoryExecution(plan_arrays, max_draws)


# Positions in `_compile_dynamic_plan`'s ``plan_arrays`` that hold gate
# matrices and their resolved structure; col/val share the offsets pointer.
_AP_MAT_PTR, _AP_DIM, _AP_OFF_PTR, _AP_CODE = 6, 7, 8, 11
_MAT_FLAT, _COL_FLAT, _VAL_FLAT = 12, 14, 15


def _patch_gate_matrices(
    packed: _NumbaTrajectoryExecution,
    plan: Sequence[ResolvedStep],
    positions: Sequence[int],
    resolve_structure: Callable[[ApplyMatrixStep], tuple],
) -> _NumbaTrajectoryExecution:
    """Repack only the gate matrices at ``positions`` of a structurally equal plan.

    ``packed`` came from `_compile_dynamic_plan` for a plan with the same step
    kinds, targets, and conditions as ``plan``; only the gates at
    ``positions`` may hold different matrices. Their matrix entries and
    structure codes are written into copies of the affected arrays, so
    ``packed`` itself is never modified and every other array is shared.
    """
    arrays = list(packed.plan_arrays)
    mat_ptr, dims, off_ptr = (arrays[i] for i in (_AP_MAT_PTR, _AP_DIM, _AP_OFF_PTR))
    codes, matrices, columns, values = (
        arrays[i].copy() for i in (_AP_CODE, _MAT_FLAT, _COL_FLAT, _VAL_FLAT)
    )
    targets = set(positions)
    gate = 0
    for position, step in enumerate(plan):
        if not isinstance(step, ApplyMatrixStep):
            continue
        if position in targets:
            d = dims[gate]
            start = mat_ptr[gate]
            matrices[start : start + d * d] = np.ascontiguousarray(
                step.matrix, dtype=np.complex128
            ).ravel()
            code, step_columns, step_values = resolve_structure(step)
            codes[gate] = code
            start = off_ptr[gate]
            if code == _DENSE:
                columns[start : start + d] = 0
                values[start : start + d] = 0
            else:
                columns[start : start + d] = step_columns
                values[start : start + d] = step_values
        gate += 1
    for index, array in zip(
        (_AP_CODE, _MAT_FLAT, _COL_FLAT, _VAL_FLAT), (codes, matrices, columns, values)
    ):
        arrays[index] = array
    return packed._replace(plan_arrays=tuple(arrays))


def _execute_sv_trajectories(
    context: _RunContext,
    compiled: _NumbaTrajectoryExecution,
    *,
    initial_state: EvolutionState[np.ndarray],
) -> np.ndarray:
    """Return classical shot rows from one compiled multi-shot call."""
    max_draws = compiled.max_draws
    shots = context.shots
    uniforms = np.empty(shots * max_draws, dtype=np.float64)
    for s, seed_sequence in enumerate(
        np.random.SeedSequence(context.seed).spawn(shots)
    ):
        uniforms[s * max_draws : (s + 1) * max_draws] = np.random.default_rng(
            seed_sequence
        ).random(max_draws)

    state_size = prod(context.system_dims) if context.system_dims else 1
    start, custom_start, start_index = _start_template(
        initial_state, context.system_dims
    )
    clbits = initial_state.classical.clbits
    clbits_start = np.asarray(clbits if clbits is not None else (), dtype=np.int64)
    return _run_sv_trajectories(
        *compiled.plan_arrays,
        start,
        state_size,
        custom_start,
        start_index,
        clbits_start,
        context.n_clbits,
        shots,
        uniforms,
        max_draws,
    )


@njit(cache=True)
def _condition_matches(
    clbits, cond_clbit, cond_value, start, length
) -> bool:  # pragma: no cover - compiled by Numba
    """Whether the lowered feedforward condition ``clbits[c] == v`` all hold."""
    for t in range(length):
        if clbits[cond_clbit[start + t]] != cond_value[start + t]:
            return False
    return True


@njit(cache=True)
def _apply_matrix_step(
    state,
    a,
    ap_mat_ptr,
    ap_dim,
    ap_off_ptr,
    ap_comp_ptr,
    ap_comp_len,
    ap_code,
    mat_flat,
    off_flat,
    col_flat,
    val_flat,
    comp_stride_flat,
    comp_dim_flat,
    size,
) -> None:  # pragma: no cover - compiled by Numba
    """Apply compiled gate ``a`` to ``state`` in place (serial coset walk).

    Structure (``ap_code``/columns/values) was resolved when the plan was
    compiled - by declared kernel key or one content scan - so no per-shot
    classification happens here.
    """
    d = ap_dim[a]
    mat_start = ap_mat_ptr[a]
    matrix = mat_flat[mat_start : mat_start + d * d].reshape(d, d)
    off_start = ap_off_ptr[a]
    offsets = off_flat[off_start : off_start + d]
    comp_start = ap_comp_ptr[a]
    comp_end = comp_start + ap_comp_len[a]
    columns = col_flat[off_start : off_start + d]
    values = val_flat[off_start : off_start + d]
    code = ap_code[a]
    _apply_resolved_range(
        state,
        code,
        matrix,
        columns,
        values,
        offsets,
        comp_stride_flat[comp_start:comp_end],
        comp_dim_flat[comp_start:comp_end],
        0,
        size // d,
    )


@njit(cache=True)
def _measure_step(
    state,
    clbits,
    m,
    me_ptr,
    me_len,
    me_classical,
    me_stride,
    me_dim,
    me_conf_ptr,
    conf_flat,
    uniforms,
    draw,
):  # pragma: no cover - compiled by Numba
    """Collapse the measured subsystems, write their reported digits to ``clbits``.

    Takes the shot's uniform pool and its draw cursor rather than a single
    uniform, because the number of draws depends on the noise attached to this
    measurement: one for the Born-sampled collapse, then one more for each
    measured subsystem carrying a readout confusion. The collapse always keeps
    the *true* outcome and only the reported classical value is resampled
    (`_report_digit_kernel`), so state export and qubit reuse are untouched
    while feedforward reads the report - the same semantics, and the same draw
    order, as ``_run_one_shot``'s ``_report_digit`` on the NumPy path.

    Returns the projected state and the advanced cursor.
    """
    start = me_ptr[m]
    length = me_len[m]
    index = _sample_index_kernel(_sv_probabilities_kernel(state), uniforms[draw])
    draw += 1
    state = _project_statevector_strided(
        state, me_stride[start : start + length], me_dim[start : start + length], index
    )
    for j in range(length):
        digit = (index // me_stride[start + j]) % me_dim[start + j]
        conf_ptr = me_conf_ptr[start + j]
        if conf_ptr >= 0:
            digit = _report_digit_kernel(
                conf_flat, conf_ptr, me_dim[start + j], digit, uniforms[draw]
            )
            draw += 1
        clbits[me_classical[start + j]] = digit
    return state, draw


@njit(cache=True)
def _reset_step(
    state, r, rs_ptr, rs_len, rs_stride, rs_dim, u
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Collapse the reset subsystems, then shift each measured branch to |0>."""
    start = rs_ptr[r]
    length = rs_len[r]
    return _sv_reset_kernel(
        state, rs_stride[start : start + length], rs_dim[start : start + length], u
    )


def _start_template(
    initial_state: EvolutionState, system_dims: Sequence[int]
) -> tuple[np.ndarray, bool, int]:
    """Return the compiled start arguments: template, whether to copy it, index."""
    quantum = initial_state.quantum
    if isinstance(quantum, _LoweredBasisState):
        # Basis states, the ground state included, pass no O(D) template.
        index = _basis_index(quantum.levels, system_dims)
        return np.empty(0, dtype=np.complex128), False, index
    assert quantum is not None, "Simulator lowers the ground state"
    state_size = prod(system_dims) if system_dims else 1
    # The custom template is read-only: each shot takes the private copy it
    # evolves. `ascontiguousarray` therefore aliases the validated common
    # case and copies only when layout or dtype requires it.
    template = np.ascontiguousarray(quantum, dtype=np.complex128).reshape(state_size)
    return template, True, 0


@njit(cache=True, inline="always")
def _initialize_shot_state(
    start, state_size, custom_start, start_index, n_clbits, clbits_start
):
    """Allocate one shot's quantum and classical buffers without sharing them."""
    if custom_start:
        # A supplied template must survive unchanged to seed every trajectory.
        quantum = start.copy()
    else:
        # Construct the basis state directly, without an O(D) template to copy.
        quantum = np.zeros(state_size, dtype=np.complex128)
        quantum[start_index] = 1.0 + 0.0j
    clbits = (
        clbits_start.copy() if clbits_start.size else np.zeros(n_clbits, dtype=np.int64)
    )
    return NumbaEvolutionState(quantum, NumbaClassicalState(clbits))


@njit(cache=True, parallel=True)
def _run_sv_trajectories(
    # per-step sequencer (one entry per plan step, in program order)
    step_kind,  # 0=gate, 1=measurement, 2=reset
    step_data,  # row index into this kind's table (gate/measure/reset)
    step_cond_ptr,  # start of this step's condition in the condition pool
    step_cond_len,  # number of (clbit, value) terms in the condition
    # condition pool (flat, shared by all conditioned steps)
    cond_clbit,  # clbit each feedforward term reads
    cond_value,  # value that clbit must equal for the step to fire
    # gate table (one entry per ApplyMatrixStep)
    ap_mat_ptr,  # start of the d*d matrix in mat_flat
    ap_dim,  # local dimension d (matrix is d*d, with d offsets)
    ap_off_ptr,  # start of the d local->flat offsets in off_flat
    ap_comp_ptr,  # start of the complement strides/dims in comp_*_flat
    ap_comp_len,  # number of complement (non-target) subsystems
    ap_code,  # structure code per gate, resolved at compile time
    # gate flat backing
    mat_flat,  # concatenated row-major gate matrices (complex128)
    off_flat,  # concatenated local-index -> flat-offset tables
    col_flat,  # per-row nonzero columns (aligned with off_flat; 0-pad if dense)
    val_flat,  # per-row nonzero values (aligned with off_flat; 0-pad if dense)
    comp_stride_flat,  # concatenated complement strides
    comp_dim_flat,  # concatenated complement dimensions
    # measurement table (one entry per MeasurementStep)
    me_ptr,  # start of this measurement's subsystems in me_*
    me_len,  # number of measured subsystems
    # measurement flat backing
    me_classical,  # clbit each measured subsystem's digit is written to
    me_stride,  # flat stride of each measured subsystem
    me_dim,  # dimension of each measured subsystem (also each confusion's side)
    me_conf_ptr,  # start of this subsystem's confusion in conf_flat, -1 if none
    conf_flat,  # concatenated row-major confusion matrices (float64)
    # reset table (one entry per ResetStep)
    rs_ptr,  # start of this reset's subsystems in rs_*
    rs_len,  # number of reset subsystems
    # reset flat backing
    rs_stride,  # flat stride of each reset subsystem
    rs_dim,  # dimension of each reset subsystem
    # channel table (one entry per ApplyChannelStep), built by `_compile_channel_table`
    ch_kra_ptr,  # start of this channel's Kraus stack in ch_kra_flat
    ch_num_kraus,  # number of Kraus operators in the stack
    ch_dim,  # local dimension d (each operator is d*d, with d offsets)
    ch_off_ptr,  # start of the d local->flat offsets in ch_off_flat
    ch_comp_ptr,  # start of the complement strides/dims in ch_comp_*_flat
    ch_comp_len,  # number of complement (non-target) subsystems
    ch_cdf_ptr,  # start of the fixed branch cdf, -1 if state-dependent
    ch_mmat_diag,  # 1 where every K^dagger K is diagonal: weigh from the marginal
    # channel flat backing (its own pools: a channel carries a stack, not one matrix)
    ch_kra_flat,  # concatenated row-major Kraus operators (complex128)
    ch_mmat_flat,  # concatenated K_i^dagger K_i per Kraus (complex128), for branch weights
    ch_off_flat,  # concatenated local-index -> flat-offset tables
    ch_comp_stride_flat,  # concatenated complement strides
    ch_comp_dim_flat,  # concatenated complement dimensions
    ch_cdf_flat,  # concatenated scaled-unitary branch cdfs, normalized (float64)
    ch_ident_flat,  # 1 where that branch is the identity, so the draw is a no-op
    start,  # custom-state template; empty when a basis state is requested
    state_size,  # flat state width, kept explicit when `start` is empty
    custom_start,  # whether each shot must copy `start`, not build a basis state
    start_index,  # basis state each shot builds when `custom_start` is false
    clbits_start,  # initial report digits; empty for zero initialization
    n_clbits,  # classical-register width: per-shot clbits and result columns
    shots,  # number of independent trajectories - the `prange` extent
    uniforms,  # pre-drawn uniforms, shots*max_draws in execution order
    max_draws,  # per-shot uniform budget; shot s reads uniforms[s*max_draws:]
) -> np.ndarray:  # pragma: no cover - compiled by Numba
    """Run ``shots`` independent dynamic trajectories in parallel.

    Each shot (a `prange` iteration) owns a local `NumbaEvolutionState`
    and interprets the compiled plan: conditioned gate application, projective
    measurement with readout confusion, conditioned reset, and conditioned channel
    noise. Uniforms are pre-drawn per shot in execution order (slice
    ``uniforms[s*max_draws:]``), consumed one per measurement, one more per
    confusion-bearing measured subsystem, and one per firing reset or channel -
    matching the serial path's RNG stream, so counts are identical. Shots are
    independent, so the result is deterministic regardless of thread scheduling.
    """
    num_steps = step_kind.shape[0]
    results = np.zeros((shots, n_clbits), dtype=np.int64)
    for shot in prange(shots):  # pylint: disable=not-an-iterable
        evolution = _initialize_shot_state(
            start, state_size, custom_start, start_index, n_clbits, clbits_start
        )
        draw = shot * max_draws

        for st in range(num_steps):
            kind = step_kind[st]
            passes = _condition_matches(
                evolution.classical.clbits,
                cond_clbit,
                cond_value,
                step_cond_ptr[st],
                step_cond_len[st],
            )
            if kind == 1:  # measurement (unconditional)
                quantum, draw = _measure_step(
                    evolution.quantum,
                    evolution.classical.clbits,
                    step_data[st],
                    me_ptr,
                    me_len,
                    me_classical,
                    me_stride,
                    me_dim,
                    me_conf_ptr,
                    conf_flat,
                    uniforms,
                    draw,
                )
                evolution = NumbaEvolutionState(quantum, evolution.classical)
            elif kind == 0 and passes:  # gate
                _apply_matrix_step(
                    evolution.quantum,
                    step_data[st],
                    ap_mat_ptr,
                    ap_dim,
                    ap_off_ptr,
                    ap_comp_ptr,
                    ap_comp_len,
                    ap_code,
                    mat_flat,
                    off_flat,
                    col_flat,
                    val_flat,
                    comp_stride_flat,
                    comp_dim_flat,
                    state_size,
                )
            elif kind == 2 and passes:  # reset
                quantum = _reset_step(
                    evolution.quantum,
                    step_data[st],
                    rs_ptr,
                    rs_len,
                    rs_stride,
                    rs_dim,
                    uniforms[draw],
                )
                evolution = NumbaEvolutionState(quantum, evolution.classical)
                draw += 1
            elif kind == 3 and passes:  # channel noise (applied in place)
                _apply_channel_step(
                    evolution.quantum,
                    step_data[st],
                    ch_kra_ptr,
                    ch_num_kraus,
                    ch_dim,
                    ch_off_ptr,
                    ch_comp_ptr,
                    ch_comp_len,
                    ch_cdf_ptr,
                    ch_mmat_diag,
                    ch_kra_flat,
                    ch_mmat_flat,
                    ch_off_flat,
                    ch_comp_stride_flat,
                    ch_comp_dim_flat,
                    ch_cdf_flat,
                    ch_ident_flat,
                    state_size,
                    uniforms[draw],
                )
                draw += 1

        for c in range(n_clbits):
            results[shot, c] = evolution.classical.clbits[c]
    return results
