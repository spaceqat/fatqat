"""Step-level gate fusion: limits, block scheduling, and fused steps."""

import subprocess
import sys
from dataclasses import replace
from math import prod

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat._backends import steps
from fatqat._backends.steps import ApplyMatrixStep, ApplyParametricMatrixStep
from fatqat._backends.steps import BuiltinKernelKey as Key
from fatqat.simulator import Simulator
from fatqat.simulator._engine.binding import _bind_parameter_batch
from fatqat.simulator._engine.numerics._gate_matrices import (
    _DENSE,
    _DIAGONAL,
    _KERNEL_SPECS,
    _PERMUTATION,
)
from fatqat.simulator._engine.optimization.fusion import materialize, schedule
from fatqat.simulator._execution_contract import (
    _GateFusionLimits,
    _KernelCapabilities,
)

# Merging is forced at every size so that each test isolates one rule.
FORCED = _GateFusionLimits(min_size=0, max_dim=8, structure_aware=False)
FORCED_AWARE = replace(FORCED, structure_aware=True)


def _gate(targets, key=None, dims=None, condition=None):
    """An identity matrix step; scheduling reads everything but the matrix."""
    size = prod((dims or {}).get(target, 2) for target in targets)
    return ApplyMatrixStep(
        matrix=np.eye(size, dtype=complex),
        target_indices=tuple(targets),
        condition=condition,
        kernel_key=key,
    )


def _assert_partition(blocks, plan):
    # A range may carry a step, so only these two checks reject blocks that
    # skip, repeat, or reorder steps.
    assert [index for block in blocks for index in block] == list(range(len(plan)))
    assert all(block.step == 1 for block in blocks)


def _schedule(plan, dims, limits):
    blocks = schedule(plan, dims, limits)
    _assert_partition(blocks, plan)
    return blocks


def test_kernel_capabilities_declare_no_gate_fusion_by_default():
    # Engines opt in to gate fusion one at a time, so a capability record
    # that does not mention it must leave plans unfused.
    capabilities = _KernelCapabilities(
        supports_kernel_threads=False, thread_capacity=1, supports_fusion=False
    )

    assert capabilities.gate_fusion is None


def test_gate_fusion_limits_are_hashable_values():
    # Limits live inside the hashable engine capabilities, so equal limits
    # must compare and hash equal.
    first = _GateFusionLimits(
        min_size=4, max_dim=8, structure_aware=True, cost_by_dim=((2, 1.0),)
    )
    second = _GateFusionLimits(
        min_size=4, max_dim=8, structure_aware=True, cost_by_dim=((2, 1.0),)
    )
    capabilities = _KernelCapabilities(
        supports_kernel_threads=False,
        thread_capacity=1,
        supports_fusion=False,
        gate_fusion=first,
    )

    assert first == second and hash(first) == hash(second)
    assert len({first, second}) == 1
    assert hash(capabilities) == hash(replace(capabilities, gate_fusion=second))


# Each pair is (first gate, second gate, blocks when structure-aware). Keys
# are diagonal (RZ, CZ), permutation (X), dense (H, RX), or undeclared (None).
STRUCTURE_PAIRS = [
    ((Key.RZ, (0,)), (Key.H, (0,)), 2),
    ((Key.H, (0,)), (Key.CZ, (0, 1)), 2),
    ((Key.X, (0,)), (Key.H, (0,)), 2),
    ((None, (0,)), (Key.RZ, (0,)), 2),
    ((Key.RZ, (0,)), (Key.X, (0,)), 1),
    ((Key.RZ, (0,)), (Key.CZ, (0, 1)), 1),
    ((Key.X, (0,)), (Key.CZ, (0, 1)), 1),
    ((Key.H, (0,)), (Key.RX, (0,)), 1),
]


@pytest.mark.parametrize("aware", [False, True], ids=["unaware", "aware"])
def test_structure_rule_merges_only_matching_classes(aware):
    # Dense merges with dense; diagonal and permutation merge with each other.
    # An undeclared key counts as dense.
    limits = replace(FORCED, structure_aware=aware)
    got = [
        len(_schedule((_gate(first, key1), _gate(second, key2)), (2, 2), limits))
        for (key1, first), (key2, second), _blocks in STRUCTURE_PAIRS
    ]

    assert got == [blocks if aware else 1 for _, _, blocks in STRUCTURE_PAIRS]


def test_kernel_specs_use_only_merge_lattice_codes():
    # The structure lattice orders diagonal, permutation, and dense only; a
    # sparse code in the spec table would have no place in it.
    assert set(_KERNEL_SPECS.values()) <= {_DENSE, _DIAGONAL, _PERMUTATION}


def test_width_limit_uses_local_hilbert_dimension():
    four = replace(FORCED, max_dim=4)
    qutrits = {0: 3, 1: 3}
    two_qutrits = (_gate((0,), dims=qutrits), _gate((1,), dims=qutrits))

    assert len(_schedule((_gate((0, 1)), _gate((0, 1))), (2, 2, 2), four)) == 1
    assert len(_schedule((_gate((0, 1)), _gate((1, 2))), (2, 2, 2), four)) == 2
    # Two qutrits span dimension 9, not 2**2.
    assert len(_schedule(two_qutrits, (3, 3), replace(FORCED, max_dim=8))) == 2
    assert len(_schedule(two_qutrits, (3, 3), replace(FORCED, max_dim=9))) == 1


@pytest.mark.parametrize(
    ("cost_by_dim", "blocks"),
    [
        (((2, 1.0), (4, 2.0)), 2),
        (((2, 1.0), (4, 1.5)), 1),
        (((2, 1.0),), 2),
    ],
    ids=["equal cost", "cheaper", "missing dimension"],
)
def test_cost_table_merges_only_when_strictly_cheaper(cost_by_dim, blocks):
    limits = replace(FORCED, cost_by_dim=cost_by_dim)

    assert len(_schedule((_gate((0,)), _gate((1,))), (2, 2), limits)) == blocks


BARRIERS = {
    "channel": steps.ApplyChannelStep(
        kraus_ops=(np.eye(2, dtype=complex),), target_indices=(0,)
    ),
    "loss": steps.LossStep(target_indices=(0,), p=0.1),
    "measurement": steps.MeasurementStep(measured_indices=(0,), classical_indices=(0,)),
    "reset": steps.ResetStep(reset_indices=(0,)),
    "put": steps.PutStep(target_indices=(0,)),
    "custom": steps.CustomOperationStep(
        operation=ops.H, implementation=None, target_indices=(0,)
    ),
    "conditioned matrix": _gate((0,), condition=((0, 1),)),
}


@pytest.mark.parametrize("barrier", BARRIERS.values(), ids=BARRIERS.keys())
def test_barrier_closes_block_and_keeps_identity(barrier):
    plan = (_gate((0,)), barrier, _gate((0,)))

    blocks = _schedule(plan, (2,), FORCED)
    fused = materialize(plan, blocks, (2,))

    assert len(blocks) == 3
    assert len(fused) == 3 and all(out is step for out, step in zip(fused, plan))


def _sweep_plan():
    """A lowered two-qubit plan whose RZ, RX, and RZ steps await binding."""
    theta = fq.Parameter("theta")
    program = fq.Program(2)
    for operation, targets in (
        (ops.RZ(theta), 0),
        (ops.CZ, (0, 1)),
        (ops.RX(theta), 0),
        (ops.RZ(theta), 0),
        (ops.H, 0),
    ):
        program.add(operation, targets)
    plan, _facts = Simulator("SV")._prepare_program(
        program, parameter_values={theta: np.array([0.0, 0.3])}
    )
    return plan


def test_materialize_rejects_unbound_parametric_plan():
    plan = _sweep_plan()

    with pytest.raises(RuntimeError, match="bound"):
        materialize(plan, schedule(plan, (2, 2), FORCED), (2, 2))


@pytest.mark.parametrize("limits", [FORCED, FORCED_AWARE], ids=["unaware", "aware"])
def test_schedule_matches_before_and_after_binding(limits):
    # Binding must not move block boundaries, so parametric steps take their
    # structure from the kernel key, not from the bound matrix: RX(0) is I.
    lowered = _sweep_plan()
    bound = list(_bind_parameter_batch(lowered, (2, 2)))
    first = lowered[0]
    # The diagonal parametric RZ merges with CZ only if its key is used.
    assert type(first) is ApplyParametricMatrixStep
    assert (
        first.implementation._kernel_key(first.operation, targets=first.targets)
        is Key.RZ
    )

    want = _schedule(lowered, (2, 2), limits)

    assert len(want) < len(lowered) and len(bound) == 2
    assert all(schedule(entry, (2, 2), limits) == want for entry in bound)


@pytest.mark.parametrize("limits", [FORCED, FORCED_AWARE], ids=["unaware", "aware"])
def test_linear_scan_groups_layers_by_width(limits):
    layers = tuple(_gate((q,), Key.X) for q in range(6))
    layers += tuple(_gate((q,), Key.Y) for q in range(6))
    dims = (2,) * 6

    blocks = _schedule(layers, dims, limits)
    targets = [
        tuple(sorted({t for i in block for t in layers[i].target_indices}))
        for block in blocks
    ]

    assert targets == [(0, 1, 2), (3, 4, 5), (0, 1, 2), (3, 4, 5)]
    assert len(_schedule(layers, dims, replace(limits, max_dim=2))) == 12
    # A barrier closes the open block even when it acts on other targets.
    measure_q5 = steps.MeasurementStep(measured_indices=(5,), classical_indices=(0,))
    assert len(_schedule((_gate((0,)), measure_q5, _gate((0,))), dims, limits)) == 3


def _fused(*gates):
    """Lower a two-qubit program and fuse it into its single step."""
    program = fq.Program(2)
    for operation, targets in gates:
        program.add(operation, targets)
    plan, _facts = Simulator("SV")._lower_program(program)
    fused = materialize(plan, schedule(plan, (2, 2), FORCED_AWARE), (2, 2))
    assert len(fused) == 1
    return fused[0]


def test_fused_step_keeps_sparse_structure_in_numba_caches():
    # A fused step has no kernel key, so the Numba caches scan its matrix and
    # keep the diagonal or permutation kernel instead of falling back to dense.
    pytest.importorskip("numba")
    from fatqat.simulator._engine.numerics.numba_kernels._matrix_cache import (
        _MatrixStructureCache,
        _SuperoperatorCache,
    )

    diagonal = _fused((ops.RZ(0.3), 0), (ops.CZ, (0, 1)))
    permutation = _fused((ops.X, 0), (ops.CZ, (0, 1)))

    assert _MatrixStructureCache().resolve(diagonal)[0] == _DIAGONAL
    assert _MatrixStructureCache().resolve(permutation)[0] == _PERMUTATION
    assert _SuperoperatorCache().resolve(diagonal)[1] == _DIAGONAL
    assert _SuperoperatorCache().resolve(permutation)[1] == _PERMUTATION


def test_neutral_gate_matrices_and_numpy_engine_import_without_numba():
    # NumPy engines and plan passes read structure codes and kernel specs from
    # the neutral module. A Numba import there breaks installs without the
    # extra, so the check runs in a fresh process with Numba blocked.
    code = """
import importlib
import sys
sys.modules['numba'] = None
for name in (
    'fatqat.simulator._engine.numerics._gate_matrices',
    'fatqat.simulator._engine.optimization.fusion',
    'fatqat.simulator._engine.engines.numpy_engine',
    'fatqat.simulator._engine.engines.base',
    'fatqat',
):
    importlib.import_module(name)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
