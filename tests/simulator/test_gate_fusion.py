"""Step-level gate fusion: limits, block scheduling, fused steps, and engine runs."""

import itertools
import subprocess
import sys
from dataclasses import replace
from math import prod
from types import SimpleNamespace

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat._backends import steps
from fatqat._backends.steps import ApplyMatrixStep, ApplyParametricMatrixStep
from fatqat._backends.steps import BuiltinKernelKey as Key
from fatqat.errors import BackendValidationError
from fatqat.simulator import ClassicalState, EvolutionState, Simulator
from fatqat.simulator._engine.binding import _bind_parameter_batch
from fatqat.simulator._engine.engines import base
from fatqat.simulator._engine.engines.numpy_engine import NumpyFixedPhotonEngine
from fatqat.simulator._engine.numerics._gate_matrices import (
    _DENSE,
    _DIAGONAL,
    _KERNEL_SPECS,
    _PERMUTATION,
)
from fatqat.simulator._engine.optimization import fusion
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


# ---- engine runs ------------------------------------------------------------
# Default limits skip small systems, so each test declares forced limits on
# the engine class that `Simulator` builds, and observes the gate pass through
# the `fusion` module attributes that the engine entry calls.

SEED = 20261103
STATE_ONLY = {"counts": False, "final_state": True}
STATE_GETTERS = {"SV": "get_statevector", "DM": "get_density_matrix"}
FUSION_ENGINES = [("SV", "numpy"), ("DM", "numpy"), ("SV", "numba"), ("DM", "numba")]


def _add_brick_layers(program, n, layers, rng):
    for layer in range(layers):
        for qubit in range(n):
            program.add(ops.RX(float(rng.uniform(0, 2 * np.pi))), qubit)
        for qubit in range(n):
            program.add(ops.RZ(float(rng.uniform(0, 2 * np.pi))), qubit)
        for qubit in range(layer % 2, n - 1, 2):
            program.add(ops.CZ, (qubit, qubit + 1))
    return program


def _brick(n, layers, clbits=0):
    """RX and RZ on every qubit, then CZ on alternating pairs, per layer."""
    program = fq.Program(n, clbits)
    return _add_brick_layers(program, n, layers, np.random.default_rng(SEED))


def _qft(n):
    program = fq.Program(n)
    for j in range(n):
        program.add(ops.H, j)
        for k in range(j + 1, n):
            program.add(ops.CPhase(np.pi / 2 ** (k - j)), (k, j))
    for j in range(n // 2):
        program.add(ops.Swap, (j, n - 1 - j))
    return program


def _mixed(n, layers=3):
    """Asymmetric gates whose lowered targets descend.

    Lowering maps program qubit q to index n-1-q, so ``CX(q, q+1)`` lowers to
    descending target indices; a fused matrix must keep that order.
    """
    rng = np.random.default_rng(SEED)
    program = fq.Program(n)
    for _ in range(layers):
        for qubit in range(n):
            program.add(ops.RX(float(rng.uniform(0, 2 * np.pi))), qubit)
            program.add(ops.RZ(float(rng.uniform(0, 2 * np.pi))), qubit)
        for qubit in range(n - 1):
            program.add(ops.CX, (qubit, qubit + 1))
        for qubit in reversed(range(n)):
            program.add(ops.RY(float(rng.uniform(0, 2 * np.pi))), qubit)
    return program


def _qudits(layers=2):
    """Two qutrits and a qubit, so a pass that assumes qubit dimensions fails.

    Every gate is dense, so blocks form under both structure rules. A qutrit
    and the qubit (dimension 6) merge, which embeds qutrit matrices; the two
    qutrits (dimension 9) exceed the forced width limit and never merge.
    """
    rng = np.random.default_rng(SEED)
    qutrits = fq.QuantumRegister(2, dim=3)
    qubit = fq.QuantumRegister(1)
    program = fq.Program([qutrits, qubit])
    for _ in range(layers):
        for qutrit in qutrits:
            program.add(ops.SubspaceRX(float(rng.uniform(0, np.pi)), (0, 1)), qutrit)
            program.add(ops.SubspaceRY(float(rng.uniform(0, np.pi)), (1, 2)), qutrit)
        program.add(ops.RX(float(rng.uniform(0, np.pi))), qubit[0])
        program.add(ops.RY(float(rng.uniform(0, np.pi))), qubit[0])
        program.add(ops.Sum, (qutrits[0], qutrits[1]))
    return program


CIRCUITS = {
    "brick": lambda: _brick(5, 3),
    "qft": lambda: _qft(5),
    "mixed": lambda: _mixed(5),
    "qudits": _qudits,
}


def _engine_class(method, runtime):
    return type(Simulator(method=method, runtime=runtime)._engine)


def _force_limits(monkeypatch, method, runtime, limits=FORCED):
    """Declare ``limits`` and fusion support on the engine class."""
    engine_cls = _engine_class(method, runtime)
    capabilities = engine_cls._capabilities
    kernels = replace(capabilities.kernels, gate_fusion=limits, supports_fusion=True)
    monkeypatch.setattr(
        engine_cls, "_capabilities", replace(capabilities, kernels=kernels)
    )
    return engine_cls


def _record(monkeypatch, engine_cls):
    """Count gate passes and record the plans that ``engine_cls`` receives."""
    record = SimpleNamespace(
        calls=0, executed=[], prepared=[], workflows=[], dims=None, process_plans=[]
    )
    original_schedule = fusion.schedule
    original_execute = engine_cls.execute
    original_prepare = engine_cls.prepare_payload
    original_processes = base._run_shots_in_processes

    def counting_schedule(*args, **kwargs):
        record.calls += 1
        return original_schedule(*args, **kwargs)

    def recording_execute(engine, plan, **kwargs):
        record.executed.append(plan)
        return original_execute(engine, plan, **kwargs)

    def recording_prepare(engine, plan, **kwargs):
        record.prepared.append(plan)
        record.workflows.append(kwargs["policy"].workflow)
        record.dims = kwargs["system_dims"]
        return original_prepare(engine, plan, **kwargs)

    def recording_processes(factory, context, plan, policy, initial_state):
        record.process_plans.append(plan)
        return original_processes(factory, context, plan, policy, initial_state)

    monkeypatch.setattr(fusion, "schedule", counting_schedule)
    monkeypatch.setattr(engine_cls, "execute", recording_execute)
    monkeypatch.setattr(engine_cls, "prepare_payload", recording_prepare)
    monkeypatch.setattr(base, "_run_shots_in_processes", recording_processes)
    return record


def _final_state(program, method, runtime, *, fused):
    result = (
        Simulator(method=method, runtime=runtime)
        .run(
            program,
            shots=0,
            result_config=STATE_ONLY,
            simulation_config={"fusion": fused},
        )
        .result()
    )
    return getattr(result, STATE_GETTERS[method])()


def _matrix_count(plan):
    return sum(isinstance(step, ApplyMatrixStep) for step in plan)


def _widest(plan, dims):
    return max(prod(dims[target] for target in step.target_indices) for step in plan)


def _same_steps(output, plan):
    return len(output) == len(plan) and all(a is b for a, b in zip(output, plan))


def _refuse(*_args, **_kwargs):
    raise AssertionError("the gate pass ran with fusion disabled")


@pytest.mark.parametrize(("method", "runtime"), FUSION_ENGINES)
def test_disabled_fusion_never_runs_the_gate_pass(monkeypatch, method, runtime):
    _force_limits(monkeypatch, method, runtime)
    monkeypatch.setattr(fusion, "schedule", _refuse)
    monkeypatch.setattr(fusion, "materialize", _refuse)

    state = _final_state(_brick(5, 3), method, runtime, fused=False)

    assert np.all(np.isfinite(state))


EQUIVALENCE_CELLS = {
    "numpy-SV": ("SV", "numpy", FORCED),
    "numpy-DM": ("DM", "numpy", FORCED),
    "numba-SV": ("SV", "numba", FORCED),
    "numba-SV-aware": ("SV", "numba", FORCED_AWARE),
    "numba-DM": ("DM", "numba", FORCED),
    "numba-DM-aware": ("DM", "numba", FORCED_AWARE),
}


@pytest.mark.parametrize("circuit", sorted(CIRCUITS))
@pytest.mark.parametrize("cell", list(EQUIVALENCE_CELLS))
def test_fused_run_matches_unfused_run(monkeypatch, cell, circuit):
    method, runtime, limits = EQUIVALENCE_CELLS[cell]
    record = _record(monkeypatch, _force_limits(monkeypatch, method, runtime, limits))
    program = CIRCUITS[circuit]()

    reference = _final_state(program, method, runtime, fused=False)
    unfused = record.prepared.pop()
    state = _final_state(program, method, runtime, fused=True)
    fused = record.prepared.pop()

    # The policy is chosen before the gate pass; fusion must not move the run
    # off the Python workflow that receives the fused plan.
    assert record.workflows == ["python", "python"]
    assert _matrix_count(fused) < _matrix_count(unfused)
    # Width is bounded in physical dimensions; single gates may exceed it.
    assert _widest(fused, record.dims) <= max(
        limits.max_dim, _widest(unfused, record.dims)
    )
    assert np.allclose(state, reference, atol=1e-10)


def _mid_circuit_program():
    """A mid-circuit measurement makes every shot evolve on its own."""
    program = _brick(4, 2, clbits=5)
    program.measure(0, 4)
    _add_brick_layers(program, 4, 2, np.random.default_rng(3))
    program.measure(tuple(range(4)), tuple(range(4)))
    return program


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
def test_process_shots_run_the_plan_fused_once_in_the_parent(monkeypatch, runtime):
    record = _record(monkeypatch, _force_limits(monkeypatch, "SV", runtime))

    def counts(fused):
        config = {"seed": 7, "shot_parallelism": "processes", "fusion": fused}
        return (
            Simulator("SV", runtime=runtime)
            .run(_mid_circuit_program(), shots=64, simulation_config=config)
            .result()
            .get_counts()
        )

    reference = counts(False)
    unfused = record.process_plans.pop()
    fused_counts = counts(True)
    fused = record.process_plans.pop()

    assert record.calls == 1
    assert len(fused) < len(unfused)
    assert len(reference) > 1
    assert fused_counts == reference


@pytest.mark.parametrize("method", ["SV", "DM"])
def test_minimum_size_compares_with_the_physical_dimension(monkeypatch, method):
    limits = replace(FORCED, min_size=2**5)
    record = _record(monkeypatch, _force_limits(monkeypatch, method, "numpy", limits))

    _final_state(_brick(5, 2), method, "numpy", fused=False)
    _final_state(_brick(5, 2), method, "numpy", fused=True)
    _final_state(_brick(4, 2), method, "numpy", fused=True)
    unfused, fused, below = record.prepared
    below_input = record.executed[-1]

    assert _matrix_count(fused) < _matrix_count(unfused)
    # Without the size threshold the smaller plan would merge.
    assert len(schedule(below_input, (2,) * 4, FORCED)) < len(below_input)
    assert _same_steps(below, below_input)


@pytest.mark.parametrize("shot_parallelism", ["serial", "processes"])
def test_sampled_expectation_tail_is_never_fused(monkeypatch, shot_parallelism):
    engine_cls = _force_limits(monkeypatch, "SV", "numpy")
    phase = {"tail": False}
    calls = {"body": 0, "tail": 0, "tail_dispatch": 0}
    original_schedule = fusion.schedule
    original_continue = engine_cls.continue_from_states
    original_dispatch = engine_cls._dispatch_by_shot_strategy

    def counting_schedule(*args, **kwargs):
        calls["tail" if phase["tail"] else "body"] += 1
        return original_schedule(*args, **kwargs)

    def continue_tail(engine, *args, **kwargs):
        phase["tail"] = True
        try:
            return original_continue(engine, *args, **kwargs)
        finally:
            phase["tail"] = False

    def counting_dispatch(engine, *args, **kwargs):
        calls["tail_dispatch"] += phase["tail"]
        return original_dispatch(engine, *args, **kwargs)

    monkeypatch.setattr(fusion, "schedule", counting_schedule)
    monkeypatch.setattr(engine_cls, "continue_from_states", continue_tail)
    monkeypatch.setattr(engine_cls, "_dispatch_by_shot_strategy", counting_dispatch)
    config = {"seed": 5, "fusion": True}
    initial_state = None
    if shot_parallelism == "processes":
        # Atom occupancy needs per-shot evolution, so the automatic strategy
        # runs the tail's shots in worker processes.
        occupied = ClassicalState(occupied={0, 1, 2, 3})
        initial_state = EvolutionState(classical=occupied)
        config["max_workers"] = 2
    else:
        config["shot_parallelism"] = "serial"

    fq.Estimator(Simulator("SV", runtime="numpy")).run(
        _brick(4, 2),
        fq.Observable([("ZZII", 1.0), ("XIII", 0.5)]),
        shots=64,
        initial_state=initial_state,
        simulation_config=config,
    ).result()

    assert calls["body"] >= 1
    assert calls["tail"] == 0
    # The tail dispatches only on the process branch.
    assert (calls["tail_dispatch"] > 0) == (shot_parallelism == "processes")


@pytest.mark.parametrize("method", ["unitary", "superop"])
def test_operator_engines_keep_their_plan_with_fusion_enabled(monkeypatch, method):
    # Operator engines fuse their own payloads; a gate pass on top would
    # change the operator fusion they already perform.
    record = _record(monkeypatch, _engine_class(method, "numba"))

    Simulator(method=method, runtime="numba").run(
        _brick(5, 3),
        shots=0,
        result_config=STATE_ONLY,
        simulation_config={"fusion": True},
    ).result()

    assert record.calls == 0
    (executed,) = record.executed
    (prepared,) = record.prepared
    assert _same_steps(prepared, executed)


def test_sweep_fuses_every_bound_entry(monkeypatch):
    record = _record(monkeypatch, _force_limits(monkeypatch, "SV", "numpy"))
    phi = fq.Parameter("phi")
    program = _brick(4, 2)
    program.add(ops.RX(phi), 0)
    _add_brick_layers(program, 4, 1, np.random.default_rng(1))

    def states(fused):
        results = (
            Simulator("SV", runtime="numpy")
            .run_sweep(
                program,
                {phi: np.array([0.1, 0.7, 2.0])},
                shots=0,
                result_config=STATE_ONLY,
                simulation_config={"fusion": fused},
            )
            .result()
        )
        return [result.get_statevector() for result in results]

    reference = states(False)
    fused = states(True)

    assert record.calls == 3
    # Distinct entries, so reusing one entry's fused plan cannot pass.
    assert not any(
        np.allclose(a, b, atol=1e-10) for a, b in itertools.combinations(reference, 2)
    )
    assert all(np.allclose(a, b, atol=1e-10) for a, b in zip(fused, reference))


# ---- default limits and the fusion switch -----------------------------------
# Minimum sizes come from brick-circuit timing: fusion pays off only from the
# listed physical dimension upward. Timing showed no gain on the Numba density
# matrix, so its minimum size is an uncalibrated starting point.

DEFAULT_LIMITS = {
    ("SV", "numpy"): _GateFusionLimits(2**20, max_dim=8, structure_aware=False),
    ("DM", "numpy"): _GateFusionLimits(2**10, max_dim=8, structure_aware=False),
    ("SV", "numba"): _GateFusionLimits(2**21, max_dim=8, structure_aware=True),
    ("DM", "numba"): _GateFusionLimits(2**7, max_dim=4, structure_aware=True),
}


@pytest.mark.parametrize(("method", "runtime"), list(DEFAULT_LIMITS))
def test_state_engines_declare_calibrated_default_limits(method, runtime):
    kernels = _engine_class(method, runtime)._capabilities.kernels

    # Limits without the switch, or the switch without limits, would make
    # `fusion=True` either unreachable or a silent no-op.
    assert kernels.supports_fusion is True
    assert kernels.gate_fusion == DEFAULT_LIMITS[(method, runtime)]


def test_default_minimum_size_decides_fusion_on_numpy_statevector(monkeypatch):
    # Calibration found a gain from 20 qubits upward and a loss below.
    record = _record(monkeypatch, _engine_class("SV", "numpy"))

    _final_state(_brick(20, 1), "SV", "numpy", fused=True)
    _final_state(_brick(19, 1), "SV", "numpy", fused=True)
    at_size, below = record.prepared
    at_size_input, below_input = record.executed

    assert _matrix_count(at_size) < _matrix_count(at_size_input)
    # Without the size threshold the smaller plan would merge.
    assert len(schedule(below_input, (2,) * 19, FORCED)) < len(below_input)
    assert _same_steps(below, below_input)


@pytest.mark.parametrize("method", ["SV", "DM"])
def test_numpy_state_engines_accept_enabling_fusion(method):
    result = (
        Simulator(method=method, runtime="numpy")
        .run(
            _brick(3, 1),
            shots=0,
            result_config=STATE_ONLY,
            simulation_config={"fusion": True},
        )
        .result()
    )

    assert result.metadata["simulation_config"]["fusion"] is True


@pytest.mark.parametrize("method", ["unitary", "superop"])
def test_numpy_operator_engines_reject_enabling_fusion(method):
    # These engines have neither operator fusion nor gate fusion limits, so
    # accepting the switch would promise a speedup that never happens.
    with pytest.raises(BackendValidationError, match="not supported.*matrix engine"):
        Simulator(method=method, runtime="numpy").run(
            _brick(3, 1), simulation_config={"fusion": True}
        )


def test_numpy_fixed_photon_engine_declares_no_fusion():
    kernels = NumpyFixedPhotonEngine._capabilities.kernels

    assert kernels.gate_fusion is None
    assert kernels.supports_fusion is False
