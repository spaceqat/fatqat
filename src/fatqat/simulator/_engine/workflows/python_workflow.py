"""Shared Python state and operator sequencing for matrix-family numerical runtimes.

Functions receive explicit evolution state and execution data. Gates, channels,
measurement, and reset use numerical functions carried by the Python execution. The engine owns initialization and export; workflows never
inspect or convert quantum arrays. NumPy handles host randomness and results.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import prod
from typing import Generic, NamedTuple

import numpy as np

from ...._backends.steps import (
    ApplyChannelStep,
    ApplyMatrixStep,
    CustomOperationStep,
    LossStep,
    MeasurementStep,
    PutStep,
    ResetStep,
    ResolvedStep,
)
from ....result import _decode_engine_indices_to_clbit_rows, reduce_to_counts
from ..._execution_contract import _RunContext
from ...custom_operations import ExecutionContext
from ..state import EvolutionState, QuantumDataT
from ._reporting import (
    ERASURE_DIGIT,
    _apply_measurement_reporting,
    _map_physical_digit,
    _report_digit,
    _reporting_by_clbit,
)


@dataclass(frozen=True, slots=True)
class _MeasurementKernels(Generic[QuantumDataT]):
    """Runtime functions shared across evolutions, with no engine or state binding.

    Projection returns native quantum storage; probabilities and sampled indices
    are host data ready for Python control flow. The sampler consumes the supplied
    generator, preserving each runtime's existing sampling algorithm and draws.
    ``shots=None`` requests one integer outcome; otherwise it returns an array.
    Compiled workflows call their kernels directly, without this Python bundle.
    """

    probabilities: Callable[[QuantumDataT], np.ndarray]
    project: Callable[[QuantumDataT, int, Sequence[int], Sequence[int]], QuantumDataT]
    sample: Callable[[np.ndarray, np.random.Generator, int | None], int | np.ndarray]


# Reset consumes randomness only for statevectors. The numerical function owns
# projection/shift or channel arithmetic; the workflow owns when reset happens.
_ResetKernel = Callable[
    [QuantumDataT, Sequence[int], Sequence[int], np.random.Generator], QuantumDataT
]


_MatrixKernel = Callable[[QuantumDataT, ApplyMatrixStep, Sequence[int]], QuantumDataT]


_ChannelKernel = Callable[
    [QuantumDataT, ApplyChannelStep, Sequence[int], np.random.Generator], QuantumDataT
]


class _PythonExecution(NamedTuple):
    # pylint: disable-next=fixme
    # TODO: Consider separate the kernels from the plan
    """Plan and numerical bindings used only by Python sequencing.

    Callables receive native state explicitly and never retain an engine.
    Bound numerical resources, including target layouts, live with this payload;
    preparing another execution cannot replace its layouts or scheduling choices.
    Compiled workflows use their own packed data without these bindings.
    """

    plan: tuple[ResolvedStep, ...]
    deferred_measurements: tuple[tuple[int, int], ...]
    apply_matrix: _MatrixKernel
    apply_channel: _ChannelKernel | None
    measurement: _MeasurementKernels | None
    reset: _ResetKernel | None


# --- physical measurement sequencing ---


def _sample_indices(
    state: EvolutionState[QuantumDataT],
    kernels: _MeasurementKernels[QuantumDataT] | None,
    shots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample final outcomes without changing the quantum state."""
    if kernels is None:
        raise NotImplementedError("operator execution has no basis-state distribution")
    return kernels.sample(kernels.probabilities(state.quantum), rng, shots)


def _collapse(
    state: EvolutionState[QuantumDataT],
    kernels: _MeasurementKernels[QuantumDataT] | None,
    dims: Sequence[int],
    indices: Sequence[int],
    rng: np.random.Generator,
) -> int:
    """Sample a physical outcome and replace the state with its projection."""
    if kernels is None:
        raise NotImplementedError("operator execution cannot represent a measurement")
    index = int(kernels.sample(kernels.probabilities(state.quantum), rng, None))
    state.quantum = kernels.project(state.quantum, index, indices, dims)
    return index


def _measure_subsystems(
    state: EvolutionState[QuantumDataT],
    kernels: _MeasurementKernels[QuantumDataT] | None,
    dims: Sequence[int],
    indices: Sequence[int],
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Collapse one grouped event and decode its physical subsystem digits."""
    assert len(indices) >= 1, "measure_subsystems requires at least one index"
    flat = _collapse(state, kernels, dims, indices, rng)
    return tuple((flat // prod(dims[:index])) % dims[index] for index in indices)


def _reset_subsystems(
    state: EvolutionState[QuantumDataT],
    kernel: _ResetKernel[QuantumDataT] | None,
    dims: Sequence[int],
    indices: Sequence[int],
    rng: np.random.Generator,
) -> None:
    """Replace quantum storage after a reset selected by the workflow."""
    if kernel is None:
        raise NotImplementedError("this representation cannot execute reset")
    state.quantum = kernel(state.quantum, indices, dims, rng)


# --- plan sequencing ---


def _condition_matches(
    condition: tuple[tuple[int, int], ...] | None, clbits: list[int] | None
) -> bool:
    """Return whether a lowered feedforward condition passes."""
    if not condition:
        return True
    assert clbits is not None, "feedforward requires initialized classical digits"
    return all(clbits[c] == v for c, v in condition)


def _run_single_pass(
    execution: _PythonExecution,
    state: EvolutionState[QuantumDataT],
    context: _RunContext,
    rng: np.random.Generator,
    *,
    state_requested: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Evolve initialized storage once and return requested outcome counts.

    The ``ResetStep`` and ``ApplyChannelStep`` branches are only reachable
    under density-matrix semantics, where both are deterministic channels;
    statevector semantics routes every plan bearing them to the dynamic
    path.
    """
    classical = state.classical
    for step in execution.plan:
        if isinstance(step, ApplyMatrixStep):
            state.quantum = execution.apply_matrix(
                state.quantum, step, context.system_dims
            )
        elif isinstance(step, ApplyChannelStep):
            state.quantum = execution.apply_channel(
                state.quantum, step, context.system_dims, rng
            )
        elif isinstance(step, ResetStep):
            _reset_subsystems(
                state, execution.reset, context.system_dims, step.reset_indices, rng
            )

    measurements = execution.deferred_measurements
    collapsed_index: int | None = None
    if state_requested and measurements:
        collapsed_index = _collapse(
            state,
            execution.measurement,
            context.system_dims,
            [q for q, _ in measurements],
            rng,
        )

    if not context.request.counts:
        return None
    if not measurements:
        indices = np.zeros(context.shots, dtype=int)
    elif collapsed_index is not None:
        indices = np.array([collapsed_index], dtype=int)
    else:
        indices = _sample_indices(state, execution.measurement, context.shots, rng)
    reporting = _reporting_by_clbit(execution.plan)
    # Without readout confusion, shots with the same sampled index report the
    # same digits, so decode each distinct index once. This is faster when
    # outcomes repeat (many shots, few measured subsystems); when nearly every
    # shot is unique it costs one extra sort. Confusion draws per shot.
    index_counts = None
    if all(confusion is None for _map, confusion in reporting.values()):
        indices, index_counts = np.unique(indices, return_counts=True)
    rows = _decode_engine_indices_to_clbit_rows(
        indices, measurements, context.system_dims, context.n_clbits
    )
    if classical.clbits is not None:
        written = {clbit for _subsystem, clbit in measurements}
        for clbit, digit in enumerate(classical.clbits):
            if clbit not in written:
                rows[:, clbit] = digit
    _apply_measurement_reporting(rows, reporting, rng)
    # Indices differing only in unmeasured subsystems report the same key;
    # weighting by index counts merges them.
    return reduce_to_counts(rows, index_counts)


def _run_one_shot(
    execution: _PythonExecution,
    state: EvolutionState[QuantumDataT],
    context: _RunContext,
    rng: np.random.Generator,
) -> EvolutionState[QuantumDataT]:
    """Run one dynamic-path shot and return its final evolution state.

    Shared by local and process-worker batches. Custom implementations may
    replace the state; built-in operations use the execution's numerical functions.

    Classical storage was initialized with the quantum state. Allocate
    report digits only when a measurement or condition needs them.
    """
    classical = state.classical
    clbits = classical.clbits
    occupied = classical.occupied
    for step in execution.plan:
        if clbits is None and (
            isinstance(step, (MeasurementStep, CustomOperationStep))
            or step.condition is not None
        ):
            clbits = [0] * context.n_clbits
            classical.clbits = clbits
        if isinstance(step, CustomOperationStep):
            if _condition_matches(step.condition, clbits):
                custom_context = ExecutionContext(
                    targets=step.target_indices,
                    system_dims=context.system_dims,
                    rng=rng,
                )
                state = step.implementation.execute(
                    step.operation, state, custom_context
                )
                # Returned containers can replace every field; do not keep stale aliases.
                classical = state.classical
                clbits = classical.clbits
                occupied = classical.occupied
        elif isinstance(step, ApplyMatrixStep) and (
            occupied is None or all(t in occupied for t in step.target_indices)
        ):
            if _condition_matches(step.condition, clbits):
                state.quantum = execution.apply_matrix(
                    state.quantum, step, context.system_dims
                )
        elif isinstance(step, ApplyChannelStep) and (
            occupied is None or all(t in occupied for t in step.target_indices)
        ):
            if _condition_matches(step.condition, clbits):
                state.quantum = execution.apply_channel(
                    state.quantum, step, context.system_dims, rng
                )
        elif isinstance(step, LossStep):
            if _condition_matches(step.condition, clbits):
                if occupied is None:
                    occupied = set(range(len(context.system_dims)))
                    classical.occupied = occupied
                for index in step.target_indices:
                    if index in occupied and rng.random() < step.p:
                        occupied.discard(index)
                        _reset_subsystems(
                            state, execution.reset, context.system_dims, (index,), rng
                        )
        elif isinstance(step, PutStep):
            if _condition_matches(step.condition, clbits) and occupied is not None:
                for index in step.target_indices:
                    if index not in occupied:
                        occupied.add(index)
                        _reset_subsystems(
                            state, execution.reset, context.system_dims, (index,), rng
                        )
        elif isinstance(step, MeasurementStep):
            assert clbits is not None, "measurement requires classical digits"
            bits = _measure_subsystems(
                state,
                execution.measurement,
                context.system_dims,
                step.measured_indices,
                rng,
            )
            confusions = step.confusions or (None,) * len(bits)
            maps = step.reported_digit_maps or (None,) * len(bits)
            # The collapse keeps the physical outcome; only its mapped,
            # optionally confused report is written to the clbits.
            for m, c, bit, reported_map, confusion in zip(
                step.measured_indices,
                step.classical_indices,
                bits,
                maps,
                confusions,
            ):
                if occupied is not None and m not in occupied:
                    clbits[c] = ERASURE_DIGIT
                else:
                    clbits[c] = _report_digit(
                        _map_physical_digit(bit, reported_map), confusion, rng
                    )
        elif isinstance(step, ResetStep):
            if _condition_matches(step.condition, clbits) and (
                occupied is None or all(t in occupied for t in step.reset_indices)
            ):
                _reset_subsystems(
                    state, execution.reset, context.system_dims, step.reset_indices, rng
                )
    return state


def _run_operator_plan(
    execution: _PythonExecution,
    state: EvolutionState[QuantumDataT],
    dims: Sequence[int],
    rng: np.random.Generator,
) -> None:
    """Apply an operator plan to already-initialized native storage."""
    for step in execution.plan:
        assert not isinstance(
            step, MeasurementStep
        ), "operator execution cannot represent a measurement"
        assert (
            step.condition is None
        ), "operator execution cannot represent a feedforward condition"
        if isinstance(step, ApplyMatrixStep):
            state.quantum = execution.apply_matrix(state.quantum, step, dims)
        elif isinstance(step, ApplyChannelStep):
            state.quantum = execution.apply_channel(state.quantum, step, dims, rng)
        elif isinstance(step, ResetStep):
            _reset_subsystems(state, execution.reset, dims, step.reset_indices, rng)
        else:
            raise TypeError(f"unknown resolved execution step {type(step).__name__}")
