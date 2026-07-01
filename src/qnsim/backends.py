"""Qubit statevector backend: validate, execute, assemble Result, return Job."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .engine import StateVectorEngine
from .errors import BackendValidationError, NoMeasurementWarning, UnsupportedOperationError
from .implementation import ApplyMatrixStep, default_implementation_map
from .job import Job
from .layout import ResourceLayout
from .operations import ResetGate
from .program import AppliedOperation, Measurement, Program
from .result import Result, ResultConfig, build_counts, count_key_from_clbits


@dataclass(frozen=True)
class MeasurementStep:
    """Resolved measurement: flat qubit index into flat clbit index."""

    qubit_index: int
    clbit_index: int


@dataclass(frozen=True)
class ResetStep:
    """Resolved reset of one flat qubit index to |0>, with optional condition.

    `Reset` is an `AppliedOperation`, so it can carry a feedforward `condition`
    just like a gate. The lowered form stores it as ``(clbit_index, value)``
    AND-terms; the per-shot loop skips the reset when the guard fails.
    """

    qubit_index: int
    condition: tuple[tuple[int, int], ...] | None = None


ResolvedStep = ApplyMatrixStep | MeasurementStep | ResetStep


@dataclass(frozen=True)
class _PlanFacts:
    """Classification facts computed while lowering a program."""

    is_dynamic: bool
    has_measurement: bool
    has_reset: bool


@dataclass(frozen=True)
class _ResultRequest:
    """Resolved result fields requested for one execution."""

    counts: bool
    statevector: bool


def _resolve_condition(
    condition: tuple[tuple[object, int], ...] | None,
    layout: ResourceLayout,
) -> tuple[tuple[int, int], ...] | None:
    """Lower a frontend condition to ``(clbit_index, value)`` AND-terms."""
    if condition is None:
        return None
    return tuple((layout.clbit_index(ref), int(val)) for ref, val in condition)


def _condition_matches(
    condition: tuple[tuple[int, int], ...] | None,
    clbits: list[int],
) -> bool:
    """Return whether a lowered feedforward condition passes."""
    return condition is None or all(clbits[c] == v for c, v in condition)


def _resolve_result_request(config: ResultConfig, facts: _PlanFacts) -> _ResultRequest:
    """Resolve default result fields from config and lowered program facts."""
    stochastic = facts.has_measurement or facts.has_reset
    counts = config.counts if config.counts is not None else facts.has_measurement
    statevector = config.statevector
    if statevector is None:
        statevector = not stochastic
    return _ResultRequest(counts=counts, statevector=statevector)


class StateVectorBackend:
    """Phase 1 statevector backend for qubit programs.

    The backend validates a `Program`, evolves supported operations with the
    matrix engine, samples terminal measurements, and returns an eager `Job`.
    A backend instance reuses one engine across runs, so it is suitable for
    repeated single-threaded use but not concurrent `run` calls.
    """

    def __init__(self) -> None:
        """Create a statevector backend."""
        self._impl_map = default_implementation_map()
        # The engine is constructed once and re-initialized per run so its
        # compiled kernels can be reused. Because it holds per-run state, a
        # single backend instance is NOT safe for concurrent run() calls
        # (single-threaded use only).
        self._engine = StateVectorEngine()

    def resolve_layout(self, program: Program) -> ResourceLayout:
        """Build the flat resource layout used by this backend.

        Args:
            program: Program whose registers should be flattened.

        Returns:
            Resource layout mapping register references to flat indices.
        """
        return ResourceLayout.from_program(program)

    def run(
        self,
        program: Program,
        *,
        shots: int = 1024,
        result_config: ResultConfig | None = None,
        seed: int | None = None,
    ) -> Job:
        """Validate and execute a program.

        Counts default to available when the program contains measurements.
        Statevector output defaults to available only when execution is
        deterministic (no measurement/reset sampling).

        Args:
            program: Program to execute.
            shots: Number of samples used when counts are requested.
            result_config: Optional `ResultConfig` controlling produced fields.
            seed: Optional random seed for this run.

        Returns:
            A completed `Job`. The job result includes metadata with the shot
            count, backend name, and effective result config. Validation
            failures raise directly; execution failures are captured in an
            error job whose `result()` re-raises.

        Raises:
            BackendValidationError: If requested fields are incompatible with
                the program or shot count.
            UnsupportedOperationError: If the program uses unsupported
                operations.

        Examples:
            ```python
            import qnsim as qs

            program = qs.Program(1, 1)
            program.add(qs.ops.X, 0)
            program.add_measurement(0, 0)

            result = qs.StateVectorBackend().run(
                program,
                shots=100,
                result_config=qs.ResultConfig(counts=True),
            ).result()
            counts = result.get_counts()
            ```
        """
        config = result_config if result_config is not None else ResultConfig()
        layout = self.resolve_layout(program)
        plan, facts = self._lower(program, layout)
        self._validate(config, shots, facts)
        try:
            return Job.done(
                self._execute(
                    config, shots, plan, facts, layout.n_qubits, layout.n_clbits, seed
                )
            )
        except Exception as exc:  # execution-stage failure
            return Job.failed(exc)

    # --- validation (raises directly from run) ---
    def _validate(
        self,
        config: ResultConfig,
        shots: int,
        facts: _PlanFacts,
    ) -> None:
        """Validate result-config / shots constraints against the lowered program.

        Operation support and dynamic classification were already resolved in
        `_lower`. Mid-circuit measurement and conditional operations are now
        supported, so they are not rejected here.
        """
        request = _resolve_result_request(config, facts)
        stochastic = facts.has_measurement or facts.has_reset
        requested_sv = config.statevector is True

        if (request.counts or (requested_sv and stochastic)) and type(shots) is not int:
            raise BackendValidationError(
                f"shots must be an int when requested results depend on it, got {shots!r}"
            )
        if request.counts and shots <= 0:
            raise BackendValidationError(f"counts require shots > 0, got shots={shots}")
        if requested_sv and stochastic and shots != 1:
            raise BackendValidationError(
                "statevector with measurement or reset is only supported for shots == 1"
            )

    # --- execution ---
    def _execute(
        self,
        config: ResultConfig,
        shots: int,
        plan: list[ResolvedStep],
        facts: _PlanFacts,
        n_qubits: int,
        n_clbits: int,
        seed: int | None,
    ) -> Result:
        """Execute a lowered program and assemble the requested result fields."""
        rng = np.random.default_rng(seed)
        request = _resolve_result_request(config, facts)

        if facts.is_dynamic:
            counts, statevector, available = self._run_per_shot(
                plan, n_qubits, n_clbits, shots, rng, request
            )
        else:
            counts, statevector, available = self._run_fast(
                plan,
                facts,
                n_qubits,
                n_clbits,
                shots,
                rng,
                request,
            )

        # NoMeasurementWarning: counts produced, some clbit never written, no state.
        if request.counts and "statevector" not in available:
            written = {s.clbit_index for s in plan if isinstance(s, MeasurementStep)}
            if any(c not in written for c in range(n_clbits)):
                warnings.warn(
                    "counts contain clbits that were never measured; "
                    "returning zero-filled counts",
                    NoMeasurementWarning,
                    stacklevel=3,
                )

        return Result(
            counts=counts,
            statevector=statevector,
            available=frozenset(available),
            metadata={
                "shots": shots,
                "backend_name": type(self).__name__,
                "result_config": config,
            },
        )

    def _run_fast(
        self,
        plan: list[ResolvedStep],
        facts: _PlanFacts,
        n_qubits: int,
        n_clbits: int,
        shots: int,
        rng: np.random.Generator,
        request: _ResultRequest,
    ) -> tuple[dict[str, int] | None, np.ndarray | None, set[str]]:
        """Phase 1 path: evolve once, then sample terminal measurements."""
        engine = self._engine
        engine.initialize(n_qubits)
        measurements: list[tuple[int, int]] = []
        for step in plan:
            if isinstance(step, ApplyMatrixStep):
                engine.apply(step)
            else:  # MeasurementStep (no ResetStep on the fast path)
                measurements.append((step.qubit_index, step.clbit_index))

        counts: dict[str, int] | None = None
        statevector: np.ndarray | None = None
        available: set[str] = set()

        collapsed_index = None
        if request.statevector and facts.has_measurement:
            measured_qubits = [q for q, _c in measurements]
            collapsed_index = engine.collapse(measured_qubits, rng)

        if request.counts:
            if facts.has_measurement:
                if collapsed_index is not None:
                    indices = np.array([collapsed_index], dtype=int)
                else:
                    indices = engine.sample_indices(shots, rng)
            else:
                indices = np.zeros(shots, dtype=int)
            counts = build_counts(indices, n_clbits, measurements)
            available.add("counts")

        if request.statevector:
            statevector = engine.export_state()
            available.add("statevector")

        return counts, statevector, available

    def _run_per_shot(
        self,
        plan: list[ResolvedStep],
        n_qubits: int,
        n_clbits: int,
        shots: int,
        rng: np.random.Generator,
        request: _ResultRequest,
    ) -> tuple[dict[str, int] | None, np.ndarray | None, set[str]]:
        """Per-shot path: run each shot independently with its own clbits."""
        engine = self._engine
        # Counts need one trajectory per shot; statevector-only runs need one
        # representative trajectory, so shots=0 cannot leave the engine
        # uninitialized.
        n_iters = shots if request.counts else (1 if request.statevector else 0)
        counts: dict[str, int] | None = {} if request.counts else None
        for _ in range(n_iters):
            engine.initialize(n_qubits)
            clbits = [0] * n_clbits
            for step in plan:
                if isinstance(step, ApplyMatrixStep):
                    if _condition_matches(step.condition, clbits):
                        engine.apply(step)
                elif isinstance(step, MeasurementStep):
                    clbits[step.clbit_index] = engine.measure_qubit(
                        step.qubit_index, rng
                    )
                else:  # ResetStep also honors its feedforward condition.
                    if _condition_matches(step.condition, clbits):
                        engine.reset_qubit(step.qubit_index, rng)
            if counts is not None:
                key = count_key_from_clbits(clbits, n_clbits)
                counts[key] = counts.get(key, 0) + 1

        statevector: np.ndarray | None = None
        available: set[str] = set()

        if request.counts:
            available.add("counts")
        if request.statevector:
            statevector = engine.export_state()
            available.add("statevector")

        return counts, statevector, available

    def _lower(
        self, program: Program, layout: ResourceLayout
    ) -> tuple[list[ResolvedStep], _PlanFacts]:
        """Lower a program into an execution plan and classify it, in one pass.

        Raises `UnsupportedOperationError` for a gate with no matrix rule.
        `Reset` is recognized by type and routed to a `ResetStep`. The pass also
        computes `is_dynamic` (reset, a condition, or a gate on an
        already-measured qubit), `has_measurement`, and `has_reset`.
        """
        plan: list[ResolvedStep] = []
        measured_qubits: set[int] = set()
        is_dynamic = False
        has_measurement = False
        has_reset = False

        for step in program.operations:
            if isinstance(step, Measurement):
                has_measurement = True
                q = layout.qubit_index(step.qreg)
                c = layout.clbit_index(step.clreg)
                measured_qubits.add(q)
                plan.append(MeasurementStep(qubit_index=q, clbit_index=c))
                continue

            if isinstance(step, AppliedOperation):
                target_indices = tuple(layout.qubit_index(t) for t in step.targets)
                if step.condition is not None:
                    is_dynamic = True
                if any(t in measured_qubits for t in target_indices):
                    is_dynamic = True

                if isinstance(step.operation, ResetGate):
                    has_reset = True
                    is_dynamic = True
                    cond = _resolve_condition(step.condition, layout)
                    plan.append(
                        ResetStep(qubit_index=target_indices[0], condition=cond)
                    )
                    continue

                rule = self._impl_map.get(type(step.operation))
                if rule is None:
                    raise UnsupportedOperationError(type(step.operation).__name__)
                matrix = rule(step)
                cond = _resolve_condition(step.condition, layout)
                plan.append(
                    ApplyMatrixStep(
                        matrix=matrix, target_indices=target_indices, condition=cond
                    )
                )

        return (
            plan,
            _PlanFacts(
                is_dynamic=is_dynamic,
                has_measurement=has_measurement,
                has_reset=has_reset,
            ),
        )
