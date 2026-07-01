"""Qubit statevector backend: validate, execute, assemble Result, return Job."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .engine import StateVectorEngine
from .errors import BackendValidationError, NoMeasurementWarning, UnsupportedOperationError
from .implementation import ApplyMatrixStep, default_implementation_map
from .implementation import MatrixImplementationMap
from .job import Job
from .layout import ResourceLayout
from .program import AppliedOperation, Measurement, Program
from .result import Result, ResultConfig, build_counts


@dataclass(frozen=True)
class MeasurementStep:
    """Resolved terminal measurement: flat qubit index into flat clbit index."""

    qubit_index: int
    clbit_index: int


ResolvedStep = ApplyMatrixStep | MeasurementStep


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
        Statevector output defaults to available only when there are no
        measurements. Measurement is terminal in Phase 1: gates after a
        measurement and conditional operations are rejected.

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
                operations, conditions, or mid-circuit measurement.

        Examples:
            ```python
            import qnsim as qs

            program = qs.Program(1, 1)
            program.add(qs.ops.X, 0)
            program.add_measurement(0, 0)

            result = qs.StateVectorBackend().run(
                program,
                shots=100,
                seed=0,
                result_config=qs.ResultConfig(counts=True),
            ).result()
            counts = result.get_counts()
            ```
        """
        config = result_config if result_config is not None else ResultConfig()
        layout = self.resolve_layout(program)
        self._validate(program, config, shots, layout)
        try:
            return Job.done(self._execute(program, config, shots, layout, seed))
        except Exception as exc:  # execution-stage failure
            return Job.failed(exc)

    # --- validation (raises directly from run) ---
    def _validate(
        self,
        program: Program,
        config: ResultConfig,
        shots: int,
        layout: ResourceLayout,
    ) -> None:
        """Validate Phase 1 backend constraints before execution begins."""
        seen_measurement = False
        has_measurement = False
        for step in program.operations:
            if isinstance(step, Measurement):
                seen_measurement = True
                has_measurement = True
                continue
            if isinstance(step, AppliedOperation):
                if seen_measurement:
                    raise UnsupportedOperationError(
                        "mid-circuit measurement is not supported in Phase 1 "
                        "(a gate appears after a measurement)"
                    )
                if step.condition is not None:
                    raise UnsupportedOperationError(
                        "conditional (feedforward) operations are not supported in Phase 1"
                    )
                if self._impl_map.get(type(step.operation)) is None:
                    raise UnsupportedOperationError(type(step.operation).__name__)
        effective_counts = config.counts if config.counts is not None else has_measurement
        measured_statevector = config.statevector is True and has_measurement
        if (effective_counts or measured_statevector) and type(shots) is not int:
            raise BackendValidationError(
                f"shots must be an int when requested results depend on it, got {shots!r}"
            )
        if effective_counts and shots <= 0:
            raise BackendValidationError(
                f"counts require shots > 0, got shots={shots}"
            )
        if measured_statevector and shots != 1:
            raise BackendValidationError(
                "statevector with measurement is only supported for shots == 1 "
                "in Phase 1"
            )

    # --- execution ---
    def _execute(
        self,
        program: Program,
        config: ResultConfig,
        shots: int,
        layout: ResourceLayout,
        seed: int | None,
    ) -> Result:
        """Execute a validated program and assemble the requested result fields."""
        plan = self._resolve_program(program, self._impl_map, layout)
        engine = self._engine
        engine.initialize(layout.n_qubits)
        measurements: list[tuple[int, int]] = []
        for step in plan:
            if isinstance(step, ApplyMatrixStep):
                engine.apply(step)
            else:  # MeasurementStep
                measurements.append((step.qubit_index, step.clbit_index))
        has_measurement = len(measurements) > 0
        rng = np.random.default_rng(seed)

        counts = None
        statevector = None
        available = set()

        # Decide statevector delivery.
        want_sv = config.statevector
        if want_sv is None:
            want_sv = not has_measurement

        collapsed_index = None
        if want_sv and has_measurement:
            # Only reached for shots == 1 (validated). Collapse on measured qubits;
            # the engine's internal state becomes the projected statevector and the
            # flat outcome index feeds counts directly.
            measured_qubits = [q for q, _c in measurements]
            collapsed_index = engine.collapse(measured_qubits, rng)

        # Counts.
        effective_counts = config.counts if config.counts is not None else has_measurement
        if effective_counts:
            if has_measurement:
                if collapsed_index is not None:
                    indices = np.array([collapsed_index], dtype=int)
                else:
                    indices = engine.sample_indices(shots, rng)
            else:
                indices = np.zeros(shots, dtype=int)
            counts = build_counts(indices, layout.n_clbits, measurements)
            available.add("counts")

        # Statevector.
        if want_sv:
            statevector = engine.export_state()
            available.add("statevector")

        # NoMeasurementWarning: counts produced, some clbit never written, no state.
        if effective_counts and "statevector" not in available:
            written = {c for _q, c in measurements}
            if any(c not in written for c in range(layout.n_clbits)):
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

    @staticmethod
    def _resolve_program(
        program: Program,
        impl_map: MatrixImplementationMap,
        layout: ResourceLayout,
    ) -> list[ResolvedStep]:
        """Lower a validated program into an ordered execution plan.

        Assumes validation has already run: every gate has a registered rule,
        conditions are rejected, and measurements are terminal. Register
        references are flattened to engine-ready integer indices here.
        """
        plan: list[ResolvedStep] = []
        for step in program.operations:
            if isinstance(step, AppliedOperation):
                matrix = impl_map.get(type(step.operation))(step)
                target_indices = tuple(layout.qubit_index(t) for t in step.targets)
                plan.append(
                    ApplyMatrixStep(matrix=matrix, target_indices=target_indices)
                )
            elif isinstance(step, Measurement):
                plan.append(
                    MeasurementStep(
                        qubit_index=layout.qubit_index(step.qreg),
                        clbit_index=layout.clbit_index(step.clreg),
                    )
                )
        return plan
