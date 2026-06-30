"""Qubit statevector backend: validate, execute, assemble Result, return Job."""

from __future__ import annotations

import warnings

import numpy as np

from .engine import StateVectorEngine
from .errors import BackendValidationError, NoMeasurementWarning, UnsupportedOperationError
from .implementation import MatrixImplementation, default_implementation_map
from .job import Job
from .layout import ResourceLayout
from .program import AppliedOperation, Measurement, Program
from .result import Result, ResultConfig, build_counts


class StateVectorBackend:
    def __init__(self, *, seed=None):
        self._seed = seed
        self._impl_map = default_implementation_map()
        # The engine is constructed once and re-initialized per run so its
        # compiled kernels can be reused. Because it holds per-run state, a
        # single backend instance is NOT safe for concurrent run() calls
        # (single-threaded use only in Phase 1).
        self._engine = StateVectorEngine()

    def resolve_layout(self, program: Program) -> ResourceLayout:
        return ResourceLayout.from_program(program)

    def run(self, program, *, shots: int = 1024, result_config=None) -> Job:
        config = result_config if result_config is not None else ResultConfig()
        layout = self.resolve_layout(program)
        self._validate(program, config, shots, layout)
        try:
            return Job.done(self._execute(program, config, shots, layout))
        except Exception as exc:  # execution-stage failure
            return Job.failed(exc)

    # --- validation (raises directly from run) ---
    def _validate(self, program, config, shots, layout) -> None:
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
        if effective_counts and shots <= 0:
            raise BackendValidationError(
                f"counts require shots > 0, got shots={shots}"
            )
        if config.statevector is True and has_measurement and shots > 1:
            raise BackendValidationError(
                "statevector with measurement is only supported for shots == 1 "
                "in Phase 1"
            )

    # --- execution ---
    def _execute(self, program, config, shots, layout) -> Result:
        self._evolve(program, layout)
        engine = self._engine
        measurements = self._measurement_map(program, layout)
        has_measurement = len(measurements) > 0
        rng = np.random.default_rng(self._seed)

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
            counts=counts, statevector=statevector, available=frozenset(available)
        )

    def _evolve(self, program, layout) -> None:
        """Reset the owned engine and apply each gate as a MatrixImplementation."""
        engine = self._engine
        engine.initialize(layout.n_qubits)
        for step in program.operations:
            if isinstance(step, AppliedOperation):
                rule = self._impl_map.get(type(step.operation))
                matrix = rule(step)
                target_indices = tuple(layout.qubit_index(t) for t in step.targets)
                engine.apply(
                    MatrixImplementation(matrix=matrix, target_indices=target_indices)
                )

    @staticmethod
    def _measurement_map(program, layout):
        out = []
        for step in program.operations:
            if isinstance(step, Measurement):
                out.append(
                    (layout.qubit_index(step.qreg), layout.clbit_index(step.clreg))
                )
        return out
