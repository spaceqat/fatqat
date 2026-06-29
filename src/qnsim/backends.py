"""Qubit statevector backend: validate, execute, assemble Result, return Job."""

from __future__ import annotations

import numpy as np

from . import engine
from .errors import BackendValidationError, UnsupportedOperationError
from .implementation import default_implementation_map
from .job import Job
from .layout import ResourceLayout
from .program import AppliedOperation, Measurement, Program
from .result import Result, ResultConfig, build_counts


class StateVectorBackend:
    def __init__(self, *, seed=None):
        self._seed = seed
        self._impl_map = default_implementation_map()

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

    # --- execution ---
    def _execute(self, program, config, shots, layout) -> Result:
        state = self._evolve(program, layout)
        measurements = self._measurement_map(program, layout)
        has_measurement = len(measurements) > 0
        effective_counts = config.counts if config.counts is not None else has_measurement

        counts = None
        available = set()
        if effective_counts:
            rng = np.random.default_rng(self._seed)
            if has_measurement:
                indices = engine.sample_indices(state, shots, rng)
            else:
                indices = np.zeros(shots, dtype=int)  # nothing measured -> all-zero clbits
            counts = build_counts(indices, layout.n_clbits, measurements)
            available.add("counts")

        return Result(counts=counts, available=frozenset(available))

    def _evolve(self, program, layout) -> np.ndarray:
        state = engine.zero_state(layout.n_qubits)
        for step in program.operations:
            if isinstance(step, AppliedOperation):
                rule = self._impl_map.get(type(step.operation))
                matrix = rule(step)
                targets = tuple(layout.qubit_index(t) for t in step.targets)
                state = engine.apply(state, matrix, targets, layout.n_qubits)
        return state

    @staticmethod
    def _measurement_map(program, layout):
        out = []
        for step in program.operations:
            if isinstance(step, Measurement):
                out.append(
                    (layout.qubit_index(step.qreg), layout.clbit_index(step.clreg))
                )
        return out
