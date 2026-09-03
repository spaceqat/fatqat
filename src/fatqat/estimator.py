"""Evaluate qubit observables through backend-owned expectation execution.

An estimator validates the backend-neutral request, then delegates exact or
sampled term execution to the selected backend. Noise, trajectories, readout,
resource mapping, and execution policy stay with the backend that owns them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ._parameter_binding import (
    _normalize_parameter_batch,
    _raise_for_unbound_parameters,
)
from .errors import BackendValidationError
from .job import Job
from .observable import Observable
from .operations import Measurement
from .parameters import Parameter, ParameterVector
from .program import Program
from .result import Result


class Estimator:
    """Evaluate observables with backend-owned expectation execution.

    Configure the method, runtime, noise model, and physical model on the
    backend before constructing the estimator. Built-in matrix simulators and
    pulse emulators implement the private execution seam used here.

    Args:
        backend: Constructed backend with a callable expectation hook.

    Raises:
        BackendValidationError: If the backend lacks the required hook.

    Examples:
        >>> import fatqat as fq
        >>> import fatqat.operations as ops
        >>> program = fq.Program(2)
        >>> program.add(ops.H, 0)
        >>> program.add(ops.CX, (0, 1))
        >>> estimator = fq.Estimator(fq.simulator.Simulator(method="SV"))
        >>> result = estimator.run(program, fq.Observable([("ZZ", 1.0)])).result()
        >>> round(float(result.get_expectation()), 10)
        1.0
    """

    def __init__(self, backend: Any) -> None:
        _validate_backend(backend)
        self._backend = backend

    def run(
        self,
        program: Program,
        observables: Observable | list[Observable] | tuple[Observable, ...],
        *,
        shots: int = 0,
        simulation_config: dict[str, Any] | None = None,
    ) -> Job[Result]:
        """Evaluate one or more observables on a program.

        Program, observable, shot, and backend capability validation errors
        raise before a job is returned. Execution failures stay on the
        returned job with the backend's original exception object.

        Args:
            program: Fully bound, measurement-free program containing only
                dimension-2 quantum registers.
            observables: One ``Observable`` or a non-empty list or tuple of
                them. Every observable must span all program qubits. A list or
                tuple preserves its order in the returned arrays.
            shots: Non-negative integer. ``0`` computes exact values. A
                positive value draws this many independent samples for each
                term and reports the resulting statistical standard error.
            simulation_config: Backend options for this execution, or
                ``None``. The dictionary is forwarded without interpreting
                its keys or values.

        Returns:
            A completed ``Job``. A successful job contains a ``Result``; a
            single observable produces scalar expectation and standard-error
            values, while a list or tuple produces one-dimensional arrays.
            Exact runs report zero standard error.

        Raises:
            TypeError: If ``observables`` is not an ``Observable`` or a
                supported collection containing only ``Observable`` values.
            BackendValidationError: If no observables are supplied; ``shots``
                is not a non-negative integer; the program is measured,
                unbound, or not qubit-only; an observable has the wrong width;
                or the backend raises this error for the required final-state
                request.
        """
        observable_list, is_sequence = _normalize_observables(observables)
        _validate_shots(shots)
        _validate_simulation_config(simulation_config)
        _validate_program(program, observable_list)
        _raise_for_unbound_parameters(program._instructions)

        internal_job = self._backend._run_expectation(
            program,
            tuple(observable_list),
            shots=shots,
            simulation_config=simulation_config,
        )
        try:
            execution = internal_job.result()
        except BaseException as exc:
            return Job(status="ERROR", error=exc)

        def shape(entries: tuple[float, ...]) -> Any:
            return np.asarray(entries) if is_sequence else entries[0]

        metadata = {
            key: execution.metadata[key]
            for key in ("method", "runtime")
            if key in execution.metadata
        }
        runtime_details = execution.metadata.get("runtime_details")
        if isinstance(runtime_details, Mapping) and "solver" in runtime_details:
            metadata["runtime_details"] = {"solver": runtime_details["solver"]}
        metadata.update(
            {
                "backend_name": type(self._backend).__name__,
                "shots": shots,
            }
        )
        return Job(
            status="DONE",
            result=Result(
                data={
                    "expectation": shape(execution.values),
                    "standard_error": shape(execution.standard_errors),
                },
                metadata=metadata,
            ),
        )

    def run_sweep(
        self,
        program: Program,
        observables: Observable | list[Observable] | tuple[Observable, ...],
        bindings: Mapping[Parameter | ParameterVector, object],
        *,
        shots: int = 0,
        simulation_config: dict[str, Any] | None = None,
    ) -> Job[list[Result]]:
        """Evaluate a complete parameter batch in input order.

        Each binding row produces an estimator result. Validation errors raise
        directly. Other row failures make the returned job fail without
        exposing a partial result list.

        Args:
            program: Parameterized template program.
            observables: One observable or a sequence evaluated for every row.
            bindings: Complete object-keyed batch. A ``Parameter`` maps to a
                rank-1 batch. A length-M ``ParameterVector`` maps to a rank-2
                batch with shape ``(N, M)``. All batches must have the same
                positive leading length.
            shots: Non-negative integer. ``0`` computes exact values; a
                positive value draws this many samples per observable term.
            simulation_config: Backend and sampling options reused for every
                row. Reusing a seed can correlate sampled row errors.

        Returns:
            A completed ``Job`` containing one ``Result`` per row, in batch
            order. Each result has the same shape as :meth:`run`. An execution
            failure exposes no partial list.

        Raises:
            TypeError: If a mapping key, batch container, scalar value, or
                observable has an unsupported type.
            ValueError: If the program is not parameterized; assignments are
                missing, foreign, or duplicated; or batch ranks, widths, or
                lengths disagree.
            BackendValidationError: If a bound row, the observables, ``shots``,
                or the backend fails normal estimator validation.

        Examples:
            A single observable produces one scalar per row:

            >>> import fatqat as fq
            >>> import fatqat.operations as ops
            >>> theta = fq.Parameter("theta")
            >>> program = fq.Program(1)
            >>> program.add(ops.RY(theta), 0)
            >>> estimator = fq.Estimator(fq.simulator.Simulator("SV"))
            >>> observable = fq.Observable([("Z", 1.0)])
            >>> results = estimator.run_sweep(
            ...     program, observable, {theta: [0.0, 1.0]}
            ... ).result()
            >>> len(results)
            2
            >>> [round(result.get_expectation(), 6) for result in results]
            [1.0, 0.540302]
        """
        rows = _normalize_parameter_batch(program._instructions, bindings)
        results: list[Result] = []
        for row in rows:
            bound = program._assign_normalized_parameters(row)
            point_job = self.run(
                bound,
                observables,
                shots=shots,
                simulation_config=simulation_config,
            )
            try:
                results.append(point_job.result())
            except Exception as exc:
                return Job(status="ERROR", error=exc)
        return Job(status="DONE", result=results)


def _validate_backend(backend: Any) -> None:
    """Require the single backend-owned expectation execution seam."""
    if callable(getattr(backend, "_run_expectation", None)):
        return
    raise BackendValidationError(
        "an estimator backend must provide a callable expectation-execution hook"
    )


def _normalize_observables(
    observables: Observable | list[Observable] | tuple[Observable, ...],
) -> tuple[list[Observable], bool]:
    """Return ``(list, was_a_sequence)`` so the output can mirror the input."""
    if isinstance(observables, Observable):
        return [observables], False
    if not isinstance(observables, (list, tuple)):
        raise TypeError(
            "observables must be an Observable or a list or tuple of Observable "
            f"values, got {type(observables)!r}"
        )
    observable_list = list(observables)
    if not observable_list:
        raise BackendValidationError("no observables given")
    for entry in observable_list:
        if not isinstance(entry, Observable):
            raise TypeError(f"expected an Observable, got {entry!r}")
    return observable_list, True


def _program_width(program: Program) -> int:
    """Total number of quantum subsystems in the program."""
    return sum(register.size for register in program.quantum_registers)


def _validate_shots(shots: int) -> None:
    if type(shots) is not int:
        raise BackendValidationError(f"shots must be an int, got {shots!r}")
    if shots < 0:
        raise BackendValidationError(f"shots must be >= 0, got {shots}")


def _validate_simulation_config(
    simulation_config: dict[str, Any] | None,
) -> None:
    if simulation_config is not None and not isinstance(simulation_config, dict):
        raise TypeError(
            "simulation_config must be a dict or None, got "
            f"{type(simulation_config)!r}"
        )


def _validate_program(
    program: Program,
    observables: list[Observable],
) -> None:
    """Reject what the estimator can decide from the public program alone.

    Backend-specific execution and capability checks remain with the backend's
    expectation hook.
    """
    if not isinstance(program, Program):
        raise BackendValidationError("program must be a Program")
    for register in program.quantum_registers:
        if register.dim != 2:
            raise BackendValidationError(
                "observables are defined on qubits only; register "
                f"{register.name!r} has dim={register.dim}"
            )

    if any(isinstance(step, Measurement) for step in program._instructions):
        raise BackendValidationError(
            "a program with a measurement has no well-defined expectation "
            "value: the measurement collapses the state. Remove the "
            "measurement, or use backend.run for counts"
        )

    width = _program_width(program)
    for observable in observables:
        if observable.num_qubits != width:
            raise BackendValidationError(
                f"observable is defined on {observable.num_qubits} qubit(s) but "
                f"the program has {width}"
            )
