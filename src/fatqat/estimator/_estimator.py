"""Evaluate observables through backend-owned expectation execution.

An estimator validates the backend-neutral request, then delegates exact or
sampled term execution to the selected backend. Noise, trajectories, readout,
resource mapping, and execution policy stay with the backend that owns them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .._backends.backend_utils import _resolve_execution_config
from .._parameter_binding import (
    _normalize_parameter_batch,
    _raise_for_unbound_parameters,
)
from ..errors import BackendValidationError
from ..job import Job
from ..observable import Observable
from ..operations import Measurement
from ..parameters import Parameter, ParameterVector
from ..program import Program
from ..result import Result


class Estimator:
    """Calculate expectation values after running a program.

    Choose the method, runtime, and noise on the backend. Supported choices
    include gate-level simulators with a state method, ``LOSimulator`` with a
    state method or ``"fixed_photon_state"``, and ``Atom2LevelEmulator`` with
    a state method. The chosen backend and method determine which observable
    operators and noise models can be used. A backend that implements only
    the ``Backend.run()`` protocol does not necessarily support estimation.

    Args:
        backend: Configured backend that supports expectation evaluation.

    Raises:
        BackendValidationError: If the backend does not implement expectation
            evaluation.

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
        initial_state: Any = None,
        execution_config: dict[str, Any] | None = None,
        simulation_config: dict[str, Any] | None = None,
    ) -> Job[Result]:
        """Evaluate one or more observables on a program.

        Program, observable, shot, and backend capability validation errors
        raise before a job is returned. Execution errors follow the backend's
        error-handling policy.

        Args:
            program: Fully bound, measurement-free program.
            observables: One ``Observable`` or a non-empty list or tuple of
                them. Every local operator must name a program subsystem: Pauli
                operators need a qubit, and a projector level must be below its
                subsystem's dimension. Unnamed subsystems carry the identity.
                A list or tuple preserves its order in the returned arrays.
            shots: Nonnegative Python ``int`` other than ``bool``.
                ``0`` computes exact values. A positive value draws this many
                independent samples for each nonconstant term. One sample
                makes that term's standard error undefined (``nan``).
                Constant terms contribute zero standard error.
            initial_state: Starting state for every shot, or ``None`` for the
                ground state. Available only for simulators; see
                :meth:`fatqat.simulator.Simulator.run` for the accepted forms.
            execution_config: Options accepted by the backend's ``run()``
                method, or ``None`` for defaults. In particular, set ``"seed"``
                to reproduce sampled results.
            simulation_config: Deprecated alias for ``execution_config``,
                to be removed before fatqat 0.1.0.

        Returns:
            A completed ``Job``. A successful job contains a ``Result``; a
            single observable produces scalar values, while a list or tuple
            produces one-dimensional NumPy arrays in input order. Read them
            with ``get_expectation()`` and ``get_standard_error()``. Exact runs
            report zero standard error. The result contains no final state.
            Result metadata includes ``backend_name``, ``method``,
            ``runtime``, and ``shots``. Pulse results also include
            ``runtime_details["solver"]`` when a solver was used.

        Raises:
            TypeError: If ``observables`` is not an ``Observable`` or a
                supported collection containing only ``Observable`` values,
                if ``execution_config`` is neither a dictionary nor ``None``,
                or if it and its deprecated alias are both supplied.
            BackendValidationError: If no observables are supplied; ``shots``
                is not a non-negative integer; the program is measured or
                unbound; an operator names a subsystem the program lacks, a Pauli
                operator targets a non-qubit, or a projector level is not below
                its subsystem's dimension; or the backend raises this error
                while preparing its native expectation execution.
        """
        execution_config = _resolve_execution_config(
            execution_config, simulation_config
        )
        observable_list, is_sequence = _normalize_observables(observables)
        _validate_shots(shots)
        _validate_execution_config(execution_config)
        _validate_program(program)
        _raise_for_unbound_parameters(program._instructions)
        job = self._evaluate(
            program,
            observable_list,
            is_sequence,
            shots=shots,
            initial_state=initial_state,
            execution_config=execution_config,
        )
        if job.status != "DONE":
            return job
        (result,) = job.result()
        return Job(status="DONE", result=result)

    def run_sweep(
        self,
        program: Program,
        observables: Observable | list[Observable] | tuple[Observable, ...],
        bindings: Mapping[Parameter | ParameterVector, object],
        *,
        shots: int = 0,
        initial_state: Any = None,
        execution_config: dict[str, Any] | None = None,
        simulation_config: dict[str, Any] | None = None,
    ) -> Job[list[Result]]:
        """Evaluate a complete parameter batch in input order.

        Each binding row produces an estimator result. Validation errors raise
        directly. Execution errors follow the backend's error policy. A failed
        row never exposes a partial result list.

        Args:
            program: Parameterized program without measurement operations.
                Supply its free parameters through ``bindings``.
            observables: One observable or a sequence evaluated for every row.
            bindings: Complete object-keyed batch. A ``Parameter`` maps to a
                rank-1 batch. A length-M ``ParameterVector`` maps to a rank-2
                batch with shape ``(N, M)``. All batches must have the same
                positive leading length.
            shots: Non-negative integer. ``0`` computes exact values; a
                positive value draws this many samples per observable term.
            initial_state: Starting state for every shot in every row, as in
                :meth:`run`. Available only for simulators.
            execution_config: Backend options reused for every row, as in
                :meth:`run`. Reusing a seed can correlate sampled row errors.
            simulation_config: Deprecated alias for ``execution_config``,
                to be removed before fatqat 0.1.0.

        Returns:
            A completed ``Job`` containing one ``Result`` per row, in batch
            order. Each result has the same shape as :meth:`run`. An execution
            failure exposes no partial list.

        Raises:
            TypeError: If a mapping key, batch container, scalar value, or
                observable has an unsupported type, or if
                ``execution_config`` and its deprecated alias are both
                supplied.
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
        execution_config = _resolve_execution_config(
            execution_config, simulation_config
        )
        columns = _normalize_parameter_batch(program._instructions, bindings)
        observable_list, is_sequence = _normalize_observables(observables)
        _validate_shots(shots)
        _validate_execution_config(execution_config)
        _validate_program(program)
        return self._evaluate(
            program,
            observable_list,
            is_sequence,
            shots=shots,
            initial_state=initial_state,
            execution_config=execution_config,
            parameter_values=columns,
        )

    def _evaluate(
        self,
        program: Program,
        observable_list: list[Observable],
        is_sequence: bool,
        *,
        shots: int,
        initial_state: Any,
        execution_config: dict[str, Any] | None,
        parameter_values: Mapping[Parameter, np.ndarray] | None = None,
    ) -> Job[list[Result]]:
        """Run the backend's expectation hook once and shape each entry's result.

        A failed backend job keeps its exception object; no partial list is
        exposed.
        """
        internal_job = self._backend._run_expectation(
            program,
            tuple(observable_list),
            shots=shots,
            initial_state=initial_state,
            execution_config=execution_config,
            parameter_values=parameter_values,
        )
        try:
            executions = internal_job.result()
        except Exception as exc:
            return Job(status="ERROR", error=exc)

        def shape(entries: tuple[float, ...]) -> Any:
            return np.asarray(entries) if is_sequence else entries[0]

        results = []
        for execution in executions:
            metadata = {"method": execution.method, "runtime": execution.runtime}
            if execution.solver is not None:
                metadata["runtime_details"] = {"solver": execution.solver}
            metadata.update(
                {"backend_name": type(self._backend).__name__, "shots": shots}
            )
            results.append(
                Result(
                    data={
                        "expectation": shape(execution.values),
                        "standard_error": shape(execution.standard_errors),
                    },
                    metadata=metadata,
                )
            )
        return Job(status="DONE", result=results)


def _validate_backend(backend: Any) -> None:
    """Require the single backend-owned expectation execution seam."""
    if callable(getattr(backend, "_run_expectation", None)):
        return
    raise BackendValidationError(
        "an estimator backend must provide a callable '_run_expectation' hook"
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


def _validate_shots(shots: int) -> None:
    if type(shots) is not int:
        raise BackendValidationError(f"shots must be an int, got {shots!r}")
    if shots < 0:
        raise BackendValidationError(f"shots must be >= 0, got {shots}")


def _validate_execution_config(
    execution_config: dict[str, Any] | None,
) -> None:
    if execution_config is not None and not isinstance(execution_config, dict):
        raise TypeError(
            "execution_config must be a dict or None, got "
            f"{type(execution_config)!r}"
        )


def _validate_program(program: Program) -> None:
    """Reject what the estimator can decide from the public program alone.

    Subsystem dimensions may only be known once a backend resolves them, so
    each backend's expectation hook checks observable operators against the
    program, along with its own execution and capability checks.
    """
    if not isinstance(program, Program):
        raise BackendValidationError("program must be a Program")

    if any(isinstance(step, Measurement) for step in program._instructions):
        raise BackendValidationError(
            "a program with a measurement has no well-defined expectation "
            "value: the measurement collapses the state. Remove the "
            "measurement, or use backend.run for counts"
        )
