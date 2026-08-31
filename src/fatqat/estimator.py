"""Evaluate qubit observables from a backend-produced final state.

An estimator evolves an unmeasured program once and evaluates every requested
observable on the returned state. Exact evaluation is the default; positive
``shots`` values sample each observable term from that same state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Callable

import numpy as np

from ._parameter_binding import (
    _normalize_parameter_batch,
    _raise_for_unbound_parameters,
)
from .errors import BackendExecutionError, BackendValidationError
from .job import Job
from .observable import Observable
from .operations import Measurement
from .parameters import Parameter, ParameterVector
from .program import Program
from .result import Result
from .simulator._engine.expectation import (
    expectation_density_matrix,
    expectation_statevector,
    squared_factors,
)


class Estimator:
    """Evaluate observables with a state-producing backend.

    Configure the method, runtime, and noise model on the backend before
    constructing the estimator. The backend must return either a statevector
    or a density matrix.

    Args:
        backend: Constructed state-producing backend.

    Raises:
        BackendValidationError: If ``backend.method`` names a representation
            other than ``"statevector"`` or ``"density_matrix"``.

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

        All observables use one evolution of ``program``. Program, observable,
        and shot validation errors raise before a job is returned. If execution
        fails, ``job.result()`` raises the failure. If the backend completes
        without a final state, ``job.result()`` raises
        ``BackendExecutionError``.

        Args:
            program: Fully bound, measurement-free program containing only
                dimension-2 quantum registers.
            observables: One ``Observable`` or a non-empty list or tuple of
                them. Every observable must span all program qubits. A list or
                tuple preserves its order in the returned arrays.
            shots: Non-negative integer. ``0`` computes exact values. A
                positive value draws this many independent samples for each
                term and reports the resulting statistical standard error.
            simulation_config: Backend options for this evolution, or
                ``None``. The dictionary is forwarded unchanged. Its ``seed``
                value, when present, also seeds estimator sampling.

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
        _validate_program(program, observable_list)
        _raise_for_unbound_parameters(program._instructions)

        try:
            values, deviations = self._evaluate(
                program, observable_list, shots, simulation_config
            )
        except BackendValidationError:
            # A validation failure is the caller's to fix, so it raises rather
            # than being packaged - whether it came from the checks above or
            # from the backend's own validation during the run.
            raise
        except Exception as exc:  # execution-stage failure
            return Job(status="ERROR", error=exc)

        def shape(entries: list[float]) -> Any:
            return np.asarray(entries) if is_sequence else entries[0]

        return Job(
            status="DONE",
            result=Result(
                data={"expectation": shape(values), "std": shape(deviations)},
                metadata={
                    "shots": shots,
                    "backend_name": type(self._backend).__name__,
                    "estimator_name": type(self).__name__,
                    "num_observables": len(observable_list),
                },
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
            except BaseException as exc:
                return Job(status="ERROR", error=exc)
        return Job(status="DONE", result=results)

    def _evaluate(
        self,
        program: Program,
        observables: list[Observable],
        shots: int,
        simulation_config: dict[str, Any] | None,
    ) -> tuple[list[float], list[float]]:
        """Evolve once, then read every observable off the same final state."""
        try:
            result = self._backend.run(
                program,
                shots=0,
                simulation_config=simulation_config,
                result_config={"counts": False, "final_state": True},
            ).result()
        except BackendValidationError as exc:
            # The backend already refuses to export a single final state when
            # the run is stochastic - reset or channel noise under statevector
            # semantics sample one branch per shot. Deciding that here would
            # mean re-deriving the backend's own answer from its private
            # attributes, and doing it worse: the backend knows whether a
            # registered channel actually landed on any operation in *this*
            # program, which a look at the noise model alone cannot tell.
            # Only that specific refusal gets the stochastic explanation;
            # every other validation failure (unsupported operation, bad
            # config key, ...) propagates with its subtype and message intact.
            if "is only supported for shots == 1" not in str(exc):
                raise
            raise BackendValidationError(
                f"no single final state is available to evaluate: {exc}. An "
                "expectation value needs a well-defined final state, so a "
                "stochastic run has none to read - or to sample from. Use "
                "method='density_matrix', where reset and channel noise are "
                "exact maps"
            ) from exc

        # The result declares its own representation, so the kernel is chosen
        # from what came back rather than predicted from the backend.
        if "statevector" in result.available_data:
            representation = "statevector"
            state, kernel = result.get_statevector(), expectation_statevector
        elif "density_matrix" in result.available_data:
            representation = "density_matrix"
            state, kernel = result.get_density_matrix(), expectation_density_matrix
        else:
            raise BackendExecutionError(
                "estimator backend returned no final state; expected a "
                "statevector or density matrix"
            )

        logical_dimension = 2 ** _program_width(program)
        expected_shape = (
            (logical_dimension,)
            if representation == "statevector"
            else (logical_dimension, logical_dimension)
        )
        if state.shape != expected_shape:
            raise BackendValidationError(
                f"estimator backend returned {representation} shape {state.shape}; "
                f"expected logical-qubit shape {expected_shape}"
            )

        if shots == 0:
            return [kernel(state, o.terms) for o in observables], [0.0] * len(
                observables
            )

        seed = (simulation_config or {}).get("seed")
        generator = np.random.default_rng(seed)
        sampled = [
            _sample(state, kernel, o.terms, shots, generator) for o in observables
        ]
        return [value for value, _ in sampled], [error for _, error in sampled]


Kernel = Callable[[np.ndarray, tuple], float]


def _outcome_probabilities(mean: float, second_moment: float) -> np.ndarray:
    """Probabilities of the outcomes ``(+1, -1, 0)`` for one term.

    A term is a product of commuting single-qubit factors, so its eigenvalues
    are products of local ones: ``+-1`` from each Pauli, ``0``/``1`` from each
    projector. Every eigenvalue therefore lies in ``{0, +1, -1}``, and one
    measurement of the term yields one of exactly three values. Two moments pin
    that whole distribution::

        P(0)  = 1 - <T**2>            the projectors rejected the state
        P(+1) = (<T**2> + <T>) / 2    since <T> = P(+1) - P(-1)
        P(-1) = (<T**2> - <T>) / 2    and P(+1) + P(-1) = <T**2>

    For a pure Pauli term ``<T**2> = 1``, so ``P(0) = 0`` and this reduces to
    the familiar ``P(+1) = (1 + <T>) / 2``.

    Rounding can push a probability a few ulp below zero (when ``|<T>|`` sits at
    ``<T**2>``, for instance), so the result is clipped and renormalized.
    """
    probabilities = np.array(
        [
            (second_moment + mean) / 2.0,
            (second_moment - mean) / 2.0,
            1.0 - second_moment,
        ]
    )
    np.clip(probabilities, 0.0, None, out=probabilities)
    return probabilities / probabilities.sum()


def _sample(
    state: np.ndarray,
    kernel: Kernel,
    terms: tuple[tuple[float, tuple[tuple[int, str], ...]], ...],
    shots: int,
    generator: np.random.Generator,
) -> tuple[float, float]:
    """Draw ``shots`` samples of an observable; return ``(mean, std)``.

    Each term is sampled independently and the results are combined by the
    observable's coefficients. Drawing the outcome *counts* from a multinomial
    is equivalent to drawing ``shots`` individual outcomes and averaging them,
    but costs the same whether ``shots`` is 100 or 10**9.

    The reported standard error is analytic - ``sqrt(sum_k c_k**2 Var(T_k) /
    shots)`` - rather than the spread of these particular draws, so it reports
    the precision of the request instead of adding a second layer of noise.
    """
    total = 0.0
    variance = 0.0
    for coefficient, factors in terms:
        if coefficient == 0.0:
            continue
        mean = kernel(state, ((1.0, factors),))
        projectors = squared_factors(factors)
        # No projector means T**2 = I exactly; skip the second pass.
        second_moment = kernel(state, ((1.0, projectors),)) if projectors else 1.0

        plus, minus, _ = generator.multinomial(
            shots, _outcome_probabilities(mean, second_moment)
        )
        total += coefficient * (plus - minus) / shots
        # max(..., 0) guards a variance driven slightly negative by rounding.
        variance += coefficient**2 * max(second_moment - mean**2, 0.0) / shots
    return total, math.sqrt(variance)


_STATE_METHODS = ("statevector", "density_matrix")


def _validate_backend(backend: Any) -> None:
    """Reject a backend whose method produces an operator rather than a state.

    An expectation value is ``<psi|O|psi>`` or ``Tr(rho O)``: it contracts an
    observable against a *state*. ``method="unitary"`` and ``method="superop"``
    produce the program's *map* instead, and a map has no expectation value
    until an input state is named. ``U|0...0>`` would be one, but assuming that
    silently would answer a question the caller never asked, so the estimator
    declines rather than guessing.

    Checked at construction, using the backend's public ``method``, so the
    mismatch surfaces at ``fq.Estimator(backend)`` - before any program is
    lowered or evolved. A backend that predates the ``method`` property is
    left alone here and validated by whatever its run produces.
    """
    method = getattr(backend, "method", None)
    if method is None or method in _STATE_METHODS:
        return
    raise BackendValidationError(
        f"an estimator needs a backend that produces a state, but "
        f"method={method!r} produces the program's operator. An expectation "
        "value contracts an observable against a state, and an operator has "
        "none until an input state is named. Use method='statevector' or "
        "method='density_matrix'"
    )


def _normalize_observables(
    observables: Observable | list[Observable] | tuple[Observable, ...],
) -> tuple[list[Observable], bool]:
    """Return ``(list, was_a_sequence)`` so the output can mirror the input."""
    if isinstance(observables, Observable):
        return [observables], False
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
    if not isinstance(shots, int) or isinstance(shots, bool):
        raise BackendValidationError(f"shots must be an int, got {shots!r}")
    if shots < 0:
        raise BackendValidationError(f"shots must be >= 0, got {shots}")


def _validate_program(
    program: Program,
    observables: list[Observable],
) -> None:
    """Reject what the estimator can decide from the public program alone.

    Deliberately does not judge whether the run is deterministic. That is the
    backend's own question, it already answers it when a final state is
    requested, and its answer is the better one - see ``_evaluate``.
    """
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
