"""Estimator: expectation values of observables on a simulator backend.

An `Estimator` wraps an already-constructed backend and reports
``<psi|O|psi>`` (or ``Tr(rho O)``) for one or more
:py:class:`~fatqat.Observable` values. The backend owns the method, runtime and
noise model; the estimator adds only the observable step, so the same backend
can serve counts through ``backend.run`` and expectation values here.

The program is evolved **once** per ``run()`` call and every observable is evaluated
against that same state. This is the structural advantage a simulator has over
hardware: hardware must fan a multi-basis observable out into several circuits
(one per commuting group, each with its own basis-rotation gates), while a
simulator holds the final state and can read any observable off it directly.
Costs scale as one evolution plus a cheap pass per term, rather than as one
circuit execution per measurement basis.

Because the expectation value is read from the final state, the program must
not measure: a measurement collapses the state, and "the expectation value of
the final state" then has no single meaning. Qiskit's estimators reject
measured circuits for the same reason.

The estimator reaches the backend only through its public surface -
``backend.run`` and the ``Result`` it returns. In particular it never asks the
backend which state representation it will produce, nor whether the run will be
deterministic: the returned ``Result`` declares its own representation through
``available_data``, and the backend already refuses to export a final state
from a stochastic run. Re-deriving either would duplicate the backend's own
answer from its internals, and get it wrong at the edges.

With ``shots > 0`` the estimator reproduces the statistical error of a
finite-shot experiment by drawing real samples from each term's eigenvalue
distribution - not by adding analytic Gaussian noise to the exact value. See
:py:func:`_outcome_probabilities` for why that distribution is fully determined
by two numbers per term.
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
from .errors import BackendValidationError
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
    """Expectation values of observables, evaluated on a backend.

    Args:
        backend: A constructed backend, e.g.
            ``fq.simulator.Simulator(method="DM", noise=noise)``. Its method,
            runtime and noise model are used as-is; the estimator never
            overrides them. The method must produce a *state* -
            ``"statevector"`` or ``"density_matrix"``; see
            :py:func:`_validate_backend`.

    Raises:
        BackendValidationError: If the backend produces an operator rather
            than a state.

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

        Args:
            program: Program to evolve. Must not contain a measurement.
            observables: A single :py:class:`~fatqat.Observable` or a sequence
                of them. All are evaluated against one evolution.
            shots: ``0`` (the default) computes the expectation value exactly
                from the final state. A positive value samples, reproducing the
                statistical error of a finite-shot experiment. Note the default
                differs from ``Simulator.run``, whose ``shots`` defaults to
                1024 - an estimator's usual request is the exact value.
            simulation_config: Optional per-run backend options, forwarded
                unchanged (e.g. ``{"seed": 7}``). A ``seed`` also seeds the
                estimator's own sampling, so a seeded ``shots > 0`` run
                reproduces.

        Returns:
            A completed ``Job``. ``result().get_expectation()`` returns a float
            for a single observable and an array for a sequence, mirroring the
            input shape. ``result().get_std()`` returns the matching standard
            error, which is ``0`` for an exact run.

        Raises:
            TypeError: If ``observables`` is not an ``Observable`` or a
                sequence containing only ``Observable`` values.
            BackendValidationError: If the program measures, if the backend's
                execution is not deterministic, if an observable's width does
                not match the program, if the program uses non-qubit registers,
                or if ``shots`` is negative.
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
        """Bind and evaluate every row of one complete parameter batch.

        Binding shapes match :meth:`fatqat.simulator.Simulator.run_sweep`.
        Each list element is the ordinary result of one :meth:`run` call, so a
        single observable remains scalar and a sequence remains array-shaped.

        Args:
            program: Parameterized template program.
            observables: One observable or a sequence evaluated for every row.
            bindings: Complete object-keyed parameter batch.
            shots: Exact or sampled Estimator mode forwarded to every row.
            simulation_config: Backend and sampling options forwarded
                unchanged. An explicit seed is reused for every row, so
                sampled row errors are correlated; see
                :doc:`../guide/parameters-and-sweeps`.

        Returns:
            An eager job carrying an ordered list of row results. If a point
            job fails, ``result()`` re-raises that error and no partial result
            list is exposed.

        Raises:
            TypeError: If ``bindings`` is not an object-keyed mapping or a
                batch contains values other than built-in ``int``/``float``
                or NumPy integer/floating scalars, or if ``observables`` is
                not an ``Observable`` or a sequence containing only
                ``Observable`` values.
            ValueError: If the program is not parameterized, assignments are
                missing or duplicated, or batch ranks and lengths disagree.
            BackendValidationError: If the observables, bound program, shots,
                or backend execution mode fail normal Estimator validation.

        Examples:
            Each result keeps the ordinary single-observable scalar shape:

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
            state, kernel = result.get_statevector(), expectation_statevector
        else:
            state, kernel = result.get_density_matrix(), expectation_density_matrix

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
