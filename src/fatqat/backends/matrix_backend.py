"""Shared matrix-family backend skeleton.

`_MatrixBackendBase` carries everything that is identical between the
matrix-family backends: options/result-config normalization, lowering,
validation, execution orchestration, and public `Result` assembly. The
concrete backends (`StateVectorBackend`, `DensityMatrixBackend`) contribute
only declarative class attributes plus their user-facing docstrings.

The per-backend variation points are class attributes, not overridable hooks:

- ``_engine_cls``: engine class constructed once per backend instance.
- ``_result_config_cls`` / ``_request_cls``: the backend's frozen result-config
  and engine-request value objects. The supported ``result_config`` keys are
  derived from ``_result_config_cls``'s dataclass fields.
- ``_state_field``: the backend's native state field name (``"statevector"``
  or ``"density_matrix"``). Drives the result-config flag read, the `Result`
  keyword, the availability name, the metadata echo, and validation wording.
- ``_reset_is_stochastic``: whether reset makes execution stochastic for this
  state representation (`True` for the statevector backend, where reset
  samples a branch; `False` for the density-matrix backend, where reset is a
  deterministic channel).

This class is backend-internal: it is not exported from the package and is
not part of the public API. The backend/engine seam is unchanged: the base
calls ``engine.initialize(system_dims, n_clbits)`` and
``engine.run(plan, shots, seed, request) -> RawResult`` exactly as the
concrete backends did before extraction.
"""

from __future__ import annotations

import warnings
from dataclasses import fields
from math import prod
from typing import Any, ClassVar

from ..errors import (
    BackendValidationError,
    MatrixImplementationError,
    NoMeasurementWarning,
    UnsupportedOperationError,
)
from ..implementation import (
    MatrixImplementation,
    DeviceOperands,
    ImplementationMap,
    default_matrix_implementation_map,
)
from ..job import Job
from ..layout import ResourceLayout
from ..operations import Measurement, Operation, ResetGate
from ..program import AppliedOperation, Program
from ..result import Result, counts_dict_from_arrays
from .backend_utils import (
    _PlanFacts,
    _normalize_dict_options,
    _resolve_condition,
)
from .engine_contract import _EngineConfig
from .steps import ApplyMatrixStep, MeasurementStep, ResetStep, ResolvedStep


class _MatrixBackendBase:
    """Execution skeleton shared by the matrix-family backends.

    Subclasses declare the class attributes documented in the module
    docstring; every method below is state-representation-agnostic.
    """

    _engine_cls: ClassVar[type]
    _result_config_cls: ClassVar[type]
    _request_cls: ClassVar[type]
    _state_field: ClassVar[str]
    _reset_is_stochastic: ClassVar[bool]

    def __init__(
        self,
        options: dict[str, Any] | None = None,
        implementation_map: ImplementationMap | None = None,
    ) -> None:
        """Create a matrix-family backend.

        Args:
            options: Optional execution-strategy options. Supported keys are
                ``max_workers`` and ``parallel_mode``; unknown keys are
                ignored with a warning.
            implementation_map: Optional matrix implementation map. ``None``
                uses ``default_matrix_implementation_map()``. The backend
                copies whatever map it receives.
        """
        config = _normalize_dict_options(
            options,
            {"max_workers", "parallel_mode"},
            _EngineConfig,
            "options",
            "backend",
            backend_name=type(self).__name__,
        )
        if implementation_map is None:
            implementation_map = default_matrix_implementation_map()
        self._impl_map = implementation_map.copy()
        # The engine is constructed once and re-initialized per run so its
        # buffers can be reused. Because it holds per-run state, a single
        # backend instance is NOT safe for concurrent run() calls
        # (single-threaded use only).
        self._engine = self._engine_cls(config)
        self._engine_system: tuple[tuple[int, ...], int] | None = None

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
        result_config: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> Job:
        """Validate, execute, and package one program run.

        Shared skeleton: normalize ``result_config``, resolve the layout,
        lower the program, validate, execute, and wrap the outcome in an
        eager `Job`. See each concrete backend's ``run`` docstring for the
        user-facing result-selection and shot semantics.
        """
        known_keys = {field.name for field in fields(self._result_config_cls)}
        config = _normalize_dict_options(
            result_config,
            known_keys,
            self._result_config_cls,
            "result_config",
            "result_config",
            backend_name=type(self).__name__,
        )
        layout = self.resolve_layout(program)
        plan, facts = self._lower(program, layout)
        self._validate(config, shots, facts)
        try:
            return Job.done(
                self._execute(
                    config,
                    shots,
                    plan,
                    facts,
                    layout.system_dims,
                    layout.classical_dims,
                    layout.n_clbits,
                    seed,
                )
            )
        except Exception as exc:  # execution-stage failure
            return Job.failed(exc)

    # --- validation (raises directly from run) ---
    def _validate(self, config: Any, shots: int, facts: _PlanFacts) -> None:
        """Validate result-config / shots constraints against the lowered program.

        Operation support and dynamic classification were already resolved in
        `_lower`. Stochasticity is representation-dependent: measurement is
        always stochastic; reset only when ``_reset_is_stochastic``.
        """
        request = _resolve_result_request(
            config, facts, self._request_cls, self._state_field, self._reset_is_stochastic
        )
        stochastic = facts.has_measurement or (
            self._reset_is_stochastic and facts.has_reset
        )
        requested_state = getattr(config, self._state_field) is True

        # shots is only checked when the result actually depends on it: counts
        # always sample per shot, and a stochastic state export needs shots==1
        # below. A non-stochastic state-only request ignores shots entirely
        # (see the engine's per-shot path), so any value - including 0 - is fine.
        if (request.counts or (requested_state and stochastic)) and type(shots) is not int:
            raise BackendValidationError(
                f"shots must be an int when requested results depend on it, got {shots!r}"
            )
        if request.counts and shots <= 0:
            raise BackendValidationError(f"counts require shots > 0, got shots={shots}")
        if requested_state and stochastic and shots != 1:
            stochastic_sources = (
                "measurement or reset" if self._reset_is_stochastic else "measurement"
            )
            raise BackendValidationError(
                f"{self._state_field} with {stochastic_sources} is only supported "
                "for shots == 1"
            )

    # --- execution ---
    def _execute(
        self,
        config: Any,
        shots: int,
        plan: list[ResolvedStep],
        facts: _PlanFacts,
        system_dims: tuple[int, ...],
        classical_dims: tuple[int, ...],
        n_clbits: int,
        seed: int | None,
    ) -> Result:
        """Execute a lowered program and assemble the requested result fields."""
        request = _resolve_result_request(
            config, facts, self._request_cls, self._state_field, self._reset_is_stochastic
        )

        system_key = (tuple(system_dims), n_clbits)
        if self._engine_system != system_key:
            self._engine.initialize(system_dims, n_clbits)
            self._engine_system = system_key

        raw = self._engine.run(plan, shots, seed, request)
        counts = None
        state = raw.state
        state_requested = getattr(request, self._state_field)
        available: set[str] = set()
        if request.counts:
            counts = counts_dict_from_arrays(raw.outcome_keys, raw.outcome_counts)
            available.add("counts")
        if state_requested:
            available.add(self._state_field)

        # NoMeasurementWarning: counts produced, some clbit never written, no state.
        if request.counts and self._state_field not in available:
            written = {
                c
                for s in plan
                if isinstance(s, MeasurementStep)
                for c in s.classical_indices
            }
            if any(c not in written for c in range(n_clbits)):
                warnings.warn(
                    "counts contain clbits that were never measured; "
                    "returning zero-filled counts",
                    NoMeasurementWarning,
                    stacklevel=3,
                )

        return Result(
            counts=counts,
            available=frozenset(available),
            classical_dims=classical_dims,
            metadata={
                "shots": shots,
                "backend_name": type(self).__name__,
                "result_config": {
                    "counts": config.counts,
                    self._state_field: getattr(config, self._state_field),
                },
            },
            **{self._state_field: state},
        )

    def _implementation_for(
        self, operation: Operation, device_operands: DeviceOperands
    ) -> MatrixImplementation:
        """Resolve the matrix rule for an operation on a device target key.

        Raises :py:exc:`~fatqat.errors.UnsupportedOperationError` if the operation has no rule at
        all, or if it has rules but none for this target key — the message
        distinguishes the two.
        """
        if not self._impl_map.supports(operation):
            raise UnsupportedOperationError(
                f"{type(operation).__name__} is not supported by this backend"
            )
        rule = self._impl_map.implementation_for(
            operation, device_operands=device_operands
        )
        if rule is None:
            raise UnsupportedOperationError(
                f"{type(operation).__name__} is not supported on device operands {device_operands}"
            )
        return rule

    def _lower(
        self, program: Program, layout: ResourceLayout
    ) -> tuple[list[ResolvedStep], _PlanFacts]:
        """Lower a program into an execution plan and classify it, in one pass.

        Raises :py:exc:`~fatqat.errors.UnsupportedOperationError` for a gate with no matrix rule.
        `Reset` is recognized by type and routed to a `ResetStep`. The pass also
        computes `has_measurement` and `has_reset`.
        """
        plan: list[ResolvedStep] = []
        has_measurement = False
        has_reset = False

        for step in program.operations:
            if isinstance(step, Measurement):
                has_measurement = True
                measured_indices = tuple(layout.subsystem_index(q) for q in step.qreg)
                classical_indices = tuple(layout.clbit_index(c) for c in step.clreg)
                plan.append(
                    MeasurementStep(
                        measured_indices=measured_indices,
                        classical_indices=classical_indices,
                    )
                )
                continue

            if isinstance(step, AppliedOperation):
                target_indices = tuple(layout.subsystem_index(t) for t in step.targets)

                if isinstance(step.operation, ResetGate):
                    has_reset = True
                    cond = _resolve_condition(step.condition, layout)
                    plan.append(
                        ResetStep(reset_indices=target_indices, condition=cond)
                    )
                    continue

                rule = self._implementation_for(step.operation, target_indices)
                try:
                    matrix = rule(step.operation, targets=step.targets)
                except Exception as exc:
                    raise MatrixImplementationError(
                        f"implementation for {type(step.operation).__name__} raised: {exc}"
                    ) from exc

                # Check matrix shape matches target dimensions
                target_dims = tuple(layout.system_dims[i] for i in target_indices)
                expected = prod(target_dims)
                if matrix.shape != (expected, expected):
                    raise BackendValidationError(
                        f"{type(step.operation).__name__} resolved to a "
                        f"{matrix.shape} matrix, incompatible with target "
                        f"dimensions {target_dims} (expected "
                        f"{(expected, expected)})"
                    )

                cond = _resolve_condition(step.condition, layout)
                plan.append(
                    ApplyMatrixStep(
                        matrix=matrix, target_indices=target_indices, condition=cond
                    )
                )

        return (
            plan,
            _PlanFacts(
                has_measurement=has_measurement,
                has_reset=has_reset,
            ),
        )


def _resolve_result_request(
    config: Any,
    facts: _PlanFacts,
    request_cls: type,
    state_field: str,
    reset_is_stochastic: bool,
) -> Any:
    """Resolve default result fields from config and lowered program facts.

    Counts default to measurement presence. The state field defaults to
    non-stochastic execution, where stochasticity is representation-dependent:
    measurement always; reset only when ``reset_is_stochastic`` (statevector
    reset samples a branch; density-matrix reset is a deterministic channel).
    """
    stochastic = facts.has_measurement or (reset_is_stochastic and facts.has_reset)
    counts = config.counts if config.counts is not None else facts.has_measurement
    state = getattr(config, state_field)
    if state is None:
        state = not stochastic
    return request_cls(counts=counts, **{state_field: state})
