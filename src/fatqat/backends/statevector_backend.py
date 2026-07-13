"""Statevector backend: validate, execute, assemble Result, return Job."""

from __future__ import annotations

from ..implementation import ImplementationMap
from ..job import Job
from ..program import Program
from ..result import _ResultConfig
from .engine_contract import _ResultRequest
from .matrix_backend import _MatrixBackendBase
from .statevector_engine import StateVectorEngine


class StateVectorBackend(_MatrixBackendBase):
    """Statevector backend for ``fatqat.Program`` execution.

    The backend supports matrix-evolvable gates, grouped measurement,
    feedforward conditions, and reset. Each run is classified into one of two
    execution strategies:

    - Fast path: used when the program has no reset, no classically
      conditioned operations, and no operation that acts on a subsystem after
      that subsystem has been measured. The statevector is evolved once;
      requested counts are then sampled from the resulting measurement
      distribution.
    - Dynamic path: used when the program contains reset, a classical
      condition, or reuse of a measured subsystem. The backend executes one
      shot at a time while tracking the classical register explicitly,
      because later operations may depend on earlier measurement outcomes.

    Backend constructor options affect only dynamic counts execution:

    - ``max_workers``: maximum worker processes for dynamic counts
      parallelism. ``None`` means automatic selection.
    - ``parallel_mode``: one of ``"auto"``, ``"serial"``, ``"multiprocessing"``,
      or ``"loky"``. ``"auto"`` prefers ``loky`` when available and otherwise
      uses ``multiprocessing``. ``"serial"`` disables process-based parallel
      execution.

    A backend instance reuses one engine across runs, so it is efficient for
    repeated single-threaded use but is not safe for concurrent ``run()``
    calls.
    """

    _engine_cls = StateVectorEngine
    _result_config_cls = _ResultConfig
    _request_cls = _ResultRequest
    _state_field = "statevector"
    _reset_is_stochastic = True

    def __init__(
        self,
        options: dict[str, Any] | None = None,
        implementation_map: ImplementationMap | None = None,
    ) -> None:
        """Create a statevector backend.

        Args:
            options: Optional execution-strategy options. Supported keys are
                ``max_workers`` and ``parallel_mode``; unknown keys are
                ignored with a warning. These options only affect the dynamic
                counts path and do not change numerical semantics.
            implementation_map: Optional matrix implementation map controlling
                which operations this backend supports and how their matrices
                are built. ``None`` (the default) uses
                ``default_matrix_implementation_map()``. The backend copies
                whatever map it receives, so mutating the caller's map object
                after construction does not change this backend's behavior.
        """
        super().__init__(options=options, implementation_map=implementation_map)

    def run(
        self,
        program: Program,
        *,
        shots: int = 1024,
        result_config: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> Job:
        """Validate, execute, and package one program run.

        This is the main user-facing execution entry point. It resolves the
        program to the backend's flat layout, chooses an execution strategy,
        runs the circuit, and returns an eager ``Job`` whose ``result()``
        yields a ``Result``.

        Result selection via ``result_config``:

        - ``{"counts": None}``: counts are produced when the program contains
          at least one measurement.
        - ``{"counts": True}``: counts are always requested.
        - ``{"counts": False}``: counts are suppressed, even if the program
          measures subsystems.
        - ``{"statevector": None}``: a statevector is produced only when
          execution is non-stochastic, meaning the program contains no
          measurement and no reset.
        - ``{"statevector": True}``: explicitly request a final statevector.
        - ``{"statevector": False}``: suppress statevector output.

        Output consequences:

        - Counts are returned through ``Result.get_counts()`` as
          little-endian classical count-key strings.
        - A statevector, when produced, is returned through
          ``Result.get_statevector()``.
        - If a field was not produced, its accessor raises
          ``ResultFieldUnavailableError``.
        - ``Result.metadata`` always includes ``shots``, ``backend_name``,
          and the effective ``result_config``.

        Execution strategy:

        - Fast path: programs without reset, classical conditions, or reuse of
          a measured subsystem are evolved once. Requested counts are sampled
          from terminal measurement mappings without replaying the full
          circuit shot by shot.
        - Dynamic path: programs with reset, classical conditions, or reuse of
          measured subsystems are executed shot by shot with an explicit
          classical register. This path preserves feedforward semantics and
          repeated measurement/reset behavior.
        - Parallel dynamic counts: when the dynamic path is used, counts are
          requested, multiple iterations are needed, and backend options
          allow it, shots may be distributed across worker processes. The
          counts are reproducible for a fixed ``seed`` regardless of serial
          vs parallel scheduling.

        Statevector semantics:

        - For non-stochastic programs, a produced statevector is the final
          evolved state after all operations.
        - For stochastic programs (any measurement or reset),
          ``statevector=True`` is only supported for ``shots == 1``; the
          returned statevector is the single-shot post-measurement/post-reset
          state.
        - A program may take the dynamic execution path yet still be
          non-stochastic, for example when it contains only classical
          conditions on never-written clbits. Such a program may still
          produce a default statevector.

        Shot semantics and validation:

        - ``shots`` matters whenever counts are requested.
        - ``shots`` must be an ``int`` whenever requested results depend on
          it.
        - Counts require ``shots > 0``.
        - Requesting a statevector for a stochastic program requires
          ``shots == 1``.

        Args:
            program: Program to execute.
            shots: Number of logical shots to run when counts are requested.
                For statevector-only deterministic execution, the value may be
                ignored.
            result_config: Optional plain dictionary describing which result
                fields to produce. Supported keys are ``counts`` and
                ``statevector``; unknown keys are ignored with a warning.
                When omitted, backend defaults are used.
            seed: Optional root seed for the run. For dynamic counts, one
                reproducible child RNG stream is derived per logical shot.

        Returns:
            A completed ``Job``. Validation failures raise directly from
            ``run()``; execution-stage failures are captured in a failed job
            whose ``result()`` re-raises the underlying exception.

        Raises:
            BackendValidationError: If requested outputs are incompatible with
                the program shape or ``shots``.
            UnsupportedOperationError: If the program contains an operation
                without a backend implementation, or one whose target key
                (e.g. a non-neighbor qubit pair) is illegal for this backend.

        Examples:
            Sample counts from a measured program:

            >>> import fatqat as fq
            >>> program = fq.Program(1, 1)
            >>> program.add(fq.ops.X, 0)
            >>> program.add_measurement(0, 0)
            >>> result = fq.backends.StateVectorBackend().run(
            ...     program,
            ...     shots=100,
            ...     result_config={"counts": True},
            ... ).result()
            >>> result.get_counts()
            {'1': 100}

            Request a deterministic statevector:

            >>> program = fq.Program(1)
            >>> program.add(fq.ops.H, 0)
            >>> result = fq.backends.StateVectorBackend().run(
            ...     program,
            ...     result_config={"counts": False, "statevector": True},
            ... ).result()
            >>> result.get_statevector()
            array([0.70710678+0.j, 0.70710678+0.j])
        """
        return super().run(
            program, shots=shots, result_config=result_config, seed=seed
        )
