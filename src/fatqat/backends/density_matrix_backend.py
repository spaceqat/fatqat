"""Density-matrix backend: validate, execute, assemble Result, return Job."""

from __future__ import annotations

from ..implementation import ImplementationMap
from ..job import Job
from ..program import Program
from ..result import _DensityMatrixResultConfig
from .engine_contract import _DensityMatrixResultRequest
from .matrix_backend import _MatrixBackendBase


class DensityMatrixBackend(_MatrixBackendBase):
    """Density-matrix backend for ``fatqat.Program`` execution.

    Sibling of ``StateVectorBackend`` with the same execution skeleton, but the
    engine-owned state is a dense density matrix, so mixed states are
    represented exactly. The backend supports matrix-evolvable gates, grouped
    measurement, feedforward conditions, and reset. Each run is classified
    into one of two execution strategies:

    - Fast path: used when the program has no classically conditioned
      operation and no operation (gate or reset) that acts on a subsystem
      after that subsystem has been measured. The density matrix is evolved
      once; unconditional resets are applied inline as the deterministic
      partial-trace channel, and requested counts are then sampled from the
      resulting measurement distribution.
    - Dynamic path: used when the program contains a classical condition or
      reuse of a measured subsystem. The backend executes one shot at a time
      while tracking the classical register explicitly, because later
      operations may depend on earlier measurement outcomes.

    Unlike the statevector backend, reset alone does not force the dynamic
    path: on a density matrix, reset is a deterministic channel (partial
    trace plus repreparation in ``|0><0|``) rather than a sampled branch.

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

    _result_config_cls = _DensityMatrixResultConfig
    _request_cls = _DensityMatrixResultRequest
    _state_field = "density_matrix"
    _reset_is_stochastic = False

    def __init__(
        self,
        options: dict[str, Any] | None = None,
        implementation_map: ImplementationMap | None = None,
    ) -> None:
        """Create a density-matrix backend.

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
        - ``{"density_matrix": None}``: a density matrix is produced only when
          the program contains no measurement. Reset does not suppress the
          default, because density-matrix reset is deterministic.
        - ``{"density_matrix": True}``: explicitly request a final density
          matrix.
        - ``{"density_matrix": False}``: suppress density-matrix output.

        Output consequences:

        - Counts are returned through ``Result.get_counts()`` as
          little-endian classical count-key strings.
        - A density matrix, when produced, is returned through
          ``Result.get_density_matrix()``.
        - If a field was not produced, its accessor raises
          ``ResultFieldUnavailableError``.
        - ``Result.metadata`` always includes ``shots``, ``backend_name``,
          and the effective ``result_config``.

        Execution strategy:

        - Fast path: programs without classical conditions or reuse of a
          measured subsystem are evolved once. Unconditional resets are
          applied inline as the deterministic partial-trace channel.
          Requested counts are sampled from terminal measurement mappings
          without replaying the full circuit shot by shot.
        - Dynamic path: programs with classical conditions or reuse of
          measured subsystems are executed shot by shot with an explicit
          classical register. This path preserves feedforward semantics and
          repeated measurement/reset behavior.
        - Parallel dynamic counts: when the dynamic path is used, counts are
          requested, multiple iterations are needed, and backend options
          allow it, shots may be distributed across worker processes. The
          counts are reproducible for a fixed ``seed`` regardless of serial
          vs parallel scheduling.

        Density-matrix semantics:

        - For measurement-free programs, the produced density matrix is the
          final evolved state after all operations, including deterministic
          reset channels; a reset acting on an entangled subsystem yields a
          mixed state, which this backend represents exactly.
        - For programs with measurement, ``density_matrix=True`` is only
          supported for ``shots == 1``; the returned density matrix is the
          single-shot post-measurement state.
        - A program may take the dynamic execution path yet contain no
          measurement, for example when it contains only classical conditions
          on never-written clbits. Such a program may still produce a default
          density matrix.

        Shot semantics and validation:

        - ``shots`` matters whenever counts are requested.
        - ``shots`` must be an ``int`` whenever requested results depend on
          it.
        - Counts require ``shots > 0``.
        - Requesting a density matrix for a program with measurement requires
          ``shots == 1``.

        Args:
            program: Program to execute.
            shots: Number of logical shots to run when counts are requested.
                For density-matrix-only deterministic execution, the value may
                be ignored.
            result_config: Optional plain dictionary describing which result
                fields to produce. Supported keys are ``counts`` and
                ``density_matrix``; unknown keys are ignored with a warning.
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
            >>> result = fq.backends.DensityMatrixBackend().run(
            ...     program,
            ...     shots=100,
            ...     result_config={"counts": True},
            ... ).result()
            >>> result.get_counts()
            {'1': 100}

            Request a deterministic density matrix:

            >>> program = fq.Program(1)
            >>> program.add(fq.ops.H, 0)
            >>> result = fq.backends.DensityMatrixBackend().run(
            ...     program,
            ...     result_config={"counts": False, "density_matrix": True},
            ... ).result()
            >>> result.get_density_matrix()
            array([[0.5+0.j, 0.5+0.j],
                   [0.5+0.j, 0.5+0.j]])
        """
        return super().run(
            program, shots=shots, result_config=result_config, seed=seed
        )
