"""Qiskit ``BackendV2`` wrapper around the gate-level FATQAT simulator."""

from __future__ import annotations

import uuid
from typing import Any, Iterable

from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit.providers.options import Options
from qiskit.transpiler import Target

from fatqat import __version__
from fatqat.noise import NoiseModel
from fatqat.simulator import Simulator

from .converter import circuit_to_program
from .errors import QiskitBackendError
from .job import FatqatJob
from .result import build_qiskit_result
from .target import build_simulator_target


class FatqatBackend(BackendV2):
    """Expose FATQAT's gate-level simulator as a Qiskit ``BackendV2``.

    Execution is synchronous: :meth:`run` completes conversion and execution
    before returning a terminal :class:`~fatqat.qiskit.FatqatJob`. The backend
    has an unbounded, fully connected target. Transpile circuits to ``target``
    before running when they contain gates outside its instruction basis.

    Args:
        method: FATQAT simulation method. Defaults to ``"statevector"`` and
            accepts the same case-insensitive names and aliases as
            :class:`fatqat.simulator.Simulator`.
        runtime: FATQAT numerical runtime, ``"numpy"`` (default) or
            ``"numba"``, case-insensitive.
        noise_model: Optional FATQAT :class:`~fatqat.NoiseModel`. Qiskit Aer
            noise models are not accepted.
        provider: Provider reported through the Qiskit backend interface.
        name: Qiskit backend name. Defaults to ``"fatqat_simulator"``.

    Raises:
        TypeError: If ``noise_model`` is neither a FATQAT ``NoiseModel`` nor
            ``None``.

    Invalid ``method`` or ``runtime`` values appear as an error job from
    :meth:`run`, rather than failing construction.
    """

    def __init__(
        self,
        *,
        method: str = "statevector",
        runtime: str = "numpy",
        noise_model: NoiseModel | None = None,
        provider: Any = None,
        name: str = "fatqat_simulator",
    ) -> None:
        if noise_model is not None and not isinstance(noise_model, NoiseModel):
            raise TypeError(
                "noise_model must be fatqat.NoiseModel; Qiskit Aer noise models "
                "are not accepted by this adapter"
            )
        self._method = method
        self._runtime = runtime
        self._noise_model = noise_model
        self._sim_target = build_simulator_target()
        super().__init__(
            provider=provider,
            name=name,
            description="fatqat gate-level simulator",
            backend_version=__version__,
        )

    @property
    def target(self) -> Target:
        """Return the supported instruction basis for transpilation."""
        return self._sim_target

    @property
    def max_circuits(self) -> int | None:
        """Return ``None``; circuit batches are not capped."""
        return None

    @property
    def coupling_map(self) -> None:
        """Return ``None``; circuits are not restricted to a coupling map."""
        return None

    @classmethod
    def _default_options(cls) -> Options:
        options = Options(
            shots=1024,
            memory=False,
            seed_simulator=None,
        )
        options.set_validator("shots", int)
        options.set_validator("memory", bool)
        return options

    def _validate_run_options(self, run_options: dict[str, Any]) -> dict[str, Any]:
        allowed = {"shots", "memory", "seed_simulator"}
        unknown = set(run_options) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise QiskitBackendError(f"unsupported run option(s): {names}")

        options = Options(
            shots=self.options.shots,
            memory=self.options.memory,
            seed_simulator=self.options.seed_simulator,
        )
        options.set_validator("shots", int)
        options.set_validator("memory", bool)
        try:
            options.update_options(**run_options)
        except (TypeError, ValueError) as exc:
            raise QiskitBackendError(str(exc)) from exc

        shots = options.shots
        if type(shots) is not int:
            raise QiskitBackendError(
                f"shots must be an integer, got {type(shots).__name__}"
            )
        if shots <= 0:
            raise QiskitBackendError(f"shots must be > 0, got {shots}")

        memory = options.memory
        if type(memory) is not bool:
            raise QiskitBackendError(
                f"memory must be a boolean, got {type(memory).__name__}"
            )

        seed = options.seed_simulator
        if seed is not None and type(seed) is not int:
            raise QiskitBackendError(
                f"seed_simulator must be an integer or None, got {type(seed).__name__}"
            )
        if seed is not None and seed < 0:
            raise QiskitBackendError(f"seed_simulator must be >= 0, got {seed}")

        return {
            "shots": shots,
            "memory": memory,
            "seed_simulator": seed,
        }

    def run(
        self,
        run_input: QuantumCircuit | Iterable[QuantumCircuit],
        **run_options: Any,
    ) -> FatqatJob:
        """Convert and execute one circuit or a nonempty circuit iterable.

        Supported run options are ``shots`` (positive ``int``, default 1024),
        ``memory`` (``bool``, default ``False``), and ``seed_simulator``
        (non-negative ``int`` or ``None``, default ``None``). Per-call values
        override ``backend.options``. Other FATQAT execution controls are not
        exposed by this backend.

        Execution completes before :meth:`run` returns. Invalid input
        collections, unknown options,
        nonpositive ``shots``, negative seeds, and option-type errors raise
        immediately. Conversion and simulator failures return a job with
        ``ERROR`` status; calling its :meth:`~FatqatJob.result` raises Qiskit's
        ``QiskitError`` with the original failure chained.

        Counts use Qiskit's normal result formatting. With ``memory=True``,
        the result includes entries consistent with those counts; do not rely
        on their order to reconstruct shot chronology. A circuit with no
        classical bits produces neither counts nor memory data.

        Args:
            run_input: One Qiskit ``QuantumCircuit`` or a nonempty iterable of
                circuits.
            **run_options: ``shots``, ``memory``, and/or ``seed_simulator``.

        Returns:
            A completed :class:`~fatqat.qiskit.FatqatJob` containing a Qiskit
            ``Result`` or execution failure.

        Raises:
            QiskitBackendError: If ``run_input`` is empty or contains a
                non-circuit value; an option is unknown; ``shots`` is not a
                positive integer; ``memory`` is not a boolean; or
                ``seed_simulator`` is not a non-negative integer or ``None``.
        """
        circuits = self._normalize_circuits(run_input)
        validated = self._validate_run_options(dict(run_options))
        shots = validated["shots"]
        memory = validated["memory"]
        seed = validated["seed_simulator"]

        programs = []
        for circuit in circuits:
            try:
                programs.append(circuit_to_program(circuit))
            except Exception as exc:
                job_id = str(uuid.uuid4())
                return FatqatJob(self, job_id, error=exc)

        fatqat_results = []
        circuit_seeds: list[int | None] = []
        try:
            for index, (program, circuit) in enumerate(zip(programs, circuits)):
                backend = Simulator(
                    method=self._method,
                    runtime=self._runtime,
                    noise=self._noise_model,
                )
                result_config = {"counts": True} if circuit.num_clbits > 0 else {}
                # Distinct per-circuit seeds, as Aer does: reusing one seed
                # would make identical circuits in a batch return identical
                # samples.
                circuit_seed = None if seed is None else seed + index
                circuit_seeds.append(circuit_seed)
                fatqat_results.append(
                    backend.run(
                        program,
                        shots=shots,
                        simulation_config={"seed": circuit_seed},
                        result_config=result_config,
                    ).result()
                )
        except Exception as exc:
            job_id = str(uuid.uuid4())
            return FatqatJob(self, job_id, error=exc)

        qiskit_result = build_qiskit_result(
            backend_name=self.name,
            backend_version=self.backend_version,
            circuits=circuits,
            fatqat_results=fatqat_results,
            shots=shots,
            memory=memory,
            seed_simulators=circuit_seeds,
        )
        return FatqatJob(self, qiskit_result.job_id, result=qiskit_result)

    @staticmethod
    def _normalize_circuits(
        run_input: QuantumCircuit | Iterable[QuantumCircuit],
    ) -> list[QuantumCircuit]:
        if isinstance(run_input, QuantumCircuit):
            return [run_input]
        circuits = list(run_input)
        if not circuits:
            raise QiskitBackendError("run_input must contain at least one circuit")
        for circuit in circuits:
            if not isinstance(circuit, QuantumCircuit):
                raise QiskitBackendError(
                    "every run_input entry must be a QuantumCircuit"
                )
        return circuits
