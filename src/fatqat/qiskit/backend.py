"""Qiskit ``BackendV2`` wrapper around fatqat simulators."""

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
    """Gate-level fatqat simulator exposed as a Qiskit ``BackendV2``."""

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
        return self._sim_target

    @property
    def max_circuits(self) -> int | None:
        return None

    @property
    def coupling_map(self) -> None:
        """Fully-connected simulator: no topology constraint."""
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
                "seed_simulator must be an integer or None, "
                f"got {type(seed).__name__}"
            )

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
        try:
            for program, circuit in zip(programs, circuits):
                backend = Simulator(
                    method=self._method,
                    runtime=self._runtime,
                    noise=self._noise_model,
                )
                result_config = {"counts": True} if circuit.num_clbits > 0 else {}
                fatqat_results.append(
                    backend.run(
                        program,
                        shots=shots,
                        simulation_config={"seed": seed},
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
            seed_simulator=seed,
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
