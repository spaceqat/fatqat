"""Public configuration boundaries for simulator backends."""

from dataclasses import dataclass

import numpy as np
import pytest

import fatqat as fq
from fatqat.backends import SCQubitIBMSimulator, SimulatorBackend
from fatqat.backends.engine_contract import _SimulationConfig
from fatqat.errors import BackendValidationError
from fatqat.result import Result, _ResultConfig


@dataclass(frozen=True)
class _ExtendedResultConfig(_ResultConfig):
    hardware_trace: bool | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.hardware_trace is not None and type(self.hardware_trace) is not bool:
            raise BackendValidationError(
                f"hardware_trace must be bool or None, got {self.hardware_trace!r}"
            )


@dataclass(frozen=True)
class _ExtendedSimulationConfig(_SimulationConfig):
    maximum_walkers: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.maximum_walkers is not None and (
            type(self.maximum_walkers) is not int or self.maximum_walkers < 1
        ):
            raise BackendValidationError(
                "maximum_walkers must be a positive int or None"
            )


class _ExtendedResultBackend(SimulatorBackend):
    _result_config_cls = _ExtendedResultConfig
    _simulation_config_cls = _ExtendedSimulationConfig

    def _validate_additional_config(self, *, config, simulation, shots, facts):
        if config.hardware_trace and simulation.maximum_walkers is None:
            raise BackendValidationError("hardware_trace requires maximum_walkers")

    def _additional_result_data(self, *, config, simulation, raw):
        if config.hardware_trace:
            return {"hardware_trace": {"maximum_walkers": simulation.maximum_walkers}}
        return {}


def _measured_superposition() -> fq.Program:
    program = fq.Program(1, 1)
    program.add(fq.ops.SX, 0)
    program.measure(0, 0)
    return program


def test_simulation_and_result_configuration_are_separate():
    result = (
        SimulatorBackend(method="statevector", runtime="numpy")
        .run(
            _measured_superposition(),
            shots=1,
            simulation_config={"seed": 4, "parallel_mode": "serial"},
            result_config={"counts": True, "final_state": True},
        )
        .result()
    )

    assert result.get_counts()
    assert np.allclose(np.sum(np.abs(result.get_statevector()) ** 2), 1)
    assert result.metadata["result_config"] == {
        "counts": True,
        "final_state": True,
    }


def test_final_state_uses_the_selected_method_and_hardware_backends_expose_it():
    result = (
        SCQubitIBMSimulator(method="density_matrix")
        .run(
            _measured_superposition(),
            shots=1,
            result_config={"counts": True, "final_state": True},
        )
        .result()
    )

    assert result.get_density_matrix().shape == (2, 2)


def test_backend_result_schema_controls_accepted_configuration_keys():
    program = _measured_superposition()

    with pytest.raises(BackendValidationError, match="does not support result_config"):
        SimulatorBackend().run(program, result_config={"hardware_trace": True})

    # The extended backend receives both declared fields and exposes its
    # requested artifact through the common Result interface.
    backend = _ExtendedResultBackend()
    result = backend.run(
        program,
        simulation_config={"maximum_walkers": 32},
        result_config={"hardware_trace": True},
    ).result()
    assert result.get_data("hardware_trace") == {"maximum_walkers": 32}
    assert result.metadata["simulation_config"]["maximum_walkers"] == 32
    assert result.metadata["result_config"]["hardware_trace"] is True

    with pytest.raises(BackendValidationError, match="requires maximum_walkers"):
        backend.run(program, result_config={"hardware_trace": True})

    with pytest.raises(BackendValidationError, match="hardware_trace must be bool"):
        backend.run(program, result_config={"hardware_trace": "yes"})


@pytest.mark.parametrize(
    "name", ["counts", "final_state", "statevector", "density_matrix"]
)
def test_result_rejects_reserved_backend_artifact_names(name):
    with pytest.raises(BackendValidationError, match="reserved field"):
        Result(data={name: object()})


@pytest.mark.parametrize(
    ("argument", "config", "match"),
    [
        ("simulation_config", {"threads": 4}, "does not support simulation_config"),
        (
            "simulation_config",
            {"maximum_walkers": 4},
            "does not support simulation_config",
        ),
        ("simulation_config", {"max_workers": 0}, "max_workers must be"),
        ("result_config", {"statevector": True}, "does not support result_config"),
        ("result_config", {"unknown": True, 3: True}, "does not support result_config"),
        ("result_config", {"final_state": "yes"}, "final_state must be bool"),
    ],
)
def test_invalid_configuration_fails_directly(argument, config, match):
    kwargs = {argument: config}

    with pytest.raises(BackendValidationError, match=match):
        SimulatorBackend().run(_measured_superposition(), **kwargs)
