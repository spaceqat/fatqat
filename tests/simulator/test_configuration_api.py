"""Public configuration boundaries for simulator backends."""

from dataclasses import dataclass

import numpy as np
import pytest

import fatqat as fq
from fatqat.simulator import SCQubitIBMSimulator, Simulator
from fatqat._backends.engine_contract import _SimulationConfig
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


class _ExtendedResultBackend(Simulator):
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
        Simulator(method="statevector", runtime="numpy")
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
        Simulator().run(program, result_config={"hardware_trace": True})

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
        Simulator().run(_measured_superposition(), **kwargs)


def test_simulator_result_shot_validation_preserves_exact_messages():
    program = _measured_superposition()

    with pytest.raises(BackendValidationError) as exc:
        Simulator("SV").run(program, shots=1.5)
    assert str(exc.value) == (
        "shots must be an int when requested results depend on it, got 1.5"
    )

    with pytest.raises(BackendValidationError) as exc:
        Simulator("SV").run(program, shots=0)
    assert str(exc.value) == "counts require shots > 0, got shots=0"

    with pytest.raises(BackendValidationError) as exc:
        Simulator("SV").run(program, shots=2, result_config={"final_state": True})
    assert str(exc.value) == (
        "statevector with measurement, reset, or channel noise is only supported "
        "for shots == 1"
    )


def test_validation_and_execution_use_identical_result_flag_resolution(monkeypatch):
    from fatqat.simulator import simulator as simulator_module

    original = simulator_module._resolve_result_flags
    resolved = []

    def record(*args, **kwargs):
        flags = original(*args, **kwargs)
        resolved.append(flags)
        return flags

    monkeypatch.setattr(simulator_module, "_resolve_result_flags", record)
    result = Simulator("SV").run(fq.Program(1), shots=7).result()

    assert result.get_statevector().shape == (2,)
    assert resolved == [(False, True), (False, True)]


# --- public method accessor --------------------------------------------------


@pytest.mark.parametrize(
    "argument, expected",
    [
        ("SV", "statevector"),
        ("statevector", "statevector"),
        ("DM", "density_matrix"),
        ("density_matrix", "density_matrix"),
        ("unitary", "unitary"),
        ("superop", "superop"),
    ],
)
def test_method_reports_the_canonical_name(argument, expected):
    # The alias is normalized away, so a caller matching on this value never
    # has to know which spelling the backend was built with.
    assert fq.simulator.Simulator(method=argument).method == expected


def test_method_matches_the_metadata_of_a_run():
    # One string for both the precondition check and the result: a caller that
    # branches on backend.method can read the same value back off the result.
    program = fq.Program(1)
    program.add(fq.ops.H, 0)

    for argument in ("SV", "DM"):
        backend = fq.simulator.Simulator(method=argument)
        result = backend.run(
            program, shots=0, result_config={"counts": False, "final_state": True}
        ).result()

        assert backend.method == result.metadata["method"]
        assert backend.method in result.available_data


def test_method_is_read_only():
    backend = fq.simulator.Simulator(method="SV")

    with pytest.raises(AttributeError):
        backend.method = "density_matrix"


def test_method_does_not_require_a_run():
    # The whole point: it is a precondition, answerable before any evolution.
    assert fq.simulator.Simulator(method="unitary").method == "unitary"
