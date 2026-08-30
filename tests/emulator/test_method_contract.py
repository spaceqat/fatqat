"""Shared public representation contract for pulse emulators."""

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.errors import BackendValidationError, ResultFieldUnavailableError
from fatqat.noise import AmplitudeDamping, NoiseModel


def _family_case(name, method):
    if name == "transmon":
        model = fq.emulator.TransmonModel.from_document(
            fq.emulator.load_model_document("transmon.reference")
        )
        program = fq.Program(2)
        program.add(ops.RZ(0.2), 0)
        return fq.emulator.TransmonEmulator(model, method=method), program, 9
    if name == "atom3":
        model = fq.emulator.Atom3LevelModel.from_document(
            fq.emulator.load_model_document("atom3level.reference")
        )
        arrangement = fq.emulator.AtomArrangement.chain(2, spacing=6.0)
        program = fq.Program(2)
        program.add(ops.RZ(0.2), 0)
        return (
            fq.emulator.Atom3LevelEmulator(
                model, arrangement=arrangement, method=method
            ),
            program,
            9,
        )
    model = fq.emulator.Atom2LevelModel.from_document(
        fq.emulator.load_model_document("atom2level.reference")
    )
    arrangement = fq.emulator.AtomArrangement.chain(2, spacing=6.0)
    program = fq.Program(2)
    program.add(
        ops.PulseOperation(
            0.1,
            (
                fq.emulator.PulseControl(
                    model.control.drive(),
                    fq.emulator.SampledWaveform((0.0, 0.1), (0.2, 0.2)),
                ),
            ),
        )
    )
    return (
        fq.emulator.Atom2LevelEmulator(model, arrangement=arrangement, method=method),
        program,
        4,
    )


@pytest.mark.parametrize("family", ("transmon", "atom3", "atom2"))
@pytest.mark.parametrize(
    ("method", "field", "shape"),
    (
        ("statevector", "statevector", lambda dimension: (dimension,)),
        ("density_matrix", "density_matrix", lambda dimension: (dimension,) * 2),
        ("unitary", "unitary", lambda dimension: (dimension,) * 2),
    ),
)
def test_method_selects_one_full_physical_artifact(family, method, field, shape):
    backend, program, dimension = _family_case(family, method)

    result = backend.run(program).result()

    assert backend.method == method
    assert result.metadata["method"] == method
    assert result.available_data == {field}
    assert result.metadata["state_axes"]
    artifact = getattr(result, f"get_{field}")()
    assert artifact.shape == shape(dimension)
    if method == "unitary":
        assert np.allclose(artifact.conj().T @ artifact, np.eye(dimension))
    for alternate in {"statevector", "density_matrix", "unitary"} - {field}:
        with pytest.raises(ResultFieldUnavailableError):
            getattr(result, f"get_{alternate}")()


@pytest.mark.parametrize(
    ("spelling", "canonical"),
    (
        ("SV", "statevector"),
        ("STATEVECTOR", "statevector"),
        ("DM", "density_matrix"),
        ("DENSITY_MATRIX", "density_matrix"),
        ("UNITARY", "unitary"),
    ),
)
def test_method_spellings_are_canonical_and_property_is_read_only(
    model, spelling, canonical
):
    backend = fq.emulator.TransmonEmulator(model, method=spelling)

    assert backend.method == canonical
    with pytest.raises(AttributeError):
        backend.method = "unitary"


@pytest.mark.parametrize(
    "method", ("sesolve", "mesolve", "mcsolve", "densitymatrix", "auto", "superop")
)
def test_pulse_methods_reject_solver_and_unsupported_representation_names(
    model, method
):
    with pytest.raises(BackendValidationError, match="method"):
        fq.emulator.TransmonEmulator(model, method=method)


def test_method_has_no_per_run_override(model):
    backend = fq.emulator.TransmonEmulator(model)
    program = fq.Program(2)

    with pytest.raises(TypeError, match="method"):
        backend.run(program, method="density_matrix")
    with pytest.raises(BackendValidationError, match="simulation_config"):
        backend.run(program, simulation_config={"method": "density_matrix"})


def test_unitary_can_suppress_operator_construction(model, monkeypatch):
    backend = fq.emulator.TransmonEmulator(model, method="unitary")

    monkeypatch.setattr(
        "fatqat.emulator._core.engine.PulseEngine.propagator",
        lambda *_args, **_kwargs: pytest.fail("unitary construction should be skipped"),
    )
    result = backend.run(fq.Program(2), result_config={"final_state": False}).result()

    assert result.available_data == frozenset()
    assert result.metadata["method"] == "unitary"
    assert "state_axes" not in result.metadata


def test_noisy_statevector_defaults_to_metadata_and_explicit_shot_is_seeded(model):
    noise = NoiseModel()
    noise.add(AmplitudeDamping(rate=(0.01, 0.01)), targets="q0")
    backend = fq.emulator.TransmonEmulator(model, method="statevector", noise=noise)
    program = fq.Program(2)
    program.add(ops.RX(np.pi), 0)

    default = backend.run(program, simulation_config={"seed": 7}).result()
    first = backend.run(
        program,
        shots=1,
        simulation_config={"seed": 7},
        result_config={"final_state": True},
    ).result()
    second = backend.run(
        program,
        shots=1,
        simulation_config={"seed": 7},
        result_config={"final_state": True},
    ).result()
    ensemble = (
        fq.emulator.TransmonEmulator(model, method="density_matrix", noise=noise)
        .run(program)
        .result()
    )

    assert default.available_data == frozenset()
    assert default.metadata["method"] == "statevector"
    assert "state_axes" not in default.metadata
    assert first.get_statevector() == pytest.approx(second.get_statevector())
    assert ensemble.get_density_matrix().shape == (9, 9)


def test_stochastic_statevector_artifacts_require_one_shot(model):
    backend = fq.emulator.TransmonEmulator(model, method="statevector")
    reset = fq.Program(2)
    reset.add(ops.Reset, 0)

    with pytest.raises(BackendValidationError, match="shots == 1"):
        backend.run(reset, shots=2, result_config={"final_state": True})


def test_unitary_rejects_counts_even_without_measurement(model):
    backend = fq.emulator.TransmonEmulator(model, method="unitary")

    with pytest.raises(BackendValidationError, match="unitary.*counts"):
        backend.run(fq.Program(2), result_config={"counts": True, "final_state": False})
