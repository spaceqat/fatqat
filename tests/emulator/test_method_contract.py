"""Shared representation contract for pulse emulators."""

import warnings

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.emulator._atom_3level import Atom3LevelEmulator
from fatqat.errors import BackendValidationError, ResultFieldUnavailableError
from fatqat.noise import NoiseModel, TransitionRelaxation


def _family_case(name, method, atom_3level_model):
    if name == "transmon":
        model = fq.emulator.TransmonModel.from_document(
            fq.emulator.load_model_document("transmon.reference")
        )
        program = fq.Program(2)
        program.add(ops.RX(0.1), 0)
        return fq.emulator.TransmonEmulator(model, method=method), program, 9
    if name == "atom3":
        arrangement = fq.emulator.AtomArrangement.chain(2, spacing=6.0)
        program = fq.Program(2)
        program.add(ops.RX(0.1), 0)
        return (
            Atom3LevelEmulator(
                atom_3level_model, arrangement=arrangement, method=method
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
def test_method_selects_one_full_physical_artifact(
    family, method, field, shape, atom_3level_model
):
    backend, program, dimension = _family_case(family, method, atom_3level_model)

    result = backend.run(program).result()

    assert backend.method == method
    assert result.metadata["method"] == method
    assert result.metadata["runtime"] == "qutip"
    assert set(result.metadata["runtime_details"]) == {
        "solver",
        "solver_options",
    }
    runtime_details = result.metadata["runtime_details"]
    assert (
        runtime_details["solver"]
        == {
            "statevector": "sesolve",
            "density_matrix": "mesolve",
            "unitary": "propagator",
        }[method]
    )
    assert runtime_details["solver_options"]["max_step"] == pytest.approx(
        {
            "transmon": 0.078125,
            "atom3": 0.1 / (4 * np.pi),
            "atom2": 0.05,
        }[family]
    )
    assert "solver" not in result.metadata
    assert "frame_convention" not in runtime_details
    assert result.available_data == {field}
    assert result.metadata["state_axes"]
    artifact = getattr(result, f"get_{field}")()
    assert artifact.shape == shape(dimension)
    if method == "unitary":
        # Reject a materially nonunitary result without demanding accuracy
        # beyond QuTiP's default integration tolerances.
        assert np.allclose(
            artifact.conj().T @ artifact,
            np.eye(dimension),
            rtol=0.0,
            atol=2e-3,
        )
    for alternate in {"statevector", "density_matrix", "unitary"} - {field}:
        with pytest.raises(ResultFieldUnavailableError):
            getattr(result, f"get_{alternate}")()


@pytest.mark.parametrize("family", ("transmon", "atom3", "atom2"))
def test_counts_only_warns_when_a_declared_clbit_is_never_measured(
    family, atom_3level_model
):
    backend, _program, _dimension = _family_case(
        family, "density_matrix", atom_3level_model
    )
    program = fq.Program(2, 2)
    program.measure(0, 0)

    with pytest.warns(UserWarning, match="clbits that were never measured"):
        result = backend.run(program, shots=2).result()

    assert result.get_counts() == {"00": 2}


@pytest.mark.parametrize("family", ("transmon", "atom3", "atom2"))
def test_counts_only_does_not_warn_when_every_declared_clbit_is_measured(
    family, atom_3level_model
):
    backend, _program, _dimension = _family_case(
        family, "density_matrix", atom_3level_model
    )
    program = fq.Program(2, 2)
    program.measure(0, 0)
    program.measure(1, 1)

    with warnings.catch_warnings(record=True) as caught:
        backend.run(program, shots=2).result()

    assert not any(
        "clbits that were never measured" in str(item.message) for item in caught
    )


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
    noise.add(
        TransitionRelaxation(
            rate=0.01,
            coefficients={(1, 0): 1, (2, 1): np.sqrt(2)},
        ),
        targets="q0",
    )
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
