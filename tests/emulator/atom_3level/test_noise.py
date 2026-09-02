"""Three-level atom family-owned noise support contracts."""

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.emulator._atom_3level import Atom3LevelEmulator
from fatqat.errors import BackendValidationError
from fatqat.noise import (
    AmplitudeDamping,
    Depolarizing,
    NoiseModel,
    PhaseDamping,
    ReadoutConfusion,
    ThermalRelaxation,
    TransitionRelaxation,
)


def _noise(channel):
    noise = NoiseModel()
    noise.add(channel, targets=0)
    return noise


def _backend(model, *, noise=None):
    return Atom3LevelEmulator(
        model,
        arrangement=fq.emulator.AtomArrangement.rectangular(1, 1, 2.0),
        noise=noise,
    )


@pytest.mark.parametrize(
    "channel",
    [
        AmplitudeDamping(rate=0.1),
        TransitionRelaxation(p=0.1, coefficients={(2, 0): 1.0}),
        PhaseDamping(rate=0.2),
        ThermalRelaxation(t1=10.0, t2=15.0),
        Depolarizing(rate=0.2),
    ],
)
def test_atom_3level_rejects_finite_and_unsupported_noise_declarations(
    atom_3level_model, channel
):
    noise = _noise(channel)
    backend = _backend(atom_3level_model)

    with pytest.raises(BackendValidationError, match=type(channel).__name__):
        backend.validate_noise_model(noise)
    with pytest.raises(BackendValidationError, match=type(channel).__name__):
        _backend(atom_3level_model, noise=noise)


def test_atom_3level_rejects_transition_levels_outside_the_physical_space(
    atom_3level_model,
):
    noise = _noise(TransitionRelaxation(rate=0.2, coefficients={(3, 0): 1.0}))

    with pytest.raises(BackendValidationError, match="outside physical dimension 3"):
        _backend(atom_3level_model, noise=noise)


def test_atom_3level_default_noisy_statevector_is_a_seeded_trajectory(
    atom_3level_model,
):
    noise = _noise(TransitionRelaxation(rate=0.3, coefficients={(1, 0): 1.0}))
    backend = _backend(atom_3level_model, noise=noise)
    program = fq.Program(1)
    program.add(ops.RX(np.pi), 0)
    kwargs = {
        "shots": 1,
        "simulation_config": {"seed": 29},
        "result_config": {"counts": False, "final_state": True},
    }

    first = backend.run(program, **kwargs).result()
    second = backend.run(program, **kwargs).result()

    assert first.metadata["runtime_details"]["solver"] == "mcsolve"
    assert first.get_statevector() == pytest.approx(second.get_statevector())
    assert first.get_statevector().shape == (3,)
    assert np.linalg.norm(first.get_statevector()) == pytest.approx(1.0)


def test_invalid_attached_noise_is_rejected_before_target_construction(
    atom_3level_model, monkeypatch
):
    noise = _noise(PhaseDamping(rate=0.2))

    def target_must_not_be_built(*_args, **_kwargs):
        raise AssertionError("target was built before noise validation")

    monkeypatch.setattr(
        "fatqat.emulator._atom_3level.backend._Atom3LevelTarget",
        target_must_not_be_built,
    )
    with pytest.raises(BackendValidationError, match="PhaseDamping"):
        _backend(atom_3level_model, noise=noise)


def test_atom_3level_accepts_binary_and_rejects_nonbinary_readout_confusion(
    atom_3level_model,
):
    valid = NoiseModel()
    valid.add(ReadoutConfusion(np.eye(2)))
    assert _backend(atom_3level_model).validate_noise_model(valid) is None

    invalid = NoiseModel()
    invalid.add(ReadoutConfusion(np.eye(3)))
    with pytest.raises(BackendValidationError, match="2 x 2"):
        _backend(atom_3level_model).validate_noise_model(invalid)
    with pytest.raises(BackendValidationError, match="2 x 2"):
        _backend(atom_3level_model, noise=invalid)
