"""Three-level atom family-owned noise support contracts."""

import numpy as np
import pytest

import fatqat as fq
from fatqat.errors import BackendValidationError
from fatqat.noise import (
    AmplitudeDamping,
    Depolarizing,
    NoiseModel,
    PhaseDamping,
    ReadoutConfusion,
    ThermalRelaxation,
)


def _noise(channel):
    noise = NoiseModel()
    noise.add(channel, targets=0)
    return noise


def _backend(model, *, noise=None):
    return fq.emulator.Atom3LevelEmulator(
        model,
        arrangement=fq.emulator.AtomArrangement.rectangular(1, 1, 2.0),
        noise=noise,
    )


@pytest.mark.parametrize(
    "channel",
    [
        AmplitudeDamping(rate=(0.1, 0.2)),
        PhaseDamping(rate=0.2),
        ThermalRelaxation(t1=10.0, t2=15.0),
        Depolarizing(rate=0.2),
    ],
)
def test_atom_3level_rejects_all_continuous_noise_declarations(
    atom_3level_model, channel
):
    noise = _noise(channel)
    backend = _backend(atom_3level_model)

    with pytest.raises(BackendValidationError, match=type(channel).__name__):
        backend.validate_noise_model(noise)
    with pytest.raises(BackendValidationError, match=type(channel).__name__):
        _backend(atom_3level_model, noise=noise)


def test_invalid_attached_noise_is_rejected_before_target_construction(
    atom_3level_model, monkeypatch
):
    noise = _noise(PhaseDamping(rate=0.2))

    def target_must_not_be_built(*_args, **_kwargs):
        raise AssertionError("target was built before noise validation")

    monkeypatch.setattr(
        "fatqat.emulator.atom_3level.backend._Atom3LevelTarget",
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
