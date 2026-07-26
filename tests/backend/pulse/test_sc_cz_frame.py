"""CZ park calibration is evaluated in the adapter's shared-q0 frame."""

import json
from pathlib import Path

import numpy as np

from fatqat.backends.pulse.superconducting import (
    load_calibration_spec,
    load_physics_model,
)

_FIXTURES = Path(__file__).parent / "fixtures"


def test_cz_park_detuning_cancels_the_declared_shared_frame_split():
    model = load_physics_model(
        json.loads((_FIXTURES / "sc_transmon_exchange.json").read_text())
    )
    calibration = load_calibration_spec(
        json.loads((_FIXTURES / "sc_transmon_exchange_calibration.json").read_text()),
        model,
    )
    recipe = calibration.recipe("cz")["edges"][0]
    first = model.subsystems[0]
    second = model.subsystems[1]
    shared_frame_split = first.frequency_ghz - second.frequency_ghz
    assert np.isclose(
        recipe["detuning_ghz"], -(shared_frame_split + first.anharmonicity_ghz)
    )
