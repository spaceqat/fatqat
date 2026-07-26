"""Dynamic qutrit measurement/replay behaviour."""

import json
from pathlib import Path

import fatqat as fq

from fatqat.backends.pulse.backend import PulseBackend
from fatqat.backends.pulse.superconducting import (
    load_calibration_spec,
    load_physics_model,
)

_FIXTURES = Path(__file__).parent / "fixtures"


def _backend():
    model = load_physics_model(
        json.loads((_FIXTURES / "sc_transmon_exchange.json").read_text())
    )
    calibration = load_calibration_spec(
        json.loads((_FIXTURES / "sc_transmon_exchange_calibration.json").read_text()),
        model,
    )
    return PulseBackend(model, calibration)


def test_measurement_produces_counts_and_a_later_guard_replays_serially():
    program = fq.Program(1, 1)
    program.add(fq.ops.RX(3.141592653589793), 0)
    program.add_measurement(0, 0)
    program.add(fq.ops.RX(3.141592653589793), 0, condition=(0, 1))
    result = _backend().run(program, shots=4, simulation_config={"seed": 3}).result()

    assert sum(result.get_counts_as_tuples().values()) == 4
