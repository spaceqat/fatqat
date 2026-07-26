"""The completed pulse surface is exported independently of fake targets."""

import fatqat as fq


def test_pulse_backend_and_data_only_factories_are_public():
    assert fq.backends.PulseBackend.__name__ == "PulseBackend"
    assert callable(fq.backends.load_physics_model)
    assert callable(fq.backends.load_calibration_spec)
