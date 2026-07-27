"""Public release-surface tests for the superconducting pulse backend."""

import fatqat as fq
from fatqat.backends import (
    PulseBackend,
    SCTransmonExchangeBuilder,
    load_calibration_spec,
    load_physics_model,
)


def test_sc_pulse_factories_are_public_without_exposing_execution_types():
    assert fq.backends.PulseBackend is PulseBackend
    assert fq.backends.SCTransmonExchangeBuilder is SCTransmonExchangeBuilder
    assert fq.backends.load_physics_model is load_physics_model
    assert fq.backends.load_calibration_spec is load_calibration_spec
    assert not hasattr(fq.backends, "SCQutipAdapter")
    assert not hasattr(fq.backends, "PulseEngine")
