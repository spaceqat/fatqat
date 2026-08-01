"""Public release-surface tests: export identity and private-type exclusion.

Strictly about what `fatqat.backends` re-exports. The documented custom-CZ
workflow that exercises those exports end to end lives in
`test_pulse_custom_implementation_workflow.py`.
"""

import fatqat as fq
from fatqat.backends import (
    PhaseShift,
    PhaseSwap,
    PulseBackend,
    PulseDefinition,
    PulseImplementationMap,
    SampledControl,
    SCTransmonExchangeBuilder,
    default_superconducting_pulse_implementation_map,
    load_calibration_spec,
    load_physics_model,
)
from fatqat.emulator.pulse import PhaseShift as _PhaseShift
from fatqat.emulator.pulse import PhaseSwap as _PhaseSwap
from fatqat.emulator.pulse import PulseDefinition as _PulseDefinition
from fatqat.emulator.pulse import PulseImplementationMap as _PulseImplementationMap
from fatqat.emulator.pulse import SampledControl as _SampledControl


def test_sc_pulse_factories_are_public_without_exposing_execution_types():
    assert fq.backends.PulseBackend is PulseBackend
    assert fq.backends.SCTransmonExchangeBuilder is SCTransmonExchangeBuilder
    assert fq.backends.load_physics_model is load_physics_model
    assert fq.backends.load_calibration_spec is load_calibration_spec
    assert not hasattr(fq.backends, "SCQutipAdapter")
    assert not hasattr(fq.backends, "PulseEngine")
    assert not hasattr(fq.backends, "PulseBlock")


def test_pulse_authoring_values_are_public_and_identical_to_private_definitions():
    assert fq.backends.PulseDefinition is PulseDefinition is _PulseDefinition
    assert fq.backends.SampledControl is SampledControl is _SampledControl
    assert fq.backends.PhaseShift is PhaseShift is _PhaseShift
    assert fq.backends.PhaseSwap is PhaseSwap is _PhaseSwap
    assert (
        fq.backends.PulseImplementationMap
        is PulseImplementationMap
        is _PulseImplementationMap
    )
    assert (
        fq.backends.default_superconducting_pulse_implementation_map
        is default_superconducting_pulse_implementation_map
    )
