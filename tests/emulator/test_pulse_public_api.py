"""Public release-surface tests for the superconducting pulse backend."""

import json
from pathlib import Path

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

_FIXTURES = Path(__file__).parent / "fixtures"


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


def test_replacing_cz_through_the_public_surface_never_imports_fatqat_emulator():
    """The documented custom-CZ workflow, reachable entirely from `fq.backends`.

    Mirrors the plan's locked construction:

        implementations = default_superconducting_pulse_implementation_map()
        implementations.add(fq.ops.CZ, custom_cz)
        backend = fq.backends.PulseBackend(
            model, calibration, pulse_implementation_map=implementations
        )
    """
    model = load_physics_model(
        json.loads((_FIXTURES / "sc_transmon_exchange.json").read_text())
    )
    calibration = load_calibration_spec(
        json.loads((_FIXTURES / "sc_transmon_exchange_calibration.json").read_text()),
        model,
    )

    def custom_cz(operation, *, targets, model, calibration):
        first, second = (model.subsystem_ids[model.bind_resource(t)] for t in targets)
        duration = 10.0
        tlist = (0.0, duration)
        return PulseDefinition(
            duration,
            (SampledControl(model.exchange_control(first, second), tlist, (0.0, 0.0)),),
            (
                model.resource(first),
                model.resource(second),
                model.coupling(first, second),
            ),
            (PhaseShift(model.frame(first), 0.05),),
        )

    implementations = default_superconducting_pulse_implementation_map()
    implementations.add(fq.ops.CZ, custom_cz)
    backend = fq.backends.PulseBackend(
        model, calibration, pulse_implementation_map=implementations
    )

    program = fq.Program(2)
    program.add(fq.ops.CZ, (0, 1))
    plan, _facts = backend._lower_program(program)

    (block,) = plan
    assert block.duration == 10.0
    (control,) = block.controls
    assert control.channel == model.exchange_control("q0", "q1")
