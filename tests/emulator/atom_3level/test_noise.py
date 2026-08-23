"""Three-level atom custom Lindblad-map and execution contracts."""

import inspect

import numpy as np
import pytest
from qutip import mesolve

import fatqat as fq
from fatqat._pulse_values import PulseControl
from fatqat.emulator._core.scheduling import schedule_pulse_run
from fatqat.errors import BackendValidationError
from fatqat.noise import (
    AmplitudeDamping,
    LindbladImplementationMap,
    NoiseModel,
    PhaseDamping,
)
from fatqat.noise.lindblad import (
    amplitude_damping_lindblad_rule,
    phase_damping_lindblad_rule,
)
from fatqat.waveforms import SampledWaveform


def _lindblad_map(rule=phase_damping_lindblad_rule):
    implementations = LindbladImplementationMap()
    implementations.register(PhaseDamping, rule)
    return implementations


def _noise(channel, *, operation=None, targets=None):
    model = NoiseModel()
    if operation is None and targets is None:
        targets = 0
    model.add(channel, operation=operation, targets=targets)
    return model


def _backend(model, *, noise=None, lindblad_map=None):
    return fq.emulator.Atom3LevelEmulator(
        model,
        arrangement=fq.emulator.AtomArrangement.rectangular(1, 1, 2.0),
        noise=noise,
        lindblad_implementation_map=lindblad_map,
    )


def _rx_program(angle=np.pi / 2):
    program = fq.Program(1)
    program.add(fq.ops.RX(angle), 0)
    return program


def test_lindblad_constructor_surface_defaults_types_and_copy_isolation(
    atom_3level_model,
):
    parameter = inspect.signature(fq.emulator.Atom3LevelEmulator).parameters[
        "lindblad_implementation_map"
    ]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None
    backend = _backend(atom_3level_model)
    assert not backend._lindblad_implementation_map.supported_channels()

    with pytest.raises(BackendValidationError, match="lindblad_implementation_map"):
        _backend(atom_3level_model, lindblad_map=object())

    supplied = _lindblad_map()
    copied = _backend(atom_3level_model, lindblad_map=supplied)
    supplied.register(PhaseDamping, lambda *_args, **_kwargs: ())
    assert (
        copied._lindblad_implementation_map.get(PhaseDamping)
        is phase_damping_lindblad_rule
    )
    explicit_empty = _backend(
        atom_3level_model,
        lindblad_map=LindbladImplementationMap(),
    )
    assert not explicit_empty._lindblad_implementation_map.supported_channels()


def test_default_rejects_physical_channels_but_custom_map_classifies_them(
    atom_3level_model,
):
    operation_noise = _noise(PhaseDamping(rate=0.2), operation=fq.ops.RX)
    with pytest.raises(BackendValidationError, match="PhaseDamping"):
        _backend(atom_3level_model, noise=operation_noise)

    custom = _backend(
        atom_3level_model,
        noise=operation_noise,
        lindblad_map=_lindblad_map(),
    )
    assert custom.check_noise_support(operation_noise).supported

    background = _noise(PhaseDamping(rate=0.3))
    assert (
        _backend(
            atom_3level_model,
            noise=background,
            lindblad_map=_lindblad_map(),
        )
        .check_noise_support(background)
        .supported
    )

    probability_background = _noise(PhaseDamping(p=0.2))
    classifier = _backend(atom_3level_model, lindblad_map=_lindblad_map())
    assert not classifier.check_noise_support(probability_background).supported
    with pytest.raises(BackendValidationError, match="PhaseDamping"):
        _backend(
            atom_3level_model,
            noise=probability_background,
            lindblad_map=_lindblad_map(),
        )


def test_custom_map_rejects_qutrit_amplitude_damping_with_wrong_arity(
    atom_3level_model,
):
    implementations = _lindblad_map()
    implementations.register(AmplitudeDamping, amplitude_damping_lindblad_rule)
    backend = _backend(atom_3level_model, lindblad_map=implementations)
    invalid = _noise(AmplitudeDamping(rate=(0.1,)), operation=fq.ops.RX)
    report = backend.check_noise_support(invalid)
    assert not report.supported
    assert report.rejected_sources == ("AmplitudeDamping(rate-arity-1)",)
    assert "requires 2 damping values" in report.warnings[0]

    valid = _noise(
        AmplitudeDamping(rate=(0.1, 0.2)),
        operation=fq.ops.RX,
    )
    assert backend.check_noise_support(valid).supported


def test_attached_noise_is_classified_once_before_target_construction(
    atom_3level_model, monkeypatch
):
    arrangement = fq.emulator.AtomArrangement.rectangular(1, 1, 2.0)
    invalid = _noise(PhaseDamping(rate=0.2))

    def target_must_not_be_built(*_args, **_kwargs):
        raise AssertionError("target was built before capability classification")

    monkeypatch.setattr(
        "fatqat.emulator.atom_3level.backend._Atom3LevelTarget",
        target_must_not_be_built,
    )
    with pytest.raises(BackendValidationError, match="PhaseDamping"):
        fq.emulator.Atom3LevelEmulator(
            atom_3level_model,
            arrangement=arrangement,
            noise=invalid,
        )

    monkeypatch.undo()
    calls = 0
    classify = fq.emulator.Atom3LevelEmulator._classify_noise

    def counted_classify(self, noise_model):
        nonlocal calls
        calls += 1
        return classify(self, noise_model)

    monkeypatch.setattr(
        fq.emulator.Atom3LevelEmulator,
        "_classify_noise",
        counted_classify,
    )
    _backend(
        atom_3level_model,
        noise=_noise(PhaseDamping(rate=0.2)),
        lindblad_map=_lindblad_map(),
    )
    assert calls == 1


def test_readout_shape_rule_is_independent_of_custom_lindblad_map(
    atom_3level_model,
):
    valid = NoiseModel()
    valid.add(fq.noise.ReadoutConfusion(np.eye(2)))
    assert (
        _backend(
            atom_3level_model,
            noise=valid,
            lindblad_map=_lindblad_map(),
        )
        .check_noise_support(valid)
        .supported
    )

    invalid = NoiseModel()
    invalid.add(fq.noise.ReadoutConfusion(np.eye(3)))
    with pytest.raises(BackendValidationError, match="2 x 2"):
        _backend(
            atom_3level_model,
            noise=invalid,
            lindblad_map=_lindblad_map(),
        )


def test_custom_rule_must_return_local_three_by_three_operators(
    atom_3level_model,
):
    def wrong_shape(_channel, **_kwargs):
        return (np.eye(2),)

    backend = _backend(
        atom_3level_model,
        noise=_noise(PhaseDamping(rate=0.2), operation=fq.ops.RX),
        lindblad_map=_lindblad_map(wrong_shape),
    )
    with pytest.raises(BackendValidationError, match=r"expected \(3, 3\)"):
        backend._prepare_program(_rx_program())


def test_operation_scoped_terms_bind_qutrit_ordinals_and_change_output(
    atom_3level_model,
):
    noise = _noise(PhaseDamping(rate=0.4), operation=fq.ops.RX)
    noisy = _backend(
        atom_3level_model,
        noise=noise,
        lindblad_map=_lindblad_map(),
    )
    program = _rx_program()
    prepared = noisy._prepare_program(program)
    term = prepared.plan[0].noise[0]
    result = noisy.run(program).result()
    ideal = _backend(atom_3level_model).run(program).result()

    assert term.local_operator.shape == (3, 3)
    assert term.engine_indices == (0,)
    assert result.metadata["solver"]["solver"] == "mesolve"
    assert not np.allclose(result.get_density_matrix(), ideal.get_density_matrix())


def test_background_terms_are_resolved_and_executed(atom_3level_model):
    backend = _backend(
        atom_3level_model,
        noise=_noise(PhaseDamping(rate=0.4)),
        lindblad_map=_lindblad_map(),
    )
    program = _rx_program()
    prepared = backend._prepare_program(program)
    result = backend.run(program).result()

    assert len(prepared.background_noise) == 1
    assert prepared.background_noise[0].engine_indices == (0,)
    assert result.metadata["solver"]["solver"] == "mesolve"


def test_operation_scoped_collapse_uses_the_scheduled_block_window(
    atom_3level_model, monkeypatch
):
    backend = _backend(
        atom_3level_model,
        noise=_noise(PhaseDamping(rate=0.4), operation=fq.ops.RX),
        lindblad_map=_lindblad_map(),
    )
    program = fq.Program(1)
    program.add(
        fq.ops.PulseOperation(
            0.2,
            (
                PulseControl(
                    atom_3level_model.control.raman(0),
                    SampledWaveform((0.0, 0.2), (0.0, 0.0)),
                ),
            ),
        )
    )
    program.add(fq.ops.RX(np.pi / 3), 0)
    prepared = backend._prepare_program(program)
    scheduled = schedule_pulse_run(prepared.plan, boundary_time=0.0)
    start = scheduled.starts[1]
    end = start + prepared.plan[1].duration
    captured = []

    def record(*args, c_ops=(), **kwargs):
        captured.append(tuple(c_ops))
        return mesolve(*args, c_ops=c_ops, **kwargs)

    monkeypatch.setattr(
        "fatqat.emulator.atom_3level.qutip_adapter.mesolve",
        record,
    )
    backend.run(program).result()

    assert len(captured) == 1 and len(captured[0]) == 1
    assert captured[0][0](start / 2).norm() == pytest.approx(0.0)
    assert captured[0][0]((start + end) / 2).norm() > 0.0
    assert captured[0][0](end).norm() == pytest.approx(0.0)


def test_propagator_rejects_elapsed_noise_but_frame_only_skips_dissipation(
    atom_3level_model, monkeypatch
):
    noise = _noise(PhaseDamping(rate=0.4))
    backend = _backend(
        atom_3level_model,
        noise=noise,
        lindblad_map=_lindblad_map(),
    )
    with pytest.raises(BackendValidationError, match="dissipative Lindblad"):
        backend.propagator(_rx_program())

    def dissipation_must_not_be_built(*_args, **_kwargs):
        raise AssertionError("frame-only propagation constructed collapse operators")

    monkeypatch.setattr(
        "fatqat.emulator.atom_3level.qutip_adapter."
        "_Atom3LevelQutipAdapter._expand_collapse_terms",
        dissipation_must_not_be_built,
    )
    monkeypatch.setattr(
        "fatqat.emulator.atom_3level.qutip_adapter.mesolve",
        dissipation_must_not_be_built,
    )
    monkeypatch.setattr(
        "fatqat.emulator.atom_3level.qutip_adapter.qutip_propagator",
        dissipation_must_not_be_built,
    )
    frame_only = fq.Program(1)
    frame_only.add(fq.ops.RZ(0.3), 0)
    actual = backend.propagator(frame_only)

    assert np.allclose(actual, np.diag((1.0, np.exp(0.3j), 1.0)))
