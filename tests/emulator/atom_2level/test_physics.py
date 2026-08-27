"""Independent coherent-physics checks for two-level execution."""

import json
from pathlib import Path

import numpy as np
import pytest
from qutip import basis, tensor

from fatqat.emulator import AtomArrangement
from fatqat._pulse_values import PulseControl
from fatqat._index_allocation import _EngineAllocation
from fatqat.emulator._core.engine import PulseEngine, _ShotContext
from fatqat.emulator._core.pulse import PulseBlock
from fatqat.emulator._core.target import _PreparedControlBinding
from fatqat.emulator._core.scheduling import schedule_pulse_run
from fatqat.emulator.atom_2level import Atom2LevelModel
from fatqat.emulator.atom_2level.qutip_adapter import _Atom2LevelQutipAdapter
from fatqat.emulator.atom_2level.target import _Atom2LevelTarget
from fatqat.emulator import SampledWaveform

from tests.emulator.atom_2level.reference.two_level_hamiltonian import (
    solve_constant,
    solve_sampled,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "atom_2level_reference.json"


def _target(site_count=1, *, c6=0.0, interaction_cutoff=2.0, spacing=2.0):
    document = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    document["parameters"]["c6"] = c6
    return _Atom2LevelTarget(
        Atom2LevelModel.from_document(document),
        AtomArrangement.rectangular(1, site_count, spacing),
        interaction_cutoff,
    )


def _block(target, duration, *, amplitude, detuning=0.0, phase=0.0):
    def sampled(value, factor=1.0):
        if isinstance(value, SampledWaveform):
            return SampledWaveform(
                value.times, tuple(sample * factor for sample in value.values)
            )
        return SampledWaveform((0.0, duration), (value * factor, value * factor))

    controls = (
        PulseControl(
            target.model.control.drive(),
            sampled(amplitude, np.exp(1j * phase)),
        ),
        PulseControl(target.model.control.detuning(), sampled(detuning)),
    )
    target_bindings = tuple(
        target.bind_control(control.channel) for control in controls
    )
    bindings = tuple(
        _PreparedControlBinding(
            binding.kind,
            tuple(
                target.device_labels.index(value) for value in binding.device_operands
            ),
        )
        for binding in target_bindings
    )
    return PulseBlock(
        duration=duration,
        controls=controls,
        control_bindings=bindings,
        resource_claims=tuple(
            dict.fromkeys(
                claim for binding in target_bindings for claim in binding.claims
            )
        ),
        post_actions=(),
        condition=None,
        target_indices=tuple(range(len(target.device_labels))),
    )


def _adapter(target):
    return _Atom2LevelQutipAdapter(
        target,
        engine_allocation=_EngineAllocation(
            target.device_labels, (2,) * len(target.device_labels)
        ),
    )


def _evolve(target, block):
    adapter = _adapter(target)
    outcome = PulseEngine(adapter).run(
        (block,), shots=1, n_clbits=0, rng=np.random.default_rng(7)
    )[0]
    return outcome.final_state


@pytest.mark.parametrize("phase", [0.0, np.pi / 2, 0.371])
def test_one_atom_resonant_rabi_oscillation_has_positive_phase_exponent(phase):
    target = _target()
    amplitude = 1.7
    duration = 0.83
    state = _evolve(
        target,
        _block(target, duration, amplitude=amplitude, phase=phase),
    )
    expected = np.asarray(
        [
            np.cos(amplitude * duration / 2),
            -1j * np.exp(1j * phase) * np.sin(amplitude * duration / 2),
        ]
    )

    assert state == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("detuning", [-1.2, 0.0, 0.9])
def test_one_atom_signed_detuning_matches_dense_oracle(detuning):
    target = _target()
    values = {"amplitude": 1.1, "detuning": detuning, "phase": 0.29}
    duration = 0.71
    state = _evolve(target, _block(target, duration, **values))
    expected = solve_constant(1, duration, interactions=(), **values)

    assert state == pytest.approx(expected, abs=2e-9)


def test_zero_controls_still_advance_a_nonzero_interval():
    target = _target()
    state = _evolve(target, _block(target, 0.75, amplitude=0.0, detuning=0.0))
    assert state == pytest.approx(np.asarray([1.0, 0.0]))


def test_two_atoms_factorize_when_c6_is_zero():
    target = _target(site_count=2, c6=0.0)
    values = {"amplitude": 1.2, "detuning": -0.4, "phase": 0.21}
    duration = 0.63
    state = _evolve(target, _block(target, duration, **values))
    one_atom = solve_constant(1, duration, interactions=(), **values)

    assert state == pytest.approx(np.kron(one_atom, one_atom), abs=3e-9)


@pytest.mark.parametrize("c6", [-32.0, 48.0])
def test_signed_two_atom_interaction_matches_independent_dense_oracle(c6):
    spacing = 2.0
    target = _target(site_count=2, c6=c6, spacing=spacing)
    values = {"amplitude": 1.3, "detuning": 0.2, "phase": -0.31}
    duration = 0.57
    state = _evolve(target, _block(target, duration, **values))
    expected = solve_constant(
        2,
        duration,
        interactions=((0, 1, c6 / spacing**6),),
        **values,
    )

    assert state == pytest.approx(expected, abs=4e-9)


def test_spacing_cutoff_omits_long_range_pair_and_no_cutoff_restores_it():
    c6 = 64.0
    spacing = 2.0
    nearest_target = _target(
        site_count=3,
        c6=c6,
        spacing=spacing,
        interaction_cutoff=spacing,
    )
    full_target = _target(
        site_count=3,
        c6=c6,
        spacing=spacing,
        interaction_cutoff=None,
    )
    nearest = _adapter(nearest_target)
    full = _adapter(full_target)
    initial = tensor(basis(2, 1), basis(2, 0), basis(2, 1))
    duration = 0.8

    def evolve_zero(adapter, target):
        run = schedule_pulse_run(
            (_block(target, duration, amplitude=0.0),), boundary_time=0.0
        )
        context = _ShotContext(initial.copy(), [], np.random.default_rng(5))
        adapter.evolve(run, context, (True,))
        return np.asarray(context.state.full()).reshape(-1)

    nearest_state = evolve_zero(nearest, nearest_target)
    full_state = evolve_zero(full, full_target)
    long_range_strength = c6 / (2 * spacing) ** 6
    expected_phase = np.exp(-1j * long_range_strength * duration)

    assert nearest_state == pytest.approx(np.asarray(initial.full()).reshape(-1))
    assert full_state[5] == pytest.approx(expected_phase)
    assert full_state != pytest.approx(nearest_state)


def test_sampled_effective_degree_execution_matches_independent_ode_oracle():
    target = _target(site_count=2, c6=16.0, spacing=2.0)
    duration = 1.2
    amplitude = SampledWaveform(
        (0.0, 0.17, 0.66, 1.2),
        (0.2, 0.285, 0.53, 0.8),
    )
    detuning = SampledWaveform(
        (0.0, 0.41, 1.2),
        (-0.2, 0.7, 0.1),
    )
    phase = 0.23
    state = _evolve(
        target,
        _block(
            target,
            duration,
            amplitude=amplitude,
            detuning=detuning,
            phase=phase,
        ),
    )
    expected = solve_sampled(
        2,
        duration,
        amplitude_times=amplitude.times,
        amplitude_values=amplitude.values,
        detuning_times=detuning.times,
        detuning_values=detuning.values,
        phase=phase,
        interactions=((0, 1, 16.0 / 2.0**6),),
    )

    assert state == pytest.approx(expected, abs=8e-8)


def test_time_varying_complex_drive_phase_matches_independent_ode_oracle():
    target = _target(site_count=1)
    duration = 1.0
    drive = SampledWaveform(
        (0.0, 0.25, 0.7, 1.0),
        (
            0.2,
            0.35 * np.exp(0.4j),
            0.5 * np.exp(-0.6j),
            0.3j,
        ),
    )
    detuning = SampledWaveform((0.0, 0.4, 1.0), (-0.2, 0.1, 0.0))

    state = _evolve(
        target,
        _block(target, duration, amplitude=drive, detuning=detuning),
    )
    expected = solve_sampled(
        1,
        duration,
        amplitude_times=drive.times,
        amplitude_values=drive.values,
        detuning_times=detuning.times,
        detuning_values=detuning.values,
        phase=0.0,
        interactions=(),
    )

    assert state == pytest.approx(expected, abs=8e-8)
