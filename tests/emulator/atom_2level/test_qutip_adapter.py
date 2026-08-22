"""Private coherent two-level QuTiP adapter contracts."""

import json
from pathlib import Path

import numpy as np
import pytest
from qutip import Qobj, basis, tensor

import fatqat.emulator.atom_2level.qutip_adapter as atom2_qutip_adapter
from fatqat import AtomArrangement
from fatqat._backends.steps import MeasurementStep, ResetStep
from fatqat._index_allocation import _EngineAllocation
from fatqat._pulse_values import PulseControl
from fatqat.emulator._core.engine import PulseEngine, _ShotContext
from fatqat.emulator._core.pulse import PulseBlock
from fatqat.emulator._core.target import _PreparedControlBinding
from fatqat.emulator._core.scheduling import schedule_pulse_run
from fatqat.emulator.atom_2level import Atom2LevelModel
from fatqat.emulator.atom_2level.qutip_adapter import _Atom2LevelQutipAdapter
from fatqat.emulator.atom_2level.target import _Atom2LevelTarget
from fatqat.errors import BackendValidationError
from fatqat.waveforms import SampledWaveform

_FIXTURE = Path(__file__).parent / "fixtures" / "atom_2level_reference.json"


def _target(site_count=2, *, c6=1.0, interaction_cutoff=2.0):
    document = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    document["parameters"]["c6"] = c6
    return _Atom2LevelTarget(
        Atom2LevelModel.from_document(document),
        AtomArrangement.rectangular(1, site_count, 2.0),
        interaction_cutoff,
    )


def _adapter(target, **kwargs):
    return _Atom2LevelQutipAdapter(
        target,
        engine_allocation=_EngineAllocation(
            target.device_labels, (2,) * len(target.device_labels)
        ),
        **kwargs,
    )


def _block(target, duration=1.0, **components):
    amplitude = components.pop("amplitude", 0.0)
    detuning = components.pop("detuning", 0.0)
    phase = components.pop("phase", 0.0)
    if components:
        raise AssertionError(f"unknown test components: {tuple(components)}")
    controls = (
        PulseControl(
            target.model.control.drive(),
            SampledWaveform(
                (0.0, duration),
                (amplitude * np.exp(1j * phase),) * 2,
            ),
        ),
        PulseControl(
            target.model.control.detuning(),
            SampledWaveform((0.0, duration), (detuning, detuning)),
        ),
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


def test_initial_ket_copy_and_finished_statevector_are_binary_and_independent():
    adapter = _adapter(_target())
    initial = adapter.initial_state()
    copied = adapter.copy_state(initial)
    context = _ShotContext(initial, [], np.random.default_rng(1))
    outcome = adapter.finish_shot(context)

    assert isinstance(initial, Qobj)
    assert initial.isket
    assert initial.shape == (4, 1)
    assert initial.full().reshape(-1).tolist() == [1.0, 0.0, 0.0, 0.0]
    assert copied is not initial
    assert outcome.final_state_kind == "statevector"
    assert outcome.final_state.shape == (4,)
    assert np.linalg.norm(outcome.final_state) == pytest.approx(1.0)
    outcome.final_state[0] = 0.0
    assert initial.full()[0, 0] == 1.0


def test_counts_only_finish_does_not_materialize_a_statevector():
    adapter = _adapter(_target(), retain_final_state=False)

    def statevector_must_not_be_called(_state):
        raise AssertionError("counts-only completion materialized the ket")

    adapter._statevector = statevector_must_not_be_called
    context = _ShotContext(adapter.initial_state(), [1], np.random.default_rng(1))

    outcome = adapter.finish_shot(context)

    assert outcome.final_state is None
    assert outcome.final_state_kind == "statevector"
    assert outcome.classical_digits == (1,)


def test_measurement_indices_are_engine_tensor_axes():
    adapter = _adapter(_target())
    context = _ShotContext(
        tensor(basis(2, 1), basis(2, 0)),
        [0, 0],
        np.random.default_rng(2),
    )
    step = MeasurementStep(
        (0, 1),
        (0, 1),
        reported_digit_maps=((0, 1), (0, 1)),
    )

    adapter.execute_boundary(step, context)

    assert context.classical_memory == [1, 0]
    assert np.asarray(context.state.full()).reshape(-1).tolist() == [0.0, 0.0, 1.0, 0.0]


def test_engine_allocation_must_match_the_complete_target():
    with pytest.raises(BackendValidationError, match="complete target"):
        _Atom2LevelQutipAdapter(
            _target(), engine_allocation=_EngineAllocation((0,), (2,))
        )


def test_terminal_measurement_uses_supplied_rng_and_collapses_the_complete_ket():
    adapter = _adapter(_target(site_count=1))
    superposition = (basis(2, 0) + basis(2, 1)).unit()
    step = MeasurementStep((0,), (0,), reported_digit_maps=((0, 1),))

    def sample(seed):
        context = _ShotContext(superposition.copy(), [0], np.random.default_rng(seed))
        adapter.execute_boundary(step, context)
        return context.classical_memory[0], adapter.finish_shot(context).final_state

    first_digit, first_state = sample(2026)
    second_digit, second_state = sample(2026)
    assert first_digit == second_digit
    assert first_state == pytest.approx(second_state)
    expected = np.asarray([1.0, 0.0]) if first_digit == 0 else np.asarray([0.0, 1.0])
    assert first_state == pytest.approx(expected)


def test_adapter_rejects_reset_boundary():
    adapter = _adapter(_target(site_count=1))
    context = _ShotContext(adapter.initial_state(), [], np.random.default_rng(0))
    with pytest.raises(BackendValidationError, match="does not support reset"):
        adapter.execute_boundary(ResetStep((0,)), context)


def test_propagator_and_engine_outputs_have_binary_shapes():
    target = _target()
    block = _block(target, duration=0.4, amplitude=1.3, detuning=-0.2)
    adapter = _adapter(target)
    run = schedule_pulse_run((block,), boundary_time=0.0)
    unitary = adapter.propagator(run)
    outcomes = PulseEngine(adapter).run(
        (block,), shots=1, n_clbits=0, rng=np.random.default_rng(3)
    )

    assert unitary.shape == (4, 4)
    assert outcomes[0].final_state.shape == (4,)
    assert np.linalg.norm(outcomes[0].final_state) == pytest.approx(1.0)


def test_density_evolution_validates_state_without_copying_it(monkeypatch):
    target = _target()
    adapter = _adapter(target, execution_mode="density_matrix")
    block = _block(target, duration=0.1, amplitude=0.0)
    run = schedule_pulse_run((block,), boundary_time=0.0)
    context = _ShotContext(adapter.initial_state(), [], np.random.default_rng(4))

    def copy_must_not_be_called(_state):
        raise AssertionError("evolve copied its input only to validate it")

    monkeypatch.setattr(adapter, "copy_state", copy_must_not_be_called)
    adapter.evolve(run, context, (True,))

    assert context.state.isoper


def test_interaction_drift_is_one_sparse_diagonal_operator():
    adapter = _adapter(_target(site_count=3, c6=64.0))
    drift = adapter.interaction_drift

    assert drift.isherm
    assert drift.data.__class__.__name__ in {"CSR", "Dia"}
    assert np.count_nonzero(drift.full() - np.diag(np.diag(drift.full()))) == 0


def test_windowed_coefficient_uses_qutip_sampled_array_interpolation(monkeypatch):
    target = _target(site_count=1)
    child = PulseControl(
        target.model.control.drive(),
        SampledWaveform(
            (0.0, 0.2, 0.8, 1.0),
            (0.1j, 0.2 + 0.3j, -0.4j, 0.2),
        ),
        start_offset=0.3,
    )
    qutip_coefficient = atom2_qutip_adapter.coefficient
    calls = []

    def traced_coefficient(base, **kwargs):
        calls.append((base, kwargs))
        return qutip_coefficient(base, **kwargs)

    monkeypatch.setattr(atom2_qutip_adapter, "coefficient", traced_coefficient)
    coefficient = _Atom2LevelQutipAdapter._windowed_coefficient(child, 0.4)

    samples, options = calls[0]
    absolute_times = np.asarray(child.waveform.times) + 0.7
    assert np.asarray(samples) == pytest.approx(child.waveform.values)
    assert np.asarray(options["tlist"]) == pytest.approx(absolute_times)
    assert options["order"] == 3
    assert coefficient(0.69) == 0.0j
    assert coefficient(1.7) == pytest.approx(0.2)
