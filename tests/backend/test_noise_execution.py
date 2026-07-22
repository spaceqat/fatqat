"""End-to-end channel noise: lowering, path classification, DM/SV semantics."""

import numpy as np
import pytest

import fatqat as fq
from fatqat.backends import ApplyChannelStep, ApplyMatrixStep, SimulatorBackend
from fatqat.errors import BackendValidationError, UnsupportedOperationError
from fatqat.noise import (
    Channel,
    Depolarizing,
    NoiseModel,
    default_channel_implementation_map,
)
from fatqat.simulator.np import NumpyDMSimulator, NumpySVSimulator


def _depolarized_x_model(p=0.2):
    noise = NoiseModel()
    noise.add_noise(fq.ops.X, Depolarizing(p=p))
    return noise


def _x_program(with_measurement=False):
    program = fq.Program(1, 1 if with_measurement else 0)
    program.add(fq.ops.X, 0)
    if with_measurement:
        program.add_measurement(0, 0)
    return program


# --- lowering ---


def test_channel_lowered_right_after_its_gate_with_same_targets():
    backend = SimulatorBackend(noise=_depolarized_x_model())
    program = _x_program()
    plan, facts = backend._lower(program, backend.resolve_layout(program))

    assert isinstance(plan[0], ApplyMatrixStep)
    assert isinstance(plan[1], ApplyChannelStep)
    assert plan[1].target_indices == plan[0].target_indices
    assert len(plan[1].kraus_ops) == 4
    assert all(not k.flags.writeable for k in plan[1].kraus_ops)
    assert facts.has_channel is True


def test_channel_inherits_the_gate_condition():
    backend = SimulatorBackend(noise=_depolarized_x_model())
    program = fq.Program(1, 1)
    program.add(fq.ops.X, 0, condition=(0, 1))
    plan, _ = backend._lower(program, backend.resolve_layout(program))

    assert isinstance(plan[1], ApplyChannelStep)
    assert plan[1].condition == plan[0].condition
    assert plan[1].condition is not None


def test_noise_free_backend_lowers_no_channel_steps():
    backend = SimulatorBackend()
    program = _x_program()
    plan, facts = backend._lower(program, backend.resolve_layout(program))

    assert all(not isinstance(s, ApplyChannelStep) for s in plan)
    assert facts.has_channel is False


def test_unresolvable_channel_type_raises():
    class Leakage(Channel):
        pass

    noise = NoiseModel()
    noise.add_noise(fq.ops.X, Leakage())
    backend = SimulatorBackend(noise=noise)
    program = _x_program()
    with pytest.raises(UnsupportedOperationError, match="Leakage"):
        backend._lower(program, backend.resolve_layout(program))


def test_mis_shaped_rule_rejected_but_non_cptp_accepted():
    # Shape is checked at resolution; trace preservation deliberately is not
    # (the same posture as gate matrices, which are never unitarity-checked).
    class Custom(Channel):
        pass

    channel_map = default_channel_implementation_map()
    channel_map.register(
        Custom, lambda channel, *, targets: (np.eye(3, dtype=complex),)
    )
    noise = NoiseModel()
    noise.add_noise(fq.ops.X, Custom())
    backend = SimulatorBackend(noise=noise, channel_implementation_map=channel_map)
    program = _x_program()
    with pytest.raises(BackendValidationError, match="shape"):
        backend._lower(program, backend.resolve_layout(program))

    channel_map.register(
        Custom, lambda channel, *, targets: (0.5 * np.eye(2, dtype=complex),)
    )
    backend = SimulatorBackend(noise=noise, channel_implementation_map=channel_map)
    plan, _ = backend._lower(program, backend.resolve_layout(program))
    assert any(isinstance(step, ApplyChannelStep) for step in plan)


def test_viewed_gate_resolves_a_channel_per_expanded_member():
    # Channel resolution moved inside the per-emission loop: a viewed rotation
    # over N members with an attached channel emits N (matrix, channel) pairs,
    # each channel carrying the member's own engine index.
    from fatqat.backends.resource_binding import BoundResource, ResourceBinding
    from fatqat.backends.resource_binding import _scalar_identity_binder
    from fatqat.registers import GridRegister, RegisterView

    def view_binder(target, flat_layout):
        if not isinstance(target, RegisterView):
            return None
        reg = target.register  # RowSelector on row 0 -> increasing column
        row = target.selector.row
        members = [reg[row * reg.cols + c] for c in range(reg.cols)]
        return tuple(
            BoundResource(
                ref=m,
                engine_index=flat_layout.subsystem_index(m),
                device_label=flat_layout.subsystem_index(m),
            )
            for m in members
        )

    atoms = GridRegister(2, 3, name="atoms")
    program = fq.Program([atoms])
    noise = NoiseModel()
    noise.add_noise(fq.ops.RX, Depolarizing(p=0.1))
    backend = SimulatorBackend(noise=noise)
    binding = ResourceBinding([view_binder, _scalar_identity_binder])
    program.add(fq.ops.RX(0.3), atoms.row(0))  # members at engine indices 0,1,2
    plan, facts = backend._lower(program, backend.resolve_layout(program), binding)

    matrix_steps = [s for s in plan if isinstance(s, ApplyMatrixStep)]
    channel_steps = [s for s in plan if isinstance(s, ApplyChannelStep)]
    assert [s.target_indices for s in matrix_steps] == [(0,), (1,), (2,)]
    assert [s.target_indices for s in channel_steps] == [(0,), (1,), (2,)]
    # Interleaved matrix-then-channel per member, not batched.
    assert [type(s).__name__ for s in plan] == [
        "ApplyMatrixStep",
        "ApplyChannelStep",
    ] * 3
    assert facts.has_channel is True


def test_reset_attached_channels_raise_until_wired():
    noise = NoiseModel()
    noise.add_noise(fq.ops.Reset, Depolarizing(p=0.1))
    backend = SimulatorBackend(noise=noise)
    program = fq.Program(1)
    program.add(fq.ops.Reset, 0)
    with pytest.raises(UnsupportedOperationError, match="Reset"):
        backend._lower(program, backend.resolve_layout(program))


# --- path classification ---


def test_unconditional_channel_keeps_density_matrix_on_fast_path():
    backend = SimulatorBackend(method="DM", noise=_depolarized_x_model())
    program = _x_program(with_measurement=True)
    plan, _ = backend._lower(program, backend.resolve_layout(program))

    assert NumpyDMSimulator()._analyze_plan(plan)[0] is False


def test_channel_forces_statevector_onto_dynamic_path():
    backend = SimulatorBackend(method="SV", noise=_depolarized_x_model())
    program = _x_program(with_measurement=True)
    plan, _ = backend._lower(program, backend.resolve_layout(program))

    assert NumpySVSimulator()._analyze_plan(plan)[0] is True


def test_statevector_export_with_noise_requires_single_shot():
    backend = SimulatorBackend(method="SV", noise=_depolarized_x_model())
    program = _x_program()
    with pytest.raises(BackendValidationError, match="channel noise"):
        backend.run(
            program,
            shots=4,
            result_config={"counts": False, "statevector": True},
        )
    result = backend.run(
        program,
        shots=1,
        seed=3,
        result_config={"counts": False, "statevector": True},
    ).result()
    assert np.isclose(np.linalg.norm(result.get_statevector()), 1.0)


# --- execution semantics ---


def test_density_matrix_channel_is_exact():
    p = 0.2
    backend = SimulatorBackend(method="DM", noise=_depolarized_x_model(p))
    result = backend.run(
        _x_program(),
        result_config={"counts": False, "density_matrix": True},
    ).result()

    expected = (1 - p) * np.diag([0.0, 1.0]) + p * np.eye(2) / 2
    assert np.allclose(result.get_density_matrix(), expected)


def test_statevector_trajectories_match_density_matrix_statistics():
    p = 0.2
    shots = 4000
    program = _x_program(with_measurement=True)
    counts = (
        SimulatorBackend(method="SV", noise=_depolarized_x_model(p))
        .run(program, shots=shots, seed=7)
        .result()
        .get_counts()
    )

    # X then depolarizing: P(1) = 1 - p/2.
    assert abs(counts.get("1", 0) / shots - (1 - p / 2)) < 0.02


def test_density_matrix_counts_sample_the_noisy_distribution():
    p = 0.2
    shots = 4000
    program = _x_program(with_measurement=True)
    counts = (
        SimulatorBackend(method="DM", noise=_depolarized_x_model(p))
        .run(program, shots=shots, seed=7)
        .result()
        .get_counts()
    )

    assert abs(counts.get("1", 0) / shots - (1 - p / 2)) < 0.02


def test_skipped_conditioned_gate_skips_its_channel():
    # q0 measures 0 deterministically, so the conditioned X (and its noise)
    # must not fire: q1 stays exactly |0><0|.
    noise = _depolarized_x_model(p=0.5)
    program = fq.Program(2, 1)
    program.add_measurement(0, 0)
    program.add(fq.ops.X, 1, condition=(0, 1))
    result = (
        SimulatorBackend(method="DM", noise=noise)
        .run(program, shots=1, seed=5, result_config={"density_matrix": True})
        .result()
    )

    rho = result.get_density_matrix()
    expected = np.zeros((4, 4), dtype=complex)
    expected[0, 0] = 1.0
    assert np.allclose(rho, expected)


def test_taken_conditioned_gate_applies_its_channel():
    # q0 is flipped first, so c reads 1 and the conditioned X on q1 fires,
    # depolarizing q1: rho_q1 = (1-p)|1><1| + p I/2 while q0 stays |1>.
    p = 0.5
    noise = _depolarized_x_model(p)
    program = fq.Program(2, 1)
    program.add(fq.ops.X, 0)
    program.add_measurement(0, 0)
    program.add(fq.ops.X, 1, condition=(0, 1))
    result = (
        SimulatorBackend(method="DM", noise=noise)
        .run(program, shots=1, seed=5, result_config={"density_matrix": True})
        .result()
    )

    rho = result.get_density_matrix()
    # Little-endian: q0 is the least-significant digit. q0=1 fixed, q1 mixed.
    expected = np.zeros((4, 4), dtype=complex)
    expected[1, 1] = p / 2  # q1=0, q0=1
    expected[3, 3] = 1 - p / 2  # q1=1, q0=1
    assert np.allclose(rho, expected)


def test_seeded_noisy_runs_are_reproducible():
    backend = SimulatorBackend(method="SV", noise=_depolarized_x_model())
    program = _x_program(with_measurement=True)
    first = backend.run(program, shots=64, seed=11).result().get_counts()
    second = backend.run(program, shots=64, seed=11).result().get_counts()

    assert first == second


def test_parallel_dynamic_shots_match_serial_with_channels():
    noise = _depolarized_x_model()
    program = _x_program(with_measurement=True)
    serial = (
        SimulatorBackend(method="SV", options={"parallel_mode": "serial"}, noise=noise)
        .run(program, shots=8, seed=13)
        .result()
        .get_counts()
    )
    parallel = (
        SimulatorBackend(method="SV", options={"max_workers": 2}, noise=noise)
        .run(program, shots=8, seed=13)
        .result()
        .get_counts()
    )

    assert parallel == serial


# --- validate_noise ---


def test_validate_noise_accepts_catalog_channels():
    report = SimulatorBackend().validate_noise(_depolarized_x_model())

    assert report.supported is True
    assert report.accepted_sources == ("Depolarizing",)
    assert report.rejected_sources == ()


def test_validate_noise_rejects_unknown_channel_and_qubit_noise_and_reset():
    class Leakage(Channel):
        pass

    noise = NoiseModel()
    noise.add_noise(fq.ops.X, Leakage())
    noise.add_noise(fq.ops.Reset, Depolarizing(p=0.1))
    noise.qubit_noise[("q", 0)] = object()
    report = SimulatorBackend().validate_noise(noise)

    assert report.supported is False
    assert set(report.rejected_sources) == {"Leakage", "qubit_noise", "Reset"}
    assert "Depolarizing" in report.accepted_sources
    assert len(report.warnings) == 3


def test_numba_simulator_falls_back_correctly_on_channel_plans():
    # The fused numba dynamic kernel only understands matrix/measure/reset
    # steps; a channel-bearing plan must take the inherited NumPy per-shot
    # path and produce identical counts, never reach the compiler.
    pytest.importorskip("numba")
    from fatqat.simulator.nb import NumbaSVSimulator, _plan_compilable

    noise = NoiseModel()
    noise.add_noise(fq.ops.X, Depolarizing(p=0.3))
    backend = SimulatorBackend(noise=noise)
    program = fq.Program(1, 1)
    program.add(fq.ops.X, 0)
    program.add_measurement(0, 0)
    plan, _ = backend._lower(program, backend.resolve_layout(program))
    assert _plan_compilable(plan) is False

    # Below the auto-parallel floor so both simulators run in-process serial.
    request = backend._request_cls(counts=True, statevector=False)
    counts = {}
    for cls in (NumpySVSimulator, NumbaSVSimulator):
        simulator = cls()
        simulator.initialize((2,), 1)
        raw = simulator.run(plan, 20, 7, request)
        counts[cls.__name__] = list(
            zip(raw.outcome_keys.tolist(), raw.outcome_counts.tolist())
        )
    assert counts["NumpySVSimulator"] == counts["NumbaSVSimulator"]
