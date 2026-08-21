"""End-to-end channel noise: lowering, path classification, DM/SV semantics."""

import numpy as np
import pytest

import fatqat as fq
from fatqat._backends.steps import ApplyChannelStep, ApplyMatrixStep
from fatqat.simulator import SCQubitIBMSimulator, Simulator
from fatqat.errors import BackendValidationError
from fatqat.noise import (
    AmplitudeDamping,
    Channel,
    Depolarizing,
    NoiseModel,
    PhaseDamping,
    ThermalRelaxation,
    default_channel_implementation_map,
)
from fatqat.simulator._engine.np import NumpyDMEngine, NumpySVEngine


def _total_variation(counts_a, counts_b, shots):
    """Total-variation distance between two count dicts, as a fraction of shots."""
    keys = set(counts_a) | set(counts_b)
    diff = sum(abs(counts_a.get(k, 0) - counts_b.get(k, 0)) for k in keys)
    return 0.5 * diff / shots


def _depolarized_x_model(p=0.2):
    noise = NoiseModel()
    noise.add(Depolarizing(p=p), operation=fq.ops.X)
    return noise


def _x_program(with_measurement=False):
    program = fq.Program(1, 1 if with_measurement else 0)
    program.add(fq.ops.X, 0)
    if with_measurement:
        program.measure(0, 0)
    return program


# --- lowering ---


def test_channel_lowered_right_after_its_gate_with_same_targets():
    backend = Simulator(noise=_depolarized_x_model())
    program = _x_program()
    plan, facts = backend._lower_program(program)

    assert isinstance(plan[0], ApplyMatrixStep)
    assert isinstance(plan[1], ApplyChannelStep)
    assert plan[1].target_indices == plan[0].target_indices
    assert len(plan[1].kraus_ops) == 4
    assert all(not k.flags.writeable for k in plan[1].kraus_ops)
    assert facts.has_channel is True


def test_channel_inherits_the_gate_condition():
    backend = Simulator(noise=_depolarized_x_model())
    program = fq.Program(1, 1)
    program.add(fq.ops.X, 0, condition=(0, 1))
    plan, _ = backend._lower_program(program)

    assert isinstance(plan[1], ApplyChannelStep)
    assert plan[1].condition == plan[0].condition
    assert plan[1].condition is not None


def test_noise_free_backend_lowers_no_channel_steps():
    backend = Simulator()
    program = _x_program()
    plan, facts = backend._lower_program(program)

    assert all(not isinstance(s, ApplyChannelStep) for s in plan)
    assert facts.has_channel is False


def test_unresolvable_channel_type_rejects_at_construction():
    class Leakage(Channel):
        pass

    noise = NoiseModel()
    noise.add(Leakage(), operation=fq.ops.X)
    with pytest.raises(BackendValidationError, match="Leakage"):
        Simulator(noise=noise)


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
    noise.add(Custom(), operation=fq.ops.X)
    backend = Simulator(noise=noise, channel_implementation_map=channel_map)
    program = _x_program()
    with pytest.raises(BackendValidationError, match="shape"):
        backend._lower_program(program)

    channel_map.register(
        Custom, lambda channel, *, targets: (0.5 * np.eye(2, dtype=complex),)
    )
    backend = Simulator(noise=noise, channel_implementation_map=channel_map)
    plan, _ = backend._lower_program(program)
    assert any(isinstance(step, ApplyChannelStep) for step in plan)


def test_viewed_gate_resolves_a_channel_per_expanded_member():
    # Channel resolution moved inside the per-emission loop: a viewed rotation
    # over N members with an attached channel emits N (matrix, channel) pairs,
    # each channel carrying the member's own engine index.
    from fatqat.registers import GridRegister

    atoms = GridRegister(2, 3, name="atoms")
    program = fq.Program([atoms])
    noise = NoiseModel()
    noise.add(Depolarizing(p=0.1), operation=fq.ops.RX)
    backend = Simulator(noise=noise)
    program.add(fq.ops.RX(0.3), atoms.row(0))  # members at engine indices 0,1,2
    plan, facts = backend._lower_program(program)

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


def test_reset_attached_channels_reject_at_admission():
    noise = NoiseModel()
    with pytest.raises(ValueError, match="Reset"):
        noise.add(Depolarizing(p=0.1), operation=fq.ops.Reset)


# --- path classification ---


def test_unconditional_channel_keeps_density_matrix_on_fast_path():
    backend = Simulator(method="DM", noise=_depolarized_x_model())
    program = _x_program(with_measurement=True)
    plan, _ = backend._lower_program(program)

    assert NumpyDMEngine()._analyze_plan(plan)[0] is False


def test_channel_forces_statevector_onto_dynamic_path():
    backend = Simulator(method="SV", noise=_depolarized_x_model())
    program = _x_program(with_measurement=True)
    plan, _ = backend._lower_program(program)

    assert NumpySVEngine()._analyze_plan(plan)[0] is True


def test_statevector_export_with_noise_requires_single_shot():
    backend = Simulator(method="SV", noise=_depolarized_x_model())
    program = _x_program()
    with pytest.raises(BackendValidationError, match="stochastic execution"):
        backend.run(
            program,
            shots=4,
            result_config={"counts": False, "final_state": True},
        )
    result = backend.run(
        program,
        shots=1,
        simulation_config={"seed": 3},
        result_config={"counts": False, "final_state": True},
    ).result()
    assert np.isclose(np.linalg.norm(result.get_statevector()), 1.0)


# --- execution semantics ---


def test_density_matrix_channel_is_exact():
    p = 0.2
    backend = Simulator(method="DM", noise=_depolarized_x_model(p))
    result = backend.run(
        _x_program(),
        result_config={"counts": False, "final_state": True},
    ).result()

    expected = (1 - p) * np.diag([0.0, 1.0]) + p * np.eye(2) / 2
    assert np.allclose(result.get_density_matrix(), expected)


def test_statevector_trajectories_match_density_matrix_statistics():
    p = 0.2
    shots = 4000
    program = _x_program(with_measurement=True)
    counts = (
        Simulator(method="SV", noise=_depolarized_x_model(p))
        .run(program, shots=shots, simulation_config={"seed": 7})
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
        Simulator(method="DM", noise=_depolarized_x_model(p))
        .run(program, shots=shots, simulation_config={"seed": 7})
        .result()
        .get_counts()
    )

    assert abs(counts.get("1", 0) / shots - (1 - p / 2)) < 0.02


def test_skipped_conditioned_gate_skips_its_channel():
    # q0 measures 0 deterministically, so the conditioned X (and its noise)
    # must not fire: q1 stays exactly |0><0|.
    noise = _depolarized_x_model(p=0.5)
    program = fq.Program(2, 1)
    program.measure(0, 0)
    program.add(fq.ops.X, 1, condition=(0, 1))
    result = (
        Simulator(method="DM", noise=noise)
        .run(
            program,
            shots=1,
            simulation_config={"seed": 5},
            result_config={"final_state": True},
        )
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
    program.measure(0, 0)
    program.add(fq.ops.X, 1, condition=(0, 1))
    result = (
        Simulator(method="DM", noise=noise)
        .run(
            program,
            shots=1,
            simulation_config={"seed": 5},
            result_config={"final_state": True},
        )
        .result()
    )

    rho = result.get_density_matrix()
    # Little-endian: q0 is the least-significant digit. q0=1 fixed, q1 mixed.
    expected = np.zeros((4, 4), dtype=complex)
    expected[1, 1] = p / 2  # q1=0, q0=1
    expected[3, 3] = 1 - p / 2  # q1=1, q0=1
    assert np.allclose(rho, expected)


def test_seeded_noisy_runs_are_reproducible():
    backend = Simulator(method="SV", noise=_depolarized_x_model())
    program = _x_program(with_measurement=True)
    first = (
        backend.run(program, shots=64, simulation_config={"seed": 11})
        .result()
        .get_counts()
    )
    second = (
        backend.run(program, shots=64, simulation_config={"seed": 11})
        .result()
        .get_counts()
    )

    assert first == second


def test_parallel_dynamic_shots_match_serial_with_channels():
    noise = _depolarized_x_model()
    program = _x_program(with_measurement=True)
    serial = (
        Simulator(method="SV", noise=noise)
        .run(
            program, shots=8, simulation_config={"seed": 13, "parallel_mode": "serial"}
        )
        .result()
        .get_counts()
    )
    parallel = (
        Simulator(method="SV", noise=noise)
        .run(program, shots=8, simulation_config={"seed": 13, "max_workers": 2})
        .result()
        .get_counts()
    )

    assert parallel == serial


# --- check_noise_support ---


def test_check_noise_support_accepts_catalog_channels():
    report = Simulator().check_noise_support(_depolarized_x_model())

    assert report.supported is True
    assert report.accepted_sources == ("Depolarizing",)
    assert report.rejected_sources == ()


def test_check_noise_support_requires_a_noise_model():
    with pytest.raises(BackendValidationError, match="noise_model"):
        Simulator().check_noise_support(object())


def test_check_noise_support_rejects_unknown_channel():
    class Leakage(Channel):
        pass

    noise = NoiseModel()
    noise.add(Leakage(), operation=fq.ops.X)
    report = Simulator().check_noise_support(noise)

    assert report.supported is False
    assert report.rejected_sources == ("Leakage",)
    assert report.accepted_sources == ()
    assert len(report.warnings) == 1


def test_check_noise_support_reports_rate_mode_damping_as_unsupported():
    noise = NoiseModel()
    noise.add(AmplitudeDamping(rate=0.01), operation=fq.ops.X)
    report = Simulator().check_noise_support(noise)

    assert report.supported is False
    assert report.rejected_sources == ("AmplitudeDamping(rate)",)
    assert report.accepted_sources == ()


def test_matrix_capability_rejects_background_and_thermal_generator_forms():
    background = NoiseModel()
    background.add(PhaseDamping(rate=0.01), targets=0)
    background_report = Simulator().check_noise_support(background)
    assert not background_report.supported
    assert "background" in background_report.rejected_sources[0]

    thermal = NoiseModel()
    thermal.add(ThermalRelaxation(t1=60e-6, t2=80e-6), operation=fq.ops.X)
    thermal_report = Simulator().check_noise_support(thermal)
    assert not thermal_report.supported
    assert thermal_report.rejected_sources == ("ThermalRelaxation",)


def test_matrix_custom_finite_channels_are_map_driven_not_field_name_driven():
    class CustomFinite(Channel):
        rate = "a finite-channel calibration label, not a generator mode"

    channel_map = default_channel_implementation_map()
    channel_map.register(
        CustomFinite,
        lambda channel, *, targets: (np.eye(2, dtype=complex),),
    )
    noise = NoiseModel()
    noise.add(CustomFinite(), operation=fq.ops.X)

    report = Simulator(channel_implementation_map=channel_map).check_noise_support(
        noise
    )
    assert report.supported
    assert report.accepted_sources == ("CustomFinite",)


def test_matrix_rejects_thermal_relaxation_even_with_a_registered_kraus_rule():
    channel_map = default_channel_implementation_map()
    channel_map.register(
        ThermalRelaxation,
        lambda channel, *, targets: (np.eye(2, dtype=complex),),
    )
    noise = NoiseModel()
    noise.add(ThermalRelaxation(t1=60e-6, t2=80e-6), operation=fq.ops.X)

    report = Simulator(channel_implementation_map=channel_map).check_noise_support(
        noise
    )
    assert not report.supported
    assert report.rejected_sources == ("ThermalRelaxation",)
    assert "matrix-family policy" in report.warnings[0]
    assert "registered channel implementation" in report.warnings[0]


def test_matrix_backend_captures_noise_registrations_at_construction():
    source = NoiseModel()
    source.add(Depolarizing(p=0.0), operation=fq.ops.X)
    backend = Simulator(method="DM", noise=source)
    source.add(AmplitudeDamping(p=1.0), operation=fq.ops.X)

    result = backend.run(
        _x_program(), result_config={"counts": False, "final_state": True}
    ).result()
    assert np.allclose(result.get_density_matrix(), np.diag([0.0, 1.0]))
    assert "AmplitudeDamping(p)" in backend.check_noise_support(source).accepted_sources


def test_matrix_backend_checks_captured_noise_once_at_construction(monkeypatch):
    calls = []
    original = Simulator.check_noise_support

    def count_checks(self, noise_model):
        calls.append(noise_model)
        return original(self, noise_model)

    monkeypatch.setattr(Simulator, "check_noise_support", count_checks)
    noise = _depolarized_x_model()
    backend = Simulator(noise=noise)

    assert len(calls) == 1
    backend.run(_x_program())
    assert len(calls) == 1


def test_check_noise_support_distinguishes_p_and_rate_mode_of_the_same_class():
    noise = NoiseModel()
    noise.add(AmplitudeDamping(p=0.1), operation=fq.ops.X)
    noise.add(AmplitudeDamping(rate=0.01), operation=fq.ops.H)
    report = Simulator().check_noise_support(noise)

    assert report.supported is False
    assert report.accepted_sources == ("AmplitudeDamping(p)",)
    assert report.rejected_sources == ("AmplitudeDamping(rate)",)


def test_constructor_rejects_rate_mode_damping():
    noise = NoiseModel()
    noise.add(AmplitudeDamping(rate=0.01), operation=fq.ops.X)
    with pytest.raises(BackendValidationError, match="rate mode"):
        Simulator(noise=noise)


# --- validate_for: run() direct-raise strict selector-identity validation ---


def test_run_rejects_foreign_logical_gate_selector_directly():
    program = _x_program()
    foreign = fq.QuantumRegister(1, name="q")
    noise = NoiseModel()
    noise.add(Depolarizing(p=0.1), operation=fq.ops.X, targets=(foreign[0],))
    backend = Simulator(noise=noise)

    with pytest.raises(BackendValidationError):
        backend.run(program)


def test_run_rejects_unmapped_physical_gate_label_directly():
    # (99,) on a three-subsystem generic-simulator program: not a member of
    # the effective layout's device labels for this run.
    program = fq.Program(3)
    program.add(fq.ops.RZ(0.1), 0)
    noise = NoiseModel()
    noise.add(Depolarizing(p=0.1), operation=fq.ops.X, targets=(99,))
    backend = Simulator(noise=noise)

    with pytest.raises(BackendValidationError):
        backend.run(program)


def test_run_succeeds_when_valid_gate_selector_matches_no_occurrence():
    # Site 15 is legal on the fixed device but is not modeled by this run.
    # The selector validates and produces no numerical term.
    program = fq.Program(3)
    program.add(fq.ops.RZ(0.1), 0)
    noise = NoiseModel()
    noise.add(Depolarizing(p=0.1), operation=fq.ops.Y, targets=(15,))
    backend = SCQubitIBMSimulator(noise=noise)

    result = backend.run(program).result()
    assert result is not None


def test_numba_fused_kernel_compiles_channel_plans_matching_numpy():
    # A channel-bearing plan compiles into the fused numba kernel, which weighs
    # quantum-jump branches from the reduced density matrix while NumPy norms
    # each branch - same distribution, counts agree statistically not bit-wise.
    pytest.importorskip("numba")
    from fatqat.simulator._engine.nb import NumbaSVEngine, _plan_compilable

    noise = NoiseModel()
    noise.add(Depolarizing(p=0.3), operation=fq.ops.X)
    backend = Simulator(noise=noise)
    program = fq.Program(1, 1)
    program.add(fq.ops.X, 0)
    program.measure(0, 0)
    plan, _ = backend._lower_program(program)
    assert _plan_compilable(plan) is True

    # Below the auto-parallel floor so both simulators run in-process serial.
    shots = 400
    request = backend._request_cls(counts=True, statevector=False)
    counts = {}
    for cls in (NumpySVEngine, NumbaSVEngine):
        simulator = cls()
        simulator.initialize((2,), 1)
        raw = simulator.run(plan, shots, 7, request)
        counts[cls.__name__] = dict(
            zip(
                (tuple(key) for key in raw.outcome_keys.tolist()),
                raw.outcome_counts.tolist(),
            )
        )
    tv = _total_variation(counts["NumpySVEngine"], counts["NumbaSVEngine"], shots)
    assert tv < 0.05


def test_numba_fused_channel_kernel_matches_numpy_on_a_qudit_dynamic_plan():
    # Every fused step kind at once on a qutrit register: two channels on one
    # occurrence, a feedforward gate, a reset, two measurements. Draw order must
    # stay aligned with NumPy; counts agree statistically, not bit-for-bit.
    pytest.importorskip("numba")

    def counts_for(runtime):
        noise = NoiseModel()
        noise.add(AmplitudeDamping(p=(0.2, 0.3)), operation=fq.ops.Shift)
        noise.add(PhaseDamping(p=0.15), operation=fq.ops.Shift)
        qreg = fq.QuantumRegister(2, dim=3)
        creg = fq.ClassicalRegister(2, dim=3)
        program = fq.Program([qreg], [creg])
        program.add(fq.ops.Shift(1), qreg[0])
        program.measure(qreg[0], creg[0])
        program.add(fq.ops.Shift(1), qreg[1], condition=(creg[0], 1))
        program.add(fq.ops.Reset, qreg[0])
        program.measure(qreg[1], creg[1])
        backend = Simulator(method="SV", runtime=runtime, noise=noise)
        job = backend.run(program, shots=400, simulation_config={"seed": 11})
        return job.result().get_counts()

    assert _total_variation(counts_for("numba"), counts_for("numpy"), 400) < 0.05


def test_reduced_diagonal_matches_the_full_reduced_density_matrix():
    # The marginal the diagonal weighing reads must be the diagonal of what
    # the general path builds, for any target tuple and mix of dimensions.
    pytest.importorskip("numba")
    from fatqat.simulator._engine.nb import (
        NumbaSVEngine,
        _reduced_density,
        _reduced_diagonal,
    )

    dims = (2, 3, 2)
    size = 12
    rng = np.random.default_rng(5)
    state = rng.normal(size=size) + 1j * rng.normal(size=size)
    state /= np.linalg.norm(state)

    engine = NumbaSVEngine()
    engine.initialize(dims, 0)
    for targets in ((0,), (1,), (0, 2), (2, 1)):
        offsets, comp_strides, comp_dims, _, _ = engine._build_apply_plan(targets)
        cosets = size // offsets.shape[0]
        full = _reduced_density(state, offsets, comp_strides, comp_dims, cosets)
        marginal = _reduced_diagonal(state, offsets, comp_strides, comp_dims, cosets)

        assert np.allclose(marginal, np.real(np.diagonal(full)))
        assert np.isclose(marginal.sum(), 1.0)


def test_numba_dm_channel_matches_numpy_on_a_qudit_channel_plan():
    # The DM channel path applies the numba-built super-operator in one pass
    # while NumPy sums per-Kraus sandwiches, so the two agree numerically not
    # bit-for-bit (kernel exactness vs the Kronecker sum: tests/noise/test_nb.py).
    pytest.importorskip("numba")

    noise = NoiseModel()
    noise.add(AmplitudeDamping(p=(0.2, 0.3)), operation=fq.ops.Shift)
    noise.add(PhaseDamping(p=0.4), operation=fq.ops.Shift)
    qreg = fq.QuantumRegister(1, dim=3)

    states = []
    for runtime in ("numpy", "numba"):
        program = fq.Program([qreg])
        program.add(fq.ops.Fourier, qreg[0])
        program.add(fq.ops.Shift(1), qreg[0])
        backend = Simulator(method="DM", runtime=runtime, noise=noise)
        job = backend.run(program, result_config={"counts": False, "final_state": True})
        states.append(job.result().get_density_matrix())

    assert np.allclose(states[0], states[1])


def test_scoped_damping_decays_only_the_selected_cz_slot():
    """Scoped noise must affect its selected qubit, not merely lower correctly."""
    for slot, expected_marginals in ((0, (0.5, 1.0)), (1, (1.0, 0.5))):
        program = fq.Program(2)
        program.add(fq.ops.X, 0)
        program.add(fq.ops.X, 1)
        # CZ changes only the |11> phase, leaving these populations intact.
        program.add(fq.ops.CZ, (0, 1))
        noise = NoiseModel()
        noise.add(
            AmplitudeDamping(p=(0.5,)), operation=fq.ops.CZ, target_positions=(slot,)
        )

        density_matrix = (
            Simulator(method="DM", noise=noise)
            .run(
                program,
                result_config={"counts": False, "final_state": True},
            )
            .result()
            .get_density_matrix()
        )
        # Fatqat's global basis order is little-endian: |q1 q0>.
        p_q0 = np.real(density_matrix[1, 1] + density_matrix[3, 3])
        p_q1 = np.real(density_matrix[2, 2] + density_matrix[3, 3])
        assert np.allclose((p_q0, p_q1), expected_marginals)


def test_scoped_channel_uses_only_the_qudit_extent_dimension():
    program = fq.Program([fq.QuantumRegister(2, dim=3)])
    program.add(fq.ops.Sum, (0, 1))
    noise = NoiseModel()
    noise.add(PhaseDamping(p=0.1), operation=fq.ops.Sum, target_positions=(1,))

    plan, _ = Simulator(noise=noise)._lower_program(program)
    channel_steps = [step for step in plan if isinstance(step, ApplyChannelStep)]

    assert channel_steps[0].target_indices == (1,)
    assert all(kraus.shape == (3, 3) for kraus in channel_steps[0].kraus_ops)


def test_multi_slot_extent_has_joint_kraus_dimension():
    program = fq.Program(3)
    program.add(fq.ops.CCX, (0, 1, 2))
    noise = NoiseModel()
    noise.add(Depolarizing(p=0.01), operation=fq.ops.CCX, target_positions=(1, 2))

    plan, _ = Simulator(noise=noise)._lower_program(program)
    channel_steps = [step for step in plan if isinstance(step, ApplyChannelStep)]

    assert channel_steps[0].target_indices == (1, 2)
    assert all(kraus.shape == (4, 4) for kraus in channel_steps[0].kraus_ops)
