"""End-to-end channel noise: lowering, path classification, DM/SV semantics."""

from xml.etree.ElementPath import ops

import numpy as np
import pytest

import fatqat as fq
from fatqat._backends.steps import ApplyChannelStep, ApplyMatrixStep, AtomLossStep
from fatqat.program import Program
from fatqat.simulator import Simulator
from fatqat.errors import BackendValidationError, UnsupportedOperationError
from fatqat.noise import (
    AmplitudeDamping,
    AtomLoss,
    Channel,
    Depolarizing,
    NoiseModel,
    PhaseDamping,
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
    noise.add_channel(Depolarizing(p=p), operation=fq.ops.X)
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


def test_unresolvable_channel_type_raises():
    class Leakage(Channel):
        pass

    noise = NoiseModel()
    noise.add_channel(Leakage(), operation=fq.ops.X)
    backend = Simulator(noise=noise)
    program = _x_program()
    with pytest.raises(UnsupportedOperationError, match="Leakage"):
        backend._lower_program(program)


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
    noise.add_channel(Custom(), operation=fq.ops.X)
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
    noise.add_channel(Depolarizing(p=0.1), operation=fq.ops.RX)
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


def test_reset_attached_channels_raise_until_wired():
    noise = NoiseModel()
    noise.add_channel(Depolarizing(p=0.1), operation=fq.ops.Reset)
    backend = Simulator(noise=noise)
    program = fq.Program(1)
    program.add(fq.ops.Reset, 0)
    with pytest.raises(UnsupportedOperationError, match="Reset"):
        backend._lower_program(program)


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
    with pytest.raises(BackendValidationError, match="channel noise"):
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


# --- validate_noise ---


def test_validate_noise_accepts_catalog_channels():
    report = Simulator().validate_noise(_depolarized_x_model())

    assert report.supported is True
    assert report.accepted_sources == ("Depolarizing",)
    assert report.rejected_sources == ()


def test_validate_noise_rejects_unknown_channel_and_reset():
    class Leakage(Channel):
        pass

    noise = NoiseModel()
    noise.add_channel(Leakage(), operation=fq.ops.X)
    noise.add_channel(Depolarizing(p=0.1), operation=fq.ops.Reset)
    report = Simulator().validate_noise(noise)

    assert report.supported is False
    assert set(report.rejected_sources) == {"Leakage", "Reset"}
    assert "Depolarizing" in report.accepted_sources
    assert len(report.warnings) == 2


def test_validate_noise_reports_rate_mode_damping_as_unsupported():
    noise = NoiseModel()
    noise.add_channel(AmplitudeDamping(rate=0.01), operation=fq.ops.X)
    report = Simulator().validate_noise(noise)

    assert report.supported is False
    assert report.rejected_sources == ("AmplitudeDamping(rate)",)
    assert report.accepted_sources == ()


def test_validate_noise_distinguishes_p_and_rate_mode_of_the_same_class():
    noise = NoiseModel()
    noise.add_channel(AmplitudeDamping(p=0.1), operation=fq.ops.X)
    noise.add_channel(AmplitudeDamping(rate=0.01), operation=fq.ops.H)
    report = Simulator().validate_noise(noise)

    assert report.supported is False
    assert report.accepted_sources == ("AmplitudeDamping(p)",)
    assert report.rejected_sources == ("AmplitudeDamping(rate)",)


def test_run_rejects_rate_mode_damping_before_execution():
    noise = NoiseModel()
    noise.add_channel(AmplitudeDamping(rate=0.01), operation=fq.ops.X)
    backend = Simulator(noise=noise)

    with pytest.raises(BackendValidationError, match="rate mode"):
        backend.run(_x_program())


# --- validate_for: run() direct-raise strict selector-identity validation ---


def test_run_rejects_foreign_logical_gate_selector_directly():
    program = _x_program()
    foreign = fq.QuantumRegister(1, name="q")
    noise = NoiseModel()
    noise.add_channel(Depolarizing(p=0.1), operation=fq.ops.X, targets=(foreign[0],))
    backend = Simulator(noise=noise)

    with pytest.raises(BackendValidationError):
        backend.run(program)


def test_run_rejects_unmapped_physical_gate_label_directly():
    # (99,) on a three-subsystem generic-simulator program: not a member of
    # the effective layout's device labels for this run.
    program = fq.Program(3)
    program.add(fq.ops.H, 0)
    noise = NoiseModel()
    noise.add_channel(Depolarizing(p=0.1), operation=fq.ops.X, targets=(99,))
    backend = Simulator(noise=noise)

    with pytest.raises(BackendValidationError):
        backend.run(program)


def test_run_succeeds_when_valid_gate_selector_matches_no_occurrence():
    # A valid selector (real device label) that the program never triggers
    # is a permitted no-effect entry, not a validation error.
    program = fq.Program(3)
    program.add(fq.ops.H, 0)
    noise = NoiseModel()
    noise.add_channel(
        Depolarizing(p=0.1), operation=fq.ops.Y, targets=(2,)
    )  # no Y in program
    backend = Simulator(noise=noise)

    result = backend.run(program).result()
    assert result is not None


def test_numba_fused_kernel_compiles_channel_plans_matching_numpy():
    # A channel-bearing plan compiles into the fused numba kernel, which weighs
    # quantum-jump branches from the reduced density matrix while NumPy norms
    # each branch - same distribution, counts agree statistically not bit-wise.
    pytest.importorskip("numba")
    from fatqat.simulator._engine.nb import NumbaSVEngine, _plan_compilable

    noise = NoiseModel()
    noise.add_channel(Depolarizing(p=0.3), operation=fq.ops.X)
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
        noise.add_channel(AmplitudeDamping(p=(0.2, 0.3)), operation=fq.ops.Shift)
        noise.add_channel(PhaseDamping(p=0.15), operation=fq.ops.Shift)
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


def test_numba_dm_channel_matches_numpy_on_a_qudit_channel_plan():
    # The DM channel path applies the numba-built super-operator in one pass
    # while NumPy sums per-Kraus sandwiches, so the two agree numerically not
    # bit-for-bit (kernel exactness vs the Kronecker sum: tests/noise/test_nb.py).
    pytest.importorskip("numba")

    noise = NoiseModel()
    noise.add_channel(AmplitudeDamping(p=(0.2, 0.3)), operation=fq.ops.Shift)
    noise.add_channel(PhaseDamping(p=0.4), operation=fq.ops.Shift)
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
        noise.add_channel(
            AmplitudeDamping(p=(0.5,)), operation=fq.ops.CZ, slots=(slot,)
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
    noise.add_channel(PhaseDamping(p=0.1), operation=fq.ops.Sum, slots=(1,))

    plan, _ = Simulator(noise=noise)._lower_program(program)
    channel_steps = [step for step in plan if isinstance(step, ApplyChannelStep)]

    assert channel_steps[0].target_indices == (1,)
    assert all(kraus.shape == (3, 3) for kraus in channel_steps[0].kraus_ops)


def test_multi_slot_extent_has_joint_kraus_dimension():
    program = fq.Program(3)
    program.add(fq.ops.CCX, (0, 1, 2))
    noise = NoiseModel()
    noise.add_channel(Depolarizing(p=0.01), operation=fq.ops.CCX, slots=(1, 2))

    plan, _ = Simulator(noise=noise)._lower_program(program)
    channel_steps = [step for step in plan if isinstance(step, ApplyChannelStep)]

    assert channel_steps[0].target_indices == (1, 2)
    assert all(kraus.shape == (4, 4) for kraus in channel_steps[0].kraus_ops)


def test_atom_loss_p1_isolated_atom_reads_erasure():
    program = fq.Program(1, 1)
    program.add(fq.ops.LoadAtoms(1, 1))
    program.add(fq.ops.RX(np.pi), 0)        
    program.measure(0, 0)

    noise = NoiseModel()
    noise.add_channel(AtomLoss(p=1.0), operation=fq.ops.RX)

    counts = (
        fq.simulator.AtomGridSimulator(grid_size=(1, 1), noise=noise)
        .run(program, shots=100, simulation_config={"seed": 0})
        .result().get_counts()
    )
    assert counts == {"2": 100}


def test_atom_loss_p0_reproduces_ideal():
    program = fq.Program(1, 1)
    program.add(fq.ops.LoadAtoms(1, 1))
    program.add(fq.ops.RX(np.pi), 0)
    program.measure(0, 0)

    noise = NoiseModel()
    noise.add_channel(AtomLoss(p=0.0), operation=fq.ops.RX)

    counts = (
        fq.simulator.AtomGridSimulator(grid_size=(1, 1), noise=noise)
        .run(program, shots=100, simulation_config={"seed": 0})
        .result().get_counts()
    )
    assert counts == {"1": 100}    


def test_lost_control_does_not_dephase_survivor():

    atoms = fq.GridRegister(1, 2, name="atoms")
    program = fq.Program([atoms], 1)
    program.add(fq.ops.LoadAtoms(1, 2))
    def h(q):                                   
        program.add(fq.ops.RZ(np.pi), q)
        program.add(fq.ops.RY(np.pi / 2), q)
    h(atoms[0])                                 
    h(atoms[1])                                 
    program.add(fq.ops.RX(0.0), atoms[0])
    program.add(fq.ops.RX(np.pi), atoms[0])
    program.add(fq.ops.CZ, (atoms[0], atoms[1]))
    h(atoms[1])                                
    program.measure(atoms[1], 0)

    noise = NoiseModel()
    noise.add_channel(AtomLoss(p=1.0), operation=fq.ops.RX)

    counts = (
        fq.simulator.AtomGridSimulator(grid_size=(1, 2), noise=noise)
        .run(program, shots=200, simulation_config={"seed": 0})
        .result().get_counts()
    )
    assert counts == {"0": 200}                 