"""Pauli sampling: the scaled-unitary branch route through the SV engines.

Pins three things: which channels qualify, that the route is the *same*
estimator as the general quantum-jump path (not merely a statistically similar
one), and that both hold on either runtime and on both the compiled multi-shot
kernel and the serial fallback.
"""

import numpy as np
import pytest

import fatqat as fq
from fatqat.noise import (
    AmplitudeDamping,
    Channel,
    Depolarizing,
    NoiseModel,
    PauliChannel,
    PhaseDamping,
    default_channel_implementation_map,
)
from fatqat.noise.base import _unitary_branch_probabilities
from fatqat.noise.catalog import (
    amplitude_damping_rule,
    depolarizing_rule,
    pauli_channel_rule,
    phase_damping_rule,
)
from fatqat.simulator import Simulator

RUNTIMES = ["numpy", "numba"]


def _refs(*dims):
    return tuple(fq.QuantumRegister(1, dim=d)[0] for d in dims)


def _runtime(name):
    if name == "numba":
        pytest.importorskip("numba")
    return name


def _ghz_program(n=3):
    program = fq.Program(n, n)
    program.add(fq.ops.H, 0)
    for q in range(n - 1):
        program.add(fq.ops.CX, (q, q + 1))
    program.measure_all()
    return program


def _pauli_noise():
    """A one- and a two-qubit Pauli channel, on the two gates of a GHZ chain."""
    noise = NoiseModel()
    noise.add(PauliChannel({"X": 0.08, "Z": 0.05}), operation=fq.ops.H)
    noise.add(PauliChannel({"XI": 0.04, "IX": 0.03, "ZZ": 0.02}), operation=fq.ops.CX)
    return noise


def _counts(runtime, noise, program, shots=2000, seed=7):
    return (
        Simulator("SV", runtime=runtime, noise=noise)
        .run(
            program,
            shots=shots,
            simulation_config={
                "seed": seed,
                "shot_parallelism": "serial",
                "kernel_parallelism": "serial",
            },
        )
        .result()
        .get_counts()
    )


def _total_variation(counts_a, counts_b, shots):
    keys = set(counts_a) | set(counts_b)
    return 0.5 * sum(abs(counts_a.get(k, 0) - counts_b.get(k, 0)) for k in keys) / shots


# --- which channels qualify ---


@pytest.mark.parametrize(
    "channel, rule, targets",
    [
        (Depolarizing(p=0.2), depolarizing_rule, _refs(2)),
        (Depolarizing(p=0.2), depolarizing_rule, _refs(2, 2)),
        (Depolarizing(p=0.2), depolarizing_rule, _refs(3)),
        (PhaseDamping(p=0.2), phase_damping_rule, _refs(2)),
        (PhaseDamping(p=0.2), phase_damping_rule, _refs(3)),
        (PauliChannel({"X": 0.1}), pauli_channel_rule, _refs(2)),
        (PauliChannel({"XZ": 0.1, "YY": 0.05}), pauli_channel_rule, _refs(2, 2)),
    ],
)
def test_scaled_unitary_channels_expose_fixed_branch_probabilities(
    channel, rule, targets
):
    # Content, not descriptor type: Weyl operators and Clock powers qualify in
    # any dimension, exactly as Pauli strings do.
    probabilities = _unitary_branch_probabilities(tuple(rule(channel, targets=targets)))

    assert probabilities is not None
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.all(probabilities >= 0.0)


def test_pauli_channel_probabilities_are_exactly_its_terms():
    channel = PauliChannel({"X": 0.08, "Z": 0.05})
    kraus_ops = tuple(pauli_channel_rule(channel, targets=_refs(2)))

    probabilities = _unitary_branch_probabilities(kraus_ops)

    assert np.allclose(probabilities, [p for _, p in channel.terms])


def test_amplitude_damping_is_not_a_scaled_unitary_channel():
    # K1 moves population between levels, so K1^H K1 is a projector, not a
    # multiple of the identity; the general quantum-jump path must handle it.
    kraus_ops = tuple(amplitude_damping_rule(AmplitudeDamping(p=0.2), targets=_refs(2)))

    assert _unitary_branch_probabilities(kraus_ops) is None


def test_degenerate_all_zero_channel_falls_back():
    assert _unitary_branch_probabilities((np.zeros((2, 2), dtype=complex),)) is None


# --- exact semantics ---


@pytest.mark.parametrize("runtime", RUNTIMES)
@pytest.mark.parametrize("string, expected_index", [("XI", 1), ("IX", 2), ("XX", 3)])
def test_a_certain_pauli_error_acts_exactly_like_the_gate(
    runtime, string, expected_index
):
    # p=1 on one term makes the channel deterministic, pinning both the sampler
    # and the string's endianness (string[0] describes targets[0], whose flat
    # stride is 1). A final-state request also routes numba to its serial
    # fallback rather than the compiled multi-shot kernel, covering that path too.
    noise = NoiseModel()
    noise.add(PauliChannel({string: 1.0}), operation=fq.ops.CX)
    program = fq.Program(2)
    program.add(fq.ops.CX, (0, 1))

    statevector = (
        Simulator("SV", runtime=_runtime(runtime), noise=noise)
        .run(program, shots=1, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )

    expected = np.zeros(4, dtype=complex)
    expected[expected_index] = 1.0
    assert np.allclose(statevector, expected)


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_sampled_unitary_route_reproduces_the_jump_route_exactly(runtime, monkeypatch):
    # <psi|K_i^H K_i|psi> *is* p_i for a scaled unitary, so with the routing
    # disabled both paths consume the same draw and land on the same branch:
    # counts agree exactly, not just statistically. (Round-off could flip a
    # draw within ~1e-16 of a cdf boundary - far below any shot count.)
    runtime = _runtime(runtime)
    program = _ghz_program()
    noise = _pauli_noise()
    sampled = _counts(runtime, noise, program)

    monkeypatch.setattr(
        "fatqat.simulator._engine.np._sampled_unitary_branches", lambda ops: None
    )
    if runtime == "numba":
        monkeypatch.setattr(
            "fatqat.noise.nb._sampled_unitary_branches", lambda ops: None
        )
    jumped = _counts(runtime, noise, program)

    assert sampled == jumped


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_long_sampled_chain_keeps_the_state_normalized(runtime):
    # No occurrence renormalizes the state, so nothing but round-off may
    # accumulate over a long chain of drawn operators.
    noise = NoiseModel()
    noise.add(PauliChannel({"X": 0.3, "Y": 0.2, "Z": 0.2}), operation=fq.ops.RY)
    program = fq.Program(4)
    for layer in range(30):
        for q in range(4):
            program.add(fq.ops.RY(0.3 + 0.1 * layer), q)

    statevector = (
        Simulator("SV", runtime=_runtime(runtime), noise=noise)
        .run(program, shots=1, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )

    assert np.linalg.norm(statevector) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_statevector_pauli_sampling_matches_the_density_matrix(runtime):
    shots = 20000
    program = _ghz_program()
    noise = _pauli_noise()

    sampled = _counts(_runtime(runtime), noise, program, shots=shots)
    exact = (
        Simulator("DM", noise=noise)
        .run(program, shots=shots, simulation_config={"seed": 7})
        .result()
        .get_counts()
    )

    assert _total_variation(sampled, exact, shots) < 0.02


def test_numpy_and_numba_agree_on_a_pauli_sampled_plan():
    pytest.importorskip("numba")
    shots = 20000
    program = _ghz_program()
    noise = _pauli_noise()

    numpy_counts = _counts("numpy", noise, program, shots=shots)
    numba_counts = _counts("numba", noise, program, shots=shots)

    assert _total_variation(numpy_counts, numba_counts, shots) < 0.02


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_mixed_plan_routes_each_channel_independently(runtime):
    # One plan carrying both a scaled-unitary channel and a damping channel:
    # the per-channel decision must not become a per-plan one.
    noise = NoiseModel()
    noise.add(PauliChannel({"X": 0.1}), operation=fq.ops.H)
    noise.add(AmplitudeDamping(p=0.15), operation=fq.ops.CX, target_positions=(1,))
    shots = 20000
    program = _ghz_program(n=2)

    sampled = _counts(_runtime(runtime), noise, program, shots=shots)
    exact = (
        Simulator("DM", noise=noise)
        .run(program, shots=shots, simulation_config={"seed": 7})
        .result()
        .get_counts()
    )

    assert _total_variation(sampled, exact, shots) < 0.02


# --- integration with the rest of the noise surface ---


def test_pauli_channel_is_an_accepted_backend_capability():
    noise = NoiseModel()
    noise.add(PauliChannel({"X": 0.1}), operation=fq.ops.X)

    report = Simulator().check_noise_support(noise)

    assert report.supported is True
    assert report.accepted_sources == ("PauliChannel",)


def test_a_custom_non_unitary_channel_rule_still_runs():
    # Nothing about the routing is descriptor-keyed, so a user rule that
    # happens to produce non-unitary operators simply takes the jump path.
    class _Leak(Channel):
        _num_subsystems = 1

    def _leak_rule(channel, *, targets):
        return (
            np.array([[1.0, 0.0], [0.0, np.sqrt(0.5)]], dtype=complex),
            np.array([[0.0, np.sqrt(0.5)], [0.0, 0.0]], dtype=complex),
        )

    channel_map = default_channel_implementation_map()
    channel_map.register(_Leak, _leak_rule)
    assert _unitary_branch_probabilities(_leak_rule(_Leak(), targets=())) is None

    noise = NoiseModel()
    noise.add(_Leak(), operation=fq.ops.H)
    program = fq.Program(1, 1)
    program.add(fq.ops.H, 0)
    program.measure(0, 0)

    counts = (
        Simulator("SV", noise=noise, channel_implementation_map=channel_map)
        .run(program, shots=512, simulation_config={"seed": 3})
        .result()
        .get_counts()
    )

    assert sum(counts.values()) == 512


def test_pauli_channel_process_batches_match_serial():
    program = _ghz_program(n=2)
    noise = _pauli_noise()
    serial = _counts("numpy", noise, program, shots=8, seed=13)
    parallel = (
        Simulator("SV", runtime="numpy", noise=noise)
        .run(
            program,
            shots=8,
            simulation_config={
                "seed": 13,
                "shot_parallelism": "processes",
                "kernel_parallelism": "serial",
                "max_workers": 2,
            },
        )
        .result()
        .get_counts()
    )

    assert parallel == serial


def test_pauli_sampling_does_not_change_the_per_shot_draw_budget():
    pytest.importorskip("numba")
    from fatqat.simulator._engine.nb import NumbaSVEngine

    backend = Simulator("SV", noise=_pauli_noise())
    program = _ghz_program()
    plan, _ = backend._lower_program(program)
    engine = NumbaSVEngine()
    engine.initialize((2, 2, 2), 3)

    _, max_draws = engine._compile_dynamic_plan(plan)

    # One draw per measurement plus one per channel occurrence (3 gates) - the
    # same budget the general jump path needs.
    assert max_draws == 1 + 3
