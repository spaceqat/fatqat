"""Estimator: exact expectation values, result shape, and rejected programs."""

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as op
from fatqat.errors import BackendValidationError, ResultFieldUnavailableError
from fatqat.observable import Observable


def _bell(measured=False):
    program = fq.Program(2, 2) if measured else fq.Program(2)
    program.add(op.H, 0)
    program.add(op.CX, (0, 1))
    if measured:
        program.measure((0, 1), (0, 1))
    return program


def _estimator(method="SV", noise=None):
    return fq.Estimator(fq.simulator.Simulator(method=method, noise=noise))


def _noise_model():
    noise = fq.NoiseModel()
    noise.add_channel(fq.noise.Depolarizing(p=0.1), operation=op.CX)
    return noise


# --- exact values ------------------------------------------------------------


@pytest.mark.parametrize(
    "label, expected",
    [("ZZ", 1.0), ("XX", 1.0), ("YY", -1.0), ("ZI", 0.0), ("IZ", 0.0)],
)
@pytest.mark.parametrize("method", ["SV", "DM"])
def test_bell_expectation_values(label, expected, method):
    result = _estimator(method).run(_bell(), Observable([(label, 1.0)])).result()

    assert result.get_expectation() == pytest.approx(expected, abs=1e-12)


def test_multi_term_observable_returns_weighted_sum():
    # 1.0*(+1) + 0.5*(+1) + 0.25*(-1)
    observable = Observable([("ZZ", 1.0), ("XX", 0.5), ("YY", 0.25)])

    value = _estimator().run(_bell(), observable).result().get_expectation()
    assert value == pytest.approx(1.25, abs=1e-12)


def test_identity_term_contributes_its_coefficient():
    value = _estimator().run(_bell(), Observable([("II", 2.5)])).result()

    assert value.get_expectation() == pytest.approx(2.5, abs=1e-12)


def test_occupation_via_one_projector():
    # Bell state: each qubit is |1> half the time.
    occupation = Observable.from_sparse([("ONE", (0,), 1.0)], num_qubits=2)

    value = _estimator().run(_bell(), occupation).result().get_expectation()
    assert value == pytest.approx(0.5, abs=1e-12)


def test_density_matrix_with_noise_stays_exact():
    # Depolarizing(0.1) on the CX pulls the ideal +1 down to +0.9 exactly -
    # a density matrix applies the Kraus sum, so no sampling is involved.
    estimator = _estimator("DM", noise=_noise_model())

    value = estimator.run(_bell(), Observable([("ZZ", 1.0)])).result()
    assert value.get_expectation() == pytest.approx(0.9, abs=1e-12)


# --- result shape mirrors input shape ---------------------------------------


def test_single_observable_returns_a_scalar():
    value = _estimator().run(_bell(), Observable([("ZZ", 1.0)])).result()

    assert np.ndim(value.get_expectation()) == 0


def test_sequence_of_observables_returns_an_array():
    observables = [
        Observable([("ZZ", 1.0)]),
        Observable([("XX", 1.0)]),
        Observable([("ZI", 1.0)]),
    ]

    values = _estimator().run(_bell(), observables).result().get_expectation()

    assert values.shape == (3,)
    assert values == pytest.approx([1.0, 1.0, 0.0], abs=1e-12)


def test_one_element_sequence_still_returns_an_array():
    values = _estimator().run(_bell(), [Observable([("ZZ", 1.0)])]).result()

    assert values.get_expectation().shape == (1,)


def test_all_observables_share_a_single_evolution():
    # Evaluating K observables must agree with evaluating them one at a time;
    # the shared evolution is an optimization, not a change of answer.
    estimator = _estimator()
    observables = [Observable([("ZZ", 1.0)]), Observable([("XX", 1.0)])]

    together = estimator.run(_bell(), observables).result().get_expectation()
    separately = [
        estimator.run(_bell(), observable).result().get_expectation()
        for observable in observables
    ]
    assert together == pytest.approx(separately, abs=1e-12)


def test_metadata_records_the_request():
    result = _estimator().run(_bell(), [Observable([("ZZ", 1.0)])]).result()

    assert result.metadata["shots"] == 0
    assert result.metadata["num_observables"] == 1
    assert result.metadata["backend_name"] == "Simulator"


# --- rejected programs -------------------------------------------------------


def test_measured_program_rejected():
    with pytest.raises(BackendValidationError, match="collapses"):
        _estimator().run(_bell(measured=True), Observable([("ZZ", 1.0)]))


def test_observable_width_must_match_the_program():
    with pytest.raises(BackendValidationError, match="3 qubit"):
        _estimator().run(_bell(), Observable([("ZZZ", 1.0)]))


def test_qudit_program_rejected():
    program = fq.Program([fq.QuantumRegister(1, dim=3)])
    with pytest.raises(BackendValidationError, match="qubits only"):
        _estimator().run(program, Observable([("Z", 1.0)]))


@pytest.mark.parametrize("shots", [0, 1000])
def test_noisy_statevector_rejected(shots):
    # Rejected at shots > 0 too: sampling draws from the final state, and a
    # trajectory run has no single one. Sampling one branch would report the
    # statistics of that trajectory rather than of the noisy channel.
    estimator = _estimator("SV", noise=_noise_model())

    with pytest.raises(BackendValidationError, match="no single final state"):
        estimator.run(_bell(), Observable([("ZZ", 1.0)]), shots=shots)


@pytest.mark.parametrize("shots", [0, 1000])
def test_statevector_reset_rejected(shots):
    program = fq.Program(2)
    program.add(op.H, 0)
    program.add(op.Reset, 0)

    with pytest.raises(BackendValidationError, match="no single final state"):
        _estimator("SV").run(program, Observable([("ZZ", 1.0)]), shots=shots)


def test_noise_that_never_fires_is_accepted():
    # The channel is registered on a gate the program never uses, so no channel
    # reaches the lowered plan and the run stays deterministic. Judging this
    # from the noise model alone would reject a perfectly well-defined value;
    # the backend knows what actually landed in *this* program.
    noise = fq.NoiseModel()
    noise.add_channel(fq.noise.Depolarizing(p=0.1), operation=op.Swap)

    value = (
        _estimator("SV", noise=noise)
        .run(_bell(), Observable([("ZZ", 1.0)]))
        .result()
        .get_expectation()
    )
    assert value == pytest.approx(1.0, abs=1e-12)


def test_backend_validation_error_raises_rather_than_failing_the_job():
    # A validation failure is the caller's to fix, so it surfaces from run()
    # itself - not deferred into a failed Job that only errors at .result().
    estimator = _estimator("SV", noise=_noise_model())

    with pytest.raises(BackendValidationError):
        estimator.run(_bell(), Observable([("ZZ", 1.0)]))


def test_density_matrix_reset_is_accepted():
    # A density-matrix reset is the deterministic partial-trace channel, so the
    # final state - and the expectation value - stay well defined.
    program = fq.Program(2)
    program.add(op.H, 0)
    program.add(op.Reset, 0)

    value = _estimator("DM").run(program, Observable([("ZI", 1.0)])).result()
    assert value.get_expectation() == pytest.approx(1.0, abs=1e-12)


def test_empty_observable_sequence_rejected():
    with pytest.raises(BackendValidationError, match="no observables"):
        _estimator().run(_bell(), [])


def test_non_observable_input_rejected():
    with pytest.raises(TypeError, match="Observable"):
        _estimator().run(_bell(), ["ZZ"])


@pytest.mark.parametrize("shots", [-1, 1.5, "100"])
def test_invalid_shots_rejected(shots):
    with pytest.raises(BackendValidationError, match="shots must be"):
        _estimator().run(_bell(), Observable([("ZZ", 1.0)]), shots=shots)


def test_expectation_absent_from_a_plain_backend_run():
    result = (
        fq.simulator.Simulator(method="SV")
        .run(_bell(), result_config={"counts": False, "final_state": True})
        .result()
    )

    with pytest.raises(ResultFieldUnavailableError, match="Estimator"):
        result.get_expectation()
    with pytest.raises(ResultFieldUnavailableError, match="Estimator"):
        result.get_std()


# --- sampling ----------------------------------------------------------------


def _sampling_program(num_qubits=3):
    program = fq.Program(num_qubits)
    for qubit in range(num_qubits):
        program.add(op.RY(0.4 + 0.3 * qubit), qubit)
    for qubit in range(num_qubits - 1):
        program.add(op.CX, (qubit, qubit + 1))
    return program


def test_exact_run_reports_zero_standard_error():
    result = _estimator().run(_bell(), Observable([("ZZ", 1.0)])).result()

    assert result.get_std() == 0.0


@pytest.mark.parametrize("method", ["SV", "DM"])
def test_sampled_value_converges_to_the_exact_one(method):
    # 200_000 shots put the standard error near 2e-3, so a 5-sigma window is
    # tight enough to catch a wrong distribution and loose enough not to flake.
    program = _sampling_program()
    observable = Observable([("ZZZ", 1.0), ("XXI", 0.5), ("IYY", -0.75)])
    estimator = _estimator(method)

    exact = estimator.run(program, observable).result().get_expectation()
    sampled = estimator.run(
        program, observable, shots=200_000, simulation_config={"seed": 11}
    ).result()

    assert sampled.get_expectation() == pytest.approx(exact, abs=5 * sampled.get_std())


def test_projector_term_sampling_converges():
    # Eigenvalues {0, 1} rather than {+1, -1}: the outcome distribution needs
    # the term's second moment, so a Pauli-only sampler would be biased here.
    program = _sampling_program()
    observable = Observable.from_sparse(
        [("ONE", (0,), 1.0), (["ONE", "Z"], (1, 2), 0.5)], num_qubits=3
    )
    estimator = _estimator()

    exact = estimator.run(program, observable).result().get_expectation()
    sampled = estimator.run(
        program, observable, shots=200_000, simulation_config={"seed": 5}
    ).result()

    assert sampled.get_expectation() == pytest.approx(exact, abs=5 * sampled.get_std())


def test_standard_error_shrinks_as_one_over_sqrt_shots():
    program = _sampling_program()
    observable = Observable([("XXI", 1.0)])
    estimator = _estimator()

    few = estimator.run(program, observable, shots=1_000).result().get_std()
    many = estimator.run(program, observable, shots=100_000).result().get_std()

    assert few / many == pytest.approx(10.0, rel=1e-9)


def test_deterministic_term_has_no_spread():
    # <ZZ> = 1 exactly on a Bell state, so every shot returns +1: a sampled
    # run must reproduce the exact value with zero error, not jitter near it.
    result = (
        _estimator()
        .run(
            _bell(), Observable([("ZZ", 1.0)]), shots=100, simulation_config={"seed": 3}
        )
        .result()
    )

    assert result.get_expectation() == pytest.approx(1.0, abs=1e-12)
    # The variance lands on the rounding floor rather than exactly 0, and std
    # takes its square root, so ~1e-17 of residue shows up as ~1e-9.
    assert result.get_std() == pytest.approx(0.0, abs=1e-8)


def test_sampling_is_reproducible_under_a_seed():
    program = _sampling_program()
    observable = Observable([("ZZZ", 1.0), ("XXI", 0.5)])
    estimator = _estimator()

    def sample(seed):
        return (
            estimator.run(
                program, observable, shots=500, simulation_config={"seed": seed}
            )
            .result()
            .get_expectation()
        )

    assert sample(2) == sample(2)
    assert sample(2) != sample(7)


def test_sampled_sequence_returns_arrays_for_both_fields():
    observables = [Observable([("ZZ", 1.0)]), Observable([("XX", 1.0)])]

    result = (
        _estimator()
        .run(_bell(), observables, shots=1_000, simulation_config={"seed": 1})
        .result()
    )

    assert result.get_expectation().shape == (2,)
    assert result.get_std().shape == (2,)


def test_sampling_works_with_a_noisy_density_matrix():
    program = _sampling_program()
    observable = Observable([("ZZZ", 1.0)])
    estimator = _estimator("DM", noise=_noise_model())

    exact = estimator.run(program, observable).result().get_expectation()
    sampled = estimator.run(
        program, observable, shots=200_000, simulation_config={"seed": 4}
    ).result()

    assert sampled.get_expectation() == pytest.approx(exact, abs=5 * sampled.get_std())


def test_sampled_metadata_records_the_shot_count():
    result = _estimator().run(_bell(), Observable([("ZZ", 1.0)]), shots=512).result()

    assert result.metadata["shots"] == 512
