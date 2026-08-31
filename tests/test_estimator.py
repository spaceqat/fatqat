"""Estimator: exact expectation values, result shape, and rejected programs."""

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.errors import (
    BackendExecutionError,
    BackendValidationError,
    ResultFieldUnavailableError,
)
from fatqat.job import Job
from fatqat.observable import Observable
from fatqat.result import Result


def _bell(measured=False):
    program = fq.Program(2, 2) if measured else fq.Program(2)
    program.add(ops.H, 0)
    program.add(ops.CX, (0, 1))
    if measured:
        program.measure((0, 1), (0, 1))
    return program


def _estimator(method="SV", noise=None):
    return fq.Estimator(fq.simulator.Simulator(method=method, noise=noise))


def _noise_model():
    noise = fq.NoiseModel()
    noise.add(fq.noise.Depolarizing(p=0.1), operation=ops.CX)
    return noise


def _parameterized_template():
    features = fq.ParameterVector("features", 2)
    theta = fq.ParameterVector("theta", 2)
    program = fq.Program(2)
    program.add(ops.RX(features[0]), 0)
    program.add(ops.RY(theta[0]), 0)
    program.add(ops.RX(features[1]), 1)
    program.add(ops.RY(theta[1]), 1)
    return program, features, theta


# --- parameter sweeps -------------------------------------------------------


def test_exact_scalar_parameter_sweep_matches_explicit_runs():
    angle = fq.Parameter("angle")
    program = fq.Program(1)
    program.add(ops.RY(angle), 0)
    observable = Observable([("Z", 1.0)])
    values = np.array([0.0, 0.4, 1.1])
    estimator = _estimator()

    swept = estimator.run_sweep(program, observable, {angle: values}).result()
    explicit = [
        estimator.run(program.assign_parameters({angle: value}), observable).result()
        for value in values
    ]

    assert [result.get_expectation() for result in swept] == pytest.approx(
        [result.get_expectation() for result in explicit]
    )
    assert [result.get_std() for result in swept] == [0.0, 0.0, 0.0]


def test_qnn_shaped_vector_sweep_preserves_multi_observable_shape():
    program, features, theta = _parameterized_template()
    feature_batch = np.array([[0.1, 0.2], [0.4, 0.5], [0.7, 0.8]])
    theta_value = np.array([0.3, 0.6])
    theta_batch = np.broadcast_to(theta_value, (len(feature_batch), len(theta)))
    observables = [Observable([("ZI", 1.0)]), Observable([("IZ", 1.0)])]
    estimator = _estimator()

    swept = estimator.run_sweep(
        program,
        observables,
        {features: feature_batch, theta: theta_batch},
    ).result()
    explicit = [
        estimator.run(
            program.assign_parameters(
                {features: feature_batch[index], theta: theta_batch[index]}
            ),
            observables,
        ).result()
        for index in range(len(feature_batch))
    ]

    assert all(result.get_expectation().shape == (2,) for result in swept)
    assert all(
        np.allclose(left.get_expectation(), right.get_expectation())
        for left, right in zip(swept, explicit, strict=True)
    )


def test_sampled_sweep_matches_same_seed_repeated_runs_and_forwards_options(
    monkeypatch,
):
    angle = fq.Parameter("angle")
    program = fq.Program(1)
    program.add(ops.RY(angle), 0)
    observable = Observable([("X", 1.0)])
    values = np.array([0.2, 0.9])
    estimator = _estimator()
    original_run = estimator.run
    forwarded = []

    def record(bound, supplied_observables, **kwargs):
        forwarded.append((supplied_observables, kwargs))
        return original_run(bound, supplied_observables, **kwargs)

    monkeypatch.setattr(estimator, "run", record)
    config = {"seed": 23}
    swept = estimator.run_sweep(
        program,
        observable,
        {angle: values},
        shots=256,
        simulation_config=config,
    ).result()
    monkeypatch.setattr(estimator, "run", original_run)
    explicit = [
        original_run(
            program.assign_parameters({angle: value}),
            observable,
            shots=256,
            simulation_config=config,
        ).result()
        for value in values
    ]

    assert [result.get_expectation() for result in swept] == [
        result.get_expectation() for result in explicit
    ]
    assert [result.get_std() for result in swept] == pytest.approx(
        [result.get_std() for result in explicit]
    )
    assert forwarded == [
        (observable, {"shots": 256, "simulation_config": config}),
        (observable, {"shots": 256, "simulation_config": config}),
    ]


def test_estimator_sweep_direct_inner_failure_propagates(monkeypatch):
    angle = fq.Parameter("angle")
    program = fq.Program(1)
    program.add(ops.RX(angle), 0)
    observable = Observable([("Z", 1.0)])
    estimator = _estimator()

    def fail_on_second(bound, _observables, **_kwargs):
        if bound._instructions[0].operation.theta == 0.2:
            raise BackendValidationError("direct point failure")
        return Job(status="DONE", result=Result(data={"expectation": 1.0, "std": 0.0}))

    monkeypatch.setattr(estimator, "run", fail_on_second)
    with pytest.raises(BackendValidationError, match="direct point failure"):
        estimator.run_sweep(program, observable, {angle: [0.1, 0.2]})


def test_estimator_sweep_failed_point_job_fails_outer_job(monkeypatch):
    angle = fq.Parameter("angle")
    program = fq.Program(1)
    program.add(ops.RX(angle), 0)
    observable = Observable([("Z", 1.0)])
    estimator = _estimator()
    error = KeyboardInterrupt("stored point failure")

    def fail_on_second(bound, _observables, **_kwargs):
        if bound._instructions[0].operation.theta == 0.2:
            return Job(status="ERROR", error=error)
        return Job(status="DONE", result=Result(data={"expectation": 1.0, "std": 0.0}))

    monkeypatch.setattr(estimator, "run", fail_on_second)
    outer = estimator.run_sweep(program, observable, {angle: [0.1, 0.2]})

    with pytest.raises(KeyboardInterrupt, match="stored point failure") as caught:
        outer.result()
    assert caught.value is error


def test_ordinary_estimator_rejects_unbound_without_translating_message():
    angle = fq.Parameter("angle")
    program = fq.Program(1)
    program.add(ops.RX(angle), 0)

    with pytest.raises(
        BackendValidationError,
        match="^program has unbound parameters: angle$",
    ) as caught:
        _estimator().run(program, Observable([("Z", 1.0)]))
    assert "no single final state" not in str(caught.value)


# --- exact values ------------------------------------------------------------


@pytest.mark.parametrize("method", ["SV", "DM"])
def test_asymmetric_observables_use_public_qubit_order_exactly(method):
    program = fq.Program(2)
    program.add(ops.X, 0)  # public |10>
    observables = [Observable([("ZI", 1.0)]), Observable([("IZ", 1.0)])]

    values = _estimator(method).run(program, observables).result().get_expectation()
    assert values == pytest.approx([-1.0, 1.0], abs=1e-12)


def test_sampled_asymmetric_observables_keep_public_target_association():
    program = fq.Program(2)
    program.add(ops.X, 0)
    observables = [Observable([("ZI", 1.0)]), Observable([("IZ", 1.0)])]

    values = (
        _estimator("SV")
        .run(program, observables, shots=64, simulation_config={"seed": 7})
        .result()
        .get_expectation()
    )
    assert values == pytest.approx([-1.0, 1.0], abs=1e-12)


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
    # Depolarizing(p=0.1) on the CX pulls the ideal +1 down to +0.9 exactly -
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
    program.add(ops.H, 0)
    program.add(ops.Reset, 0)

    with pytest.raises(BackendValidationError, match="no single final state"):
        _estimator("SV").run(program, Observable([("ZZ", 1.0)]), shots=shots)


def test_noise_that_never_fires_is_accepted():
    # The channel is registered on a gate the program never uses, so no channel
    # reaches the lowered plan and the run stays deterministic. Judging this
    # from the noise model alone would reject a perfectly well-defined value;
    # the backend knows what actually landed in *this* program.
    noise = fq.NoiseModel()
    noise.add(fq.noise.Depolarizing(p=0.1), operation=ops.Swap)

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
    program.add(ops.H, 0)
    program.add(ops.Reset, 0)

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
        program.add(ops.RY(0.4 + 0.3 * qubit), qubit)
    for qubit in range(num_qubits - 1):
        program.add(ops.CX, (qubit, qubit + 1))
    return program


def test_exact_run_reports_zero_standard_error():
    result = _estimator().run(_bell(), Observable([("ZZ", 1.0)])).result()

    assert result.get_std() == 0.0
    assert result.get_data("std") == 0.0
    assert result.available_data == frozenset({"expectation", "std"})


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


# --- backends that produce an operator, not a state --------------------------


@pytest.mark.parametrize("method", ["unitary", "superop"])
def test_operator_backend_rejected_at_construction(method):
    # Rejected when the estimator is built, not when it is run: the mismatch
    # is a property of the backend alone, so there is no reason to make the
    # caller lower and evolve a program before hearing about it.
    backend = fq.simulator.Simulator(method=method)

    with pytest.raises(BackendValidationError, match="produces a state"):
        fq.Estimator(backend)


@pytest.mark.parametrize("method", ["unitary", "superop"])
def test_operator_rejection_names_the_method_and_the_fix(method):
    backend = fq.simulator.Simulator(method=method)

    with pytest.raises(BackendValidationError) as caught:
        fq.Estimator(backend)

    message = str(caught.value)
    assert method in message  # says which method it saw
    assert "statevector" in message and "density_matrix" in message  # and the fix


@pytest.mark.parametrize("method", ["SV", "DM", "statevector", "density_matrix"])
def test_state_backends_still_accepted(method):
    assert fq.Estimator(fq.simulator.Simulator(method=method)) is not None


def test_backend_without_a_method_property_is_left_alone():
    # Duck typing is the constructor's only contract, so a backend that
    # predates the property must not be refused on that basis.
    class _Bare:
        def run(self, *args, **kwargs):
            raise AssertionError("not reached")

    assert fq.Estimator(_Bare()) is not None


def test_backend_result_without_state_returns_failed_job():
    class _NoStateBackend:
        method = "statevector"

        def run(self, *_args, **_kwargs):
            return Job(status="DONE", result=Result(data={"diagnostic": "complete"}))

    job = fq.Estimator(_NoStateBackend()).run(_bell(), Observable([("ZZ", 1.0)]))

    assert job.status == "ERROR"
    with pytest.raises(
        BackendExecutionError,
        match="estimator backend returned no final state; expected a "
        "statevector or density matrix",
    ):
        job.result()


class _FixedStateBackend:
    def __init__(self, representation, state):
        self.method = representation
        self._representation = representation
        self._state = state

    def run(self, *_args, **_kwargs):
        return Job(
            status="DONE",
            result=Result(
                **{
                    self._representation: self._state,
                    "available": frozenset({self._representation}),
                }
            ),
        )


@pytest.mark.parametrize(
    ("representation", "state", "expected_shape"),
    [
        ("statevector", np.zeros(9, dtype=complex), (4,)),
        ("density_matrix", np.zeros((9, 9), dtype=complex), (4, 4)),
    ],
)
def test_estimator_rejects_nonlogical_result_shapes(
    representation, state, expected_shape
):
    estimator = fq.Estimator(_FixedStateBackend(representation, state))

    with pytest.raises(BackendValidationError) as caught:
        estimator.run(_bell(), Observable([("ZZ", 1.0)]))

    message = str(caught.value)
    assert representation in message
    assert str(expected_shape) in message
    assert str(state.shape) in message


def test_qutrit_density_is_rejected_before_binary_expectation_kernels(
    monkeypatch,
):
    density = np.zeros((9, 9), dtype=complex)
    density[4, 4] = 1.0
    estimator = fq.Estimator(_FixedStateBackend("density_matrix", density))

    def kernel_must_not_run(*_args, **_kwargs):
        raise AssertionError("binary expectation kernel received a qutrit state")

    monkeypatch.setattr(
        "fatqat.estimator.expectation_statevector",
        kernel_must_not_run,
    )
    monkeypatch.setattr(
        "fatqat.estimator.expectation_density_matrix",
        kernel_must_not_run,
    )

    with pytest.raises(BackendValidationError, match="density_matrix") as caught:
        estimator.run(_bell(), Observable([("IZ", 1.0), ("ZI", 1.0)]))

    message = str(caught.value)
    assert "(4, 4)" in message
    assert "(9, 9)" in message


def test_unrelated_validation_errors_are_not_rewrapped():
    from fatqat.errors import UnsupportedOperationError

    class Unknown(ops.Operation):
        name = "Unknown"
        num_subsystems = 1

    program = fq.Program(1)
    program.add(Unknown(), 0)

    with pytest.raises(UnsupportedOperationError) as caught:
        _estimator().run(program, Observable([("Z", 1.0)]))
    assert "no single final state" not in str(caught.value)


def test_bad_simulation_config_error_is_not_rewrapped():
    program = fq.Program(1)
    program.add(ops.H, 0)

    with pytest.raises(BackendValidationError) as caught:
        _estimator().run(
            program,
            Observable([("Z", 1.0)]),
            simulation_config={"no_such_option": True},
        )
    assert "no single final state" not in str(caught.value)
