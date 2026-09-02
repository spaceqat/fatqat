"""Catalog channel rules: Kraus counts, CPTP completeness, and channel action."""

import numpy as np
import pytest

import fatqat as fq
from fatqat.errors import BackendValidationError
from fatqat.implementation.matrices import _X
from fatqat.noise.catalog import (
    AmplitudeDamping,
    Depolarizing,
    PauliChannel,
    PhaseDamping,
    _pauli_string_matrix,
    amplitude_damping_rule,
    depolarizing_rule,
    pauli_channel_rule,
    phase_damping_rule,
)
from fatqat.noise.lindblad import (
    depolarizing_lindblad_rule,
    phase_damping_lindblad_rule,
)


def _refs(*dims):
    return tuple(fq.QuantumRegister(1, dim=d)[0] for d in dims)


def _apply(kraus_ops, rho):
    return sum(k @ rho @ k.conj().T for k in kraus_ops)


def _random_rho(dim, seed=11):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    rho = a @ a.conj().T
    return rho / np.trace(rho)


def _assert_cptp(kraus_ops, dim):
    completeness = sum(k.conj().T @ k for k in kraus_ops)
    assert np.allclose(completeness, np.eye(dim))


@pytest.mark.parametrize("dim", [2, 3])
def test_depolarizing_action_matches_closed_form(dim):
    p = 0.3
    kraus_ops = depolarizing_rule(Depolarizing(p=p), targets=_refs(dim))

    assert len(kraus_ops) == dim**2
    _assert_cptp(kraus_ops, dim)
    rho = _random_rho(dim)
    expected = (1 - p) * rho + p * np.eye(dim) / dim
    assert np.allclose(_apply(kraus_ops, rho), expected)


def test_depolarizing_acts_jointly_on_multi_subsystem_targets():
    p = 0.2
    kraus_ops = depolarizing_rule(Depolarizing(p=p), targets=_refs(2, 2))

    assert len(kraus_ops) == 16
    _assert_cptp(kraus_ops, 4)
    rho = _random_rho(4)
    expected = (1 - p) * rho + p * np.eye(4) / 4
    assert np.allclose(_apply(kraus_ops, rho), expected)


def test_depolarizing_dual_mode_contract_is_keyword_only_and_mode_specific():
    with pytest.raises(TypeError):
        Depolarizing(0.1)
    with pytest.raises(ValueError, match="exactly one"):
        Depolarizing()
    with pytest.raises(ValueError, match="exactly one"):
        Depolarizing(p=0.1, rate=0.2)

    finite = Depolarizing(p=0.1)
    continuous = Depolarizing(rate=0.2)
    assert (finite.p, finite.rate, finite.num_subsystems) == (0.1, None, None)
    assert (continuous.p, continuous.rate, continuous.num_subsystems) == (
        None,
        0.2,
        1,
    )


@pytest.mark.parametrize("bad_rate", [-0.1, True, "0.1", np.inf, np.nan])
def test_depolarizing_rate_validation(bad_rate):
    with pytest.raises(ValueError, match="rate"):
        Depolarizing(rate=bad_rate)


def test_depolarizing_explicit_modes_round_trip():
    duration = 2.0
    rate = 0.05
    probability = Depolarizing(rate=rate).as_probability(duration)
    assert probability == pytest.approx(1 - np.exp(-rate * duration))
    assert Depolarizing(p=probability).as_rate(duration) == pytest.approx(rate)


def test_matrix_depolarizing_rule_rejects_rate_mode():
    with pytest.raises(BackendValidationError, match="rate mode"):
        depolarizing_rule(Depolarizing(rate=0.1), targets=_refs(2))


@pytest.mark.parametrize("dim", [2, 3])
def test_depolarizing_lindblad_rule_matches_dimension_generic_generator(dim):
    rate = 0.3
    operators = depolarizing_lindblad_rule(
        Depolarizing(rate=rate), physical_dimension=dim
    )
    rho = _random_rho(dim)
    generated = sum(
        operator @ rho @ operator.conj().T
        - 0.5
        * (operator.conj().T @ operator @ rho + rho @ operator.conj().T @ operator)
        for operator in operators
    )

    assert len(operators) == dim**2 - 1
    assert all(operator.shape == (dim, dim) for operator in operators)
    expected = rate * (np.trace(rho) * np.eye(dim) / dim - rho)
    assert np.allclose(generated, expected)


def test_amplitude_damping_qubit_decay():
    gamma = 0.4
    kraus_ops = amplitude_damping_rule(AmplitudeDamping(p=gamma), targets=_refs(2))

    _assert_cptp(kraus_ops, 2)
    excited = np.diag([0.0, 1.0]).astype(complex)
    assert np.allclose(_apply(kraus_ops, excited), np.diag([gamma, 1 - gamma]))


def test_amplitude_damping_rejects_non_qubit_target():
    with pytest.raises(BackendValidationError, match="qubits only"):
        amplitude_damping_rule(AmplitudeDamping(p=0.1), targets=_refs(3))


def test_amplitude_damping_rejects_positional_construction():
    with pytest.raises(TypeError):
        AmplitudeDamping(0.1)  # noqa: pyright-ignore - deliberately positional


def test_phase_damping_rejects_positional_construction():
    with pytest.raises(TypeError):
        PhaseDamping(0.1)  # noqa: pyright-ignore - deliberately positional


def test_amplitude_damping_requires_exactly_one_scalar_mode():
    with pytest.raises(ValueError, match="exactly one"):
        AmplitudeDamping()
    with pytest.raises(ValueError, match="exactly one"):
        AmplitudeDamping(p=0.1, rate=0.1)


def test_amplitude_damping_rule_rejects_rate_mode():
    with pytest.raises(BackendValidationError, match="rate mode"):
        amplitude_damping_rule(AmplitudeDamping(rate=0.1), targets=_refs(2))


def test_phase_damping_requires_exactly_one_parameterization():
    with pytest.raises(ValueError, match="exactly one"):
        PhaseDamping()
    with pytest.raises(ValueError, match="exactly one"):
        PhaseDamping(p=0.1, rate=0.1)
    with pytest.raises(ValueError, match="exactly one"):
        PhaseDamping(p=0.1, t_phi=2.0)


def test_phase_damping_normalizes_t_phi_to_rate():
    assert PhaseDamping(t_phi=20.0) == PhaseDamping(rate=0.05)


def test_phase_damping_t_phi_preserves_multilevel_number_operator_convention():
    (operator,) = phase_damping_lindblad_rule(
        PhaseDamping(t_phi=20.0), physical_dimension=3
    )

    assert np.allclose(operator, np.sqrt(2 / 20.0) * np.diag([0.0, 1.0, 2.0]))


@pytest.mark.parametrize(
    "bad_t_phi", [0.0, -1.0, True, "2", float("inf"), float("nan")]
)
def test_phase_damping_t_phi_validation(bad_t_phi):
    with pytest.raises(ValueError, match="t_phi"):
        PhaseDamping(t_phi=bad_t_phi)


@pytest.mark.parametrize(
    "bad_rate", [-0.1, True, "0.1", (0.1,), float("inf"), float("nan")]
)
def test_amplitude_damping_rate_validation(bad_rate):
    with pytest.raises(ValueError):
        AmplitudeDamping(rate=bad_rate)


@pytest.mark.parametrize("bad_rate", [-0.1, True, "0.1", float("inf"), float("nan")])
def test_phase_damping_rate_validation(bad_rate):
    with pytest.raises(ValueError):
        PhaseDamping(rate=bad_rate)


def test_amplitude_damping_as_probability_and_as_rate_round_trip():
    duration = 2.0
    rate = 0.05
    channel = AmplitudeDamping(rate=rate)
    p = channel.as_probability(duration)
    assert p == pytest.approx(1 - np.exp(-rate * duration))
    back = AmplitudeDamping(p=p).as_rate(duration)
    assert back == pytest.approx(rate)


def test_phase_damping_as_probability_and_as_rate_round_trip():
    duration = 2.0
    rate = 0.05
    channel = PhaseDamping(rate=rate)
    p = channel.as_probability(duration)
    assert p == pytest.approx(1 - np.exp(-rate * duration))
    back = PhaseDamping(p=p).as_rate(duration)
    assert back == pytest.approx(rate)


def test_zero_duration_and_zero_value_identity_conversions():
    assert PhaseDamping(rate=0.0).as_probability(0.0) == 0.0
    assert PhaseDamping(p=0.0).as_rate(0.0) == 0.0
    # Any finite rate converts to probability 0 over zero duration.
    assert PhaseDamping(rate=123.0).as_probability(0.0) == 0.0


def test_zero_rate_and_zero_probability_convert_identically():
    assert PhaseDamping(rate=0.0).as_probability(5.0) == 0.0
    assert PhaseDamping(p=0.0).as_rate(5.0) == 0.0


def test_probability_one_has_no_finite_rate():
    with pytest.raises(ValueError, match="no finite rate"):
        PhaseDamping(p=1.0).as_rate(1.0)


def test_nonzero_probability_at_zero_duration_has_no_finite_rate():
    with pytest.raises(ValueError, match="no finite rate"):
        PhaseDamping(p=0.5).as_rate(0.0)


@pytest.mark.parametrize("dim", [2, 3])
def test_phase_damping_preserves_populations_and_decays_coherence(dim):
    p = 0.6
    kraus_ops = phase_damping_rule(PhaseDamping(p=p), targets=_refs(dim))

    assert len(kraus_ops) == dim
    _assert_cptp(kraus_ops, dim)
    rho = _random_rho(dim)
    expected = (1 - p) * rho + p * np.diag(np.diag(rho))
    assert np.allclose(_apply(kraus_ops, rho), expected)


def test_amplitude_damping_rejects_multi_target_gates():
    with pytest.raises(BackendValidationError, match="single-subsystem"):
        amplitude_damping_rule(AmplitudeDamping(p=0.1), targets=_refs(2, 2))


def test_phase_damping_rejects_multi_target_gates():
    with pytest.raises(BackendValidationError, match="single-subsystem"):
        phase_damping_rule(PhaseDamping(p=0.1), targets=_refs(2, 2))


@pytest.mark.parametrize("bad_p", [-0.1, 1.5, True, "0.1", np.inf, np.nan])
def test_descriptor_probability_validation(bad_p):
    with pytest.raises(ValueError):
        Depolarizing(p=bad_p)
    with pytest.raises(ValueError):
        AmplitudeDamping(p=bad_p)
    with pytest.raises(ValueError):
        PhaseDamping(p=bad_p)


def test_descriptors_hold_parameters_not_arrays():
    channel = Depolarizing(p=0.1)
    assert not any(
        isinstance(v, np.ndarray) for v in vars(channel).values()
    ), "descriptors must never precompute or store Kraus arrays"


# --- PauliChannel ---


def test_pauli_channel_leads_with_the_unassigned_identity_weight():
    channel = PauliChannel({"X": 0.01, "Z": 0.02})

    assert channel.terms == (("I", 0.97), ("X", 0.01), ("Z", 0.02))
    assert channel.num_subsystems == 1


def test_pauli_channel_accepts_a_pair_sequence_and_an_explicit_identity():
    from_pairs = PauliChannel([("XX", 0.05), ("ZI", 0.01)])
    with_identity = PauliChannel({"II": 0.94, "XX": 0.05, "ZI": 0.01})

    assert from_pairs.terms == with_identity.terms
    assert from_pairs.num_subsystems == 2


def test_pauli_channel_rejects_an_identity_the_other_terms_contradict():
    with pytest.raises(ValueError, match="conflicts"):
        PauliChannel({"I": 0.5, "X": 0.1})


@pytest.mark.parametrize(
    "bad_terms",
    [
        {},
        {"X": 0.6, "Z": 0.7},
        {"X": 0.1, "YY": 0.1},
        {"Q": 0.1},
        {"": 0.1},
        {"X": 1.5},
        [("X", 0.1), ("X", 0.2)],
    ],
)
def test_pauli_channel_descriptor_validation(bad_terms):
    with pytest.raises(ValueError):
        PauliChannel(bad_terms)


@pytest.mark.parametrize("width", [1, 2])
def test_pauli_channel_action_matches_its_term_sum(width):
    terms = {"X" * width: 0.1, "Z" * width: 0.05}
    channel = PauliChannel(terms)
    kraus_ops = pauli_channel_rule(channel, targets=_refs(*([2] * width)))

    dim = 2**width
    assert len(kraus_ops) == 3
    _assert_cptp(kraus_ops, dim)
    rho = _random_rho(dim)
    expected = sum(
        p * _pauli_string_matrix(s) @ rho @ _pauli_string_matrix(s).conj().T
        for s, p in channel.terms
    )
    assert np.allclose(_apply(kraus_ops, rho), expected)


def test_pauli_channel_reproduces_depolarizing_at_one_qubit():
    # Depolarizing(p) at d=2 is the uniform Pauli channel with weight p/4 each.
    p = 0.3
    pauli = pauli_channel_rule(
        PauliChannel({"X": p / 4, "Y": p / 4, "Z": p / 4}), targets=_refs(2)
    )
    rho = _random_rho(2)

    expected = _apply(depolarizing_rule(Depolarizing(p=p), targets=_refs(2)), rho)
    assert np.allclose(_apply(pauli, rho), expected)


def test_pauli_string_reads_left_to_right_in_target_order():
    # string[0] describes targets[0], which is the local matrix's MSB - the
    # same convention gate matrices use, and the reverse of Qiskit's Pauli.
    assert np.allclose(_pauli_string_matrix("XI"), np.kron(_X, np.eye(2)))
    assert np.allclose(_pauli_string_matrix("IX"), np.kron(np.eye(2), _X))


def test_pauli_channel_rejects_non_qubit_targets():
    with pytest.raises(BackendValidationError, match="qubits only"):
        pauli_channel_rule(PauliChannel({"X": 0.1}), targets=_refs(3))


def test_pauli_channel_rejects_a_target_count_its_terms_do_not_cover():
    with pytest.raises(BackendValidationError):
        pauli_channel_rule(PauliChannel({"X": 0.1}), targets=_refs(2, 2))


def test_pauli_channel_holds_parameters_not_arrays():
    channel = PauliChannel({"X": 0.1})
    assert not any(
        isinstance(v, np.ndarray) for v in vars(channel).values()
    ), "descriptors must never precompute or store Kraus arrays"
