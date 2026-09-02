"""The explicit transition-relaxation descriptor and finite realization."""

import numpy as np
import pytest

import fatqat as fq
from fatqat.errors import BackendValidationError
from fatqat.noise import TransitionRelaxation
from fatqat.noise.catalog import amplitude_damping_rule
from fatqat.noise.transition_relaxation import transition_relaxation_rule


def _target(dimension):
    return (fq.QuantumRegister(1, dim=dimension)[0],)


def test_transition_relaxation_requires_one_strength_mode():
    with pytest.raises(ValueError, match="exactly one"):
        TransitionRelaxation(coefficients={(1, 0): 1})
    with pytest.raises(ValueError, match="exactly one"):
        TransitionRelaxation(p=0.1, rate=0.2, coefficients={(1, 0): 1})

    assert TransitionRelaxation(p=0.0, coefficients={(1, 0): 1}).p == 0.0
    assert TransitionRelaxation(rate=0.0, coefficients={(1, 0): 1}).rate == 0.0


@pytest.mark.parametrize(("strength", "value"), [("p", 1.1), ("rate", -0.1)])
def test_transition_relaxation_validates_its_strength(strength, value):
    with pytest.raises(ValueError, match=strength):
        TransitionRelaxation(
            **{strength: value},
            coefficients={(1, 0): 1},
        )


@pytest.mark.parametrize(
    "coefficients",
    [
        {},
        {(0, 0): 1},
        {(-1, 0): 1},
        {(True, 0): 1},
        {(1, 0): 0},
        {(1, 0): np.inf},
        {(1,): 1},
    ],
)
def test_transition_relaxation_validates_coefficients(coefficients):
    with pytest.raises((TypeError, ValueError)):
        TransitionRelaxation(p=0.1, coefficients=coefficients)


def test_transition_relaxation_copies_and_canonicalizes_coefficients():
    source = {(2, 1): np.sqrt(2), (1, 0): 1j}
    channel = TransitionRelaxation(p=0.1, coefficients=source)
    source[(1, 0)] = 9

    assert channel.coefficients[(1, 0)] == 1j
    assert channel == TransitionRelaxation(
        p=0.1, coefficients={(1, 0): 1j, (2, 1): np.sqrt(2)}
    )
    with pytest.raises(TypeError):
        channel.coefficients[(1, 0)] = 2


def test_finite_transition_preserves_authored_coefficient_scale():
    density_matrix = np.diag([0.0, 0.0, 1.0])

    def destination_population(coefficient):
        channel = TransitionRelaxation(
            p=0.2,
            coefficients={(2, 1): coefficient},
        )
        kraus_ops = transition_relaxation_rule(channel, targets=_target(3))
        output = sum(kraus @ density_matrix @ kraus.conj().T for kraus in kraus_ops)
        return output[1, 1].real

    assert destination_population(1) == pytest.approx(0.2)
    assert destination_population(np.sqrt(2)) == pytest.approx(0.4)


def test_finite_transition_validates_the_physical_dimension():
    channel = TransitionRelaxation(p=0.1, coefficients={(2, 0): 1})
    with pytest.raises(BackendValidationError, match="physical dimension 2"):
        transition_relaxation_rule(channel, targets=_target(2))


def test_finite_transition_with_nondiagonal_gram_matrix_is_complete():
    channel = TransitionRelaxation(p=0.2, coefficients={(1, 0): 1, (2, 0): 1j})
    kraus_ops = transition_relaxation_rule(channel, targets=_target(3))
    completeness = sum(kraus.conj().T @ kraus for kraus in kraus_ops)

    assert np.allclose(completeness, np.eye(3))


def test_finite_transition_rejects_an_overfull_jump():
    channel = TransitionRelaxation(p=0.6, coefficients={(1, 0): np.sqrt(2)})

    with pytest.raises(BackendValidationError, match="trace-preserving"):
        transition_relaxation_rule(channel, targets=_target(2))


def test_single_qubit_transition_matches_amplitude_damping():
    probability = 0.3
    target = _target(2)
    channel = TransitionRelaxation(p=probability, coefficients={(1, 0): 1})
    transition_kraus = transition_relaxation_rule(channel, targets=target)
    amplitude_kraus = amplitude_damping_rule(
        fq.noise.AmplitudeDamping(p=probability), targets=target
    )

    state = np.array([np.sqrt(0.4), np.sqrt(0.6) * 1j])
    density_matrix = np.outer(state, state.conj())
    transition_output = sum(
        kraus @ density_matrix @ kraus.conj().T for kraus in transition_kraus
    )
    amplitude_output = sum(
        kraus @ density_matrix @ kraus.conj().T for kraus in amplitude_kraus
    )

    assert np.allclose(transition_output, amplitude_output)
