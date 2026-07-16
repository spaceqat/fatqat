"""Catalog channel rules: Kraus counts, CPTP completeness, and channel action."""

import numpy as np
import pytest

import fatqat as fq
from fatqat.errors import BackendValidationError
from fatqat.noise.catalog import (
    AmplitudeDamping,
    Depolarizing,
    PhaseDamping,
    amplitude_damping_rule,
    depolarizing_rule,
    phase_damping_rule,
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


def test_amplitude_damping_qubit_decay():
    gamma = 0.4
    kraus_ops = amplitude_damping_rule(
        AmplitudeDamping(gammas=(gamma,)), targets=_refs(2)
    )

    _assert_cptp(kraus_ops, 2)
    excited = np.diag([0.0, 1.0]).astype(complex)
    assert np.allclose(_apply(kraus_ops, excited), np.diag([gamma, 1 - gamma]))


def test_amplitude_damping_qutrit_ladder_decays_one_level():
    kraus_ops = amplitude_damping_rule(
        AmplitudeDamping(gammas=(0.2, 0.5)), targets=_refs(3)
    )

    _assert_cptp(kraus_ops, 3)
    # Level 2 population moves only to level 1 (ladder), never straight to 0.
    top = np.diag([0.0, 0.0, 1.0]).astype(complex)
    out = _apply(kraus_ops, top)
    assert np.allclose(np.diag(out), [0.0, 0.5, 0.5])


def test_amplitude_damping_rate_count_must_match_dimension():
    with pytest.raises(BackendValidationError, match="decay rate"):
        amplitude_damping_rule(AmplitudeDamping(gammas=(0.1,)), targets=_refs(3))


@pytest.mark.parametrize("dim", [2, 3])
def test_phase_damping_preserves_populations_and_decays_coherence(dim):
    p = 0.6
    kraus_ops = phase_damping_rule(PhaseDamping(p=p), targets=_refs(dim))

    assert len(kraus_ops) == dim
    _assert_cptp(kraus_ops, dim)
    rho = _random_rho(dim)
    out = _apply(kraus_ops, rho)
    assert np.allclose(np.diag(out), np.diag(rho))
    if dim == 2:
        assert np.allclose(out[0, 1], (1 - p) * rho[0, 1])


def test_amplitude_damping_rejects_multi_target_gates():
    with pytest.raises(BackendValidationError, match="single-subsystem"):
        amplitude_damping_rule(AmplitudeDamping(gammas=(0.1,)), targets=_refs(2, 2))


def test_phase_damping_rejects_multi_target_gates():
    with pytest.raises(BackendValidationError, match="single-subsystem"):
        phase_damping_rule(PhaseDamping(p=0.1), targets=_refs(2, 2))


@pytest.mark.parametrize("bad_p", [-0.1, 1.5, True, "0.1"])
def test_descriptor_probability_validation(bad_p):
    with pytest.raises(ValueError):
        Depolarizing(p=bad_p)
    with pytest.raises(ValueError):
        PhaseDamping(p=bad_p)


def test_amplitude_damping_descriptor_validation():
    with pytest.raises(ValueError):
        AmplitudeDamping(gammas=())
    with pytest.raises(ValueError):
        AmplitudeDamping(gammas=(0.2, 1.4))


def test_descriptors_hold_parameters_not_arrays():
    channel = Depolarizing(p=0.1)
    assert not any(
        isinstance(v, np.ndarray) for v in vars(channel).values()
    ), "descriptors must never precompute or store Kraus arrays"
