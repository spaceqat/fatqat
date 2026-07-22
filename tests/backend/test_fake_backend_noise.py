"""Fake 4x4 backend's calibration-derived default noise (from-backend flow)."""

import numpy as np

import fatqat as fq
from fatqat.backends import FakeSuperconducting4x4Backend
from fatqat.noise import AmplitudeDamping, Depolarizing, PhaseDamping


def _sx_sx_program():
    program = fq.Program(1, 1)
    program.add(fq.ops.SX, 0)
    program.add(fq.ops.SX, 0)
    program.add_measurement(0, 0)
    return program


def test_backend_is_ideal_by_default():
    counts = (
        FakeSuperconducting4x4Backend()
        .run(_sx_sx_program(), shots=100, seed=1)
        .result()
        .get_counts()
    )

    assert counts == {"1": 100}  # SX SX = X up to phase, no noise


def test_default_noise_model_is_fully_supported():
    model = FakeSuperconducting4x4Backend.default_noise_model()
    report = FakeSuperconducting4x4Backend().validate_noise(model)

    assert report.supported is True
    assert set(report.accepted_sources) == {
        "AmplitudeDamping",
        "Depolarizing",
        "PhaseDamping",
        "readout_error",
    }
    # Authored before any user program exists; applies to any program.
    assert model.channel_types() == frozenset(
        {AmplitudeDamping, Depolarizing, PhaseDamping}
    )


def test_noisy_backend_leaks_errors_but_stays_mostly_correct():
    backend = FakeSuperconducting4x4Backend(
        noise=FakeSuperconducting4x4Backend.default_noise_model()
    )
    shots = 4000
    counts = backend.run(_sx_sx_program(), shots=shots, seed=1).result().get_counts()

    # Readout p10 = 0.04 dominates the tiny relaxation rates.
    assert 0 < counts.get("0", 0) < 0.10 * shots


def test_rz_stays_noise_free():
    backend = FakeSuperconducting4x4Backend(
        noise=FakeSuperconducting4x4Backend.default_noise_model()
    )
    program = fq.Program(1)
    program.add(fq.ops.RZ(0.7), 0)  # virtual gate: no relaxation attached
    state = (
        backend.run(program, result_config={"counts": False, "statevector": True})
        .result()
        .get_statevector()
    )

    assert np.isclose(abs(state[0]), 1.0)


def test_default_noise_model_is_a_fresh_extensible_model():
    first = FakeSuperconducting4x4Backend.default_noise_model()
    second = FakeSuperconducting4x4Backend.default_noise_model()
    first.add_noise(fq.ops.SX, Depolarizing(p=0.5))

    assert Depolarizing in first.channel_types()
    # Each call builds an independent model; user edits never leak back.
    program = fq.Program(1)
    backend = FakeSuperconducting4x4Backend()
    assert not any(
        isinstance(c, Depolarizing) and c.p == 0.5
        for c in second.channels_for(
            fq.ops.SX,
            (program.qreg[0][0],),
            backend._resolve_resource_layout(program),
        )
    )
