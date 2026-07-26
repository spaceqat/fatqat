"""Numerical cross-validation against Qiskit Aer, ideal and noisy.

Every case builds the same small circuit twice - once as a fatqat `Program`,
once as a Qiskit `QuantumCircuit` - runs both, and compares the resulting
state matrices exactly (``atol=1e-12``, no relative slack): statevectors
under ``method="SV"`` and density matrices under ``method="DM"``. Every
case runs on both fatqat runtimes (the ``runtime`` fixture): ``numpy``, and
``numba`` when the optional dependency is installed (skipped otherwise; CI
installs it, so both axes are mandatory there). The internal parity tests
pin numba against numpy shot-for-shot; this axis pins both against a
reference that cannot share a bug with either. Ideal
circuits compare against ``qiskit.quantum_info`` exact evolution (Qiskit's
canonical reference, no transpiler in the loop); noisy circuits compare
against Aer's density-matrix simulator with native error constructors. Both
libraries use little-endian flat indexing, and both build the state by
applying identical gate matrices to ``|0...0>``, so matrices (including
global phase) must agree elementwise with no permutation.

Noise cases attach fatqat catalog channels and Aer's *native* error
constructors, asserting that fatqat's parameterization means what Aer's
means - not merely that identical Kraus arrays execute identically. Where
the parameterizations deliberately differ (`PhaseDamping`), the conversion
is applied and commented at the test site.

Deliberately out of scope for now: probabilities/counts comparisons (needs
a statistical-tolerance policy), readout error (counts-level, same phase),
statevector execution under noise (stochastic trajectories), qudits (no Aer
analogue), and reset/feedforward dynamics (Aer's semantics differ).

Requires the optional ``qiskit`` dependency group; the module self-skips
without it, exactly like the numba-only tests.
"""

import numpy as np
import pytest

import fatqat as fq

pytest.importorskip("qiskit_aer")

# pylint: disable=wrong-import-position,wrong-import-order  # need importorskip first
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel as AerNoiseModel,
    amplitude_damping_error,
    depolarizing_error,
    phase_damping_error,
    thermal_relaxation_error,
)

# pylint: enable=wrong-import-position,wrong-import-order

_ATOL = 1e-12


@pytest.fixture(params=["numpy", "numba"], name="runtime")
def _runtime(request):
    """Both execution runtimes; the numba axis skips if numba is absent."""
    if request.param == "numba":
        pytest.importorskip("numba")
    return request.param


def _assert_close(ours: np.ndarray, theirs: np.ndarray) -> None:
    theirs = np.asarray(theirs)
    assert ours.shape == theirs.shape
    assert np.allclose(ours, theirs, rtol=0, atol=_ATOL)


# --- runners -----------------------------------------------------------------


def _fatqat_state(program: fq.Program, runtime: str) -> np.ndarray:
    return (
        fq.backends.SimulatorBackend(method="SV", runtime=runtime)
        .run(program, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )


def _fatqat_rho(
    program: fq.Program,
    runtime: str,
    noise: fq.NoiseModel | None = None,
) -> np.ndarray:
    backend = fq.backends.SimulatorBackend(method="DM", runtime=runtime, noise=noise)
    return (
        backend.run(program, result_config={"counts": False, "final_state": True})
        .result()
        .get_density_matrix()
    )


def _qiskit_state(circuit: QuantumCircuit) -> np.ndarray:
    # qiskit.quantum_info exact evolution: Qiskit's canonical reference for
    # ideal circuits - no transpiler pass (which may permute qubit layout),
    # no simulator basis restrictions, every standard gate supported.
    return np.asarray(Statevector(circuit))


def _qiskit_rho(circuit: QuantumCircuit) -> np.ndarray:
    return np.asarray(DensityMatrix(circuit))


def _aer_rho(
    circuit: QuantumCircuit,
    noise_model: AerNoiseModel,
    basis_gates: list[str],
) -> np.ndarray:
    # Noisy reference: Aer's density-matrix simulator. The circuit is built
    # inside `basis_gates` already, so Aer runs it untranspiled and the
    # noise model attaches to exactly the gates the builder named.
    simulator = AerSimulator(
        method="density_matrix", noise_model=noise_model, basis_gates=basis_gates
    )
    run = circuit.copy()
    run.save_density_matrix()
    return np.asarray(simulator.run(run).result().data()["density_matrix"])


# --- ideal circuits ----------------------------------------------------------
# Each builder returns the same circuit twice: (fatqat Program, QuantumCircuit).


def _bell():
    program = fq.Program(2)
    program.add(fq.ops.H, 0)
    program.add(fq.ops.CX, (0, 1))
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    return program, circuit


def _ghz3():
    program = fq.Program(3)
    program.add(fq.ops.H, 0)
    program.add(fq.ops.CX, (0, 1))
    program.add(fq.ops.CX, (1, 2))
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    return program, circuit


def _fixed_gate_coverage():
    # Every shared fixed one-qubit gate, interleaved with CX so per-gate
    # errors cannot cancel against each other.
    program = fq.Program(2)
    circuit = QuantumCircuit(2)
    program.add(fq.ops.H, 0)
    circuit.h(0)
    program.add(fq.ops.X, 1)
    circuit.x(1)
    program.add(fq.ops.CX, (0, 1))
    circuit.cx(0, 1)
    program.add(fq.ops.Y, 0)
    circuit.y(0)
    program.add(fq.ops.Z, 1)
    circuit.z(1)
    program.add(fq.ops.S, 0)
    circuit.s(0)
    program.add(fq.ops.Sdg, 1)
    circuit.sdg(1)
    program.add(fq.ops.CX, (1, 0))
    circuit.cx(1, 0)
    program.add(fq.ops.T, 0)
    circuit.t(0)
    program.add(fq.ops.Tdg, 1)
    circuit.tdg(1)
    program.add(fq.ops.SX, 0)
    circuit.sx(0)
    return program, circuit


def _parametric_coverage():
    # Generic angles only - special values (0, pi/2, pi) can mask convention
    # mismatches behind extra symmetry.
    program = fq.Program(2)
    circuit = QuantumCircuit(2)
    program.add(fq.ops.RX(0.3), 0)
    circuit.rx(0.3, 0)
    program.add(fq.ops.RY(1.1), 1)
    circuit.ry(1.1, 1)
    program.add(fq.ops.CX, (0, 1))
    circuit.cx(0, 1)
    program.add(fq.ops.RZ(0.7), 0)
    circuit.rz(0.7, 0)
    program.add(fq.ops.Phase(0.5), 1)
    circuit.p(0.5, 1)
    program.add(fq.ops.CPhase(0.9), (0, 1))
    circuit.cp(0.9, 0, 1)
    return program, circuit


def _multi_qubit_coverage():
    program = fq.Program(3)
    circuit = QuantumCircuit(3)
    for q in range(3):
        program.add(fq.ops.H, q)
        circuit.h(q)
    program.add(fq.ops.T, 1)
    circuit.t(1)
    program.add(fq.ops.CZ, (0, 1))
    circuit.cz(0, 1)
    program.add(fq.ops.Swap, (1, 2))
    circuit.swap(1, 2)
    program.add(fq.ops.iSwap, (0, 1))
    circuit.iswap(0, 1)
    program.add(fq.ops.CY, (1, 2))
    circuit.cy(1, 2)
    program.add(fq.ops.CS, (0, 2))
    circuit.cs(0, 2)
    program.add(fq.ops.CCX, (0, 1, 2))
    circuit.ccx(0, 1, 2)
    program.add(fq.ops.CSwap, (2, 0, 1))
    circuit.cswap(2, 0, 1)
    return program, circuit


def _qft3():
    # Hand-rolled 3-qubit QFT from H / CPhase / Swap, on a non-trivial input.
    program = fq.Program(3)
    circuit = QuantumCircuit(3)
    program.add(fq.ops.X, 0)
    circuit.x(0)
    program.add(fq.ops.X, 2)
    circuit.x(2)
    program.add(fq.ops.H, 2)
    circuit.h(2)
    program.add(fq.ops.CPhase(np.pi / 2), (1, 2))
    circuit.cp(np.pi / 2, 1, 2)
    program.add(fq.ops.CPhase(np.pi / 4), (0, 2))
    circuit.cp(np.pi / 4, 0, 2)
    program.add(fq.ops.H, 1)
    circuit.h(1)
    program.add(fq.ops.CPhase(np.pi / 2), (0, 1))
    circuit.cp(np.pi / 2, 0, 1)
    program.add(fq.ops.H, 0)
    circuit.h(0)
    program.add(fq.ops.Swap, (0, 2))
    circuit.swap(0, 2)
    return program, circuit


_IDEAL_CASES = {
    "bell": _bell,
    "ghz3": _ghz3,
    "fixed_gates": _fixed_gate_coverage,
    "parametric_gates": _parametric_coverage,
    "multi_qubit_gates": _multi_qubit_coverage,
    "qft3": _qft3,
}


@pytest.mark.parametrize("build", _IDEAL_CASES.values(), ids=_IDEAL_CASES.keys())
def test_statevector_matches_qiskit(build, runtime):
    program, circuit = build()
    _assert_close(_fatqat_state(program, runtime), _qiskit_state(circuit))


@pytest.mark.parametrize("build", _IDEAL_CASES.values(), ids=_IDEAL_CASES.keys())
def test_density_matrix_matches_qiskit(build, runtime):
    program, circuit = build()
    _assert_close(_fatqat_rho(program, runtime), _qiskit_rho(circuit))


# --- noisy circuits (density matrices; Aer native error constructors) --------


def _aer_model(error, gate_names):
    model = AerNoiseModel()
    model.add_all_qubit_quantum_error(error, gate_names)
    return model


def test_depolarizing_on_two_qubit_gate_matches_aer(runtime):
    program, circuit = _bell()
    noise = fq.NoiseModel()
    noise.add_channel(fq.ops.CX, fq.noise.Depolarizing(p=0.1))
    aer_model = _aer_model(depolarizing_error(0.1, 2), ["cx"])

    _assert_close(
        _fatqat_rho(program, runtime, noise),
        _aer_rho(circuit, aer_model, basis_gates=["h", "cx"]),
    )


def test_amplitude_damping_matches_aer(runtime):
    program, circuit = _bell()
    noise = fq.NoiseModel()
    noise.add_channel(fq.ops.H, fq.noise.AmplitudeDamping(gammas=(0.2,)))
    aer_model = _aer_model(amplitude_damping_error(0.2), ["h"])

    _assert_close(
        _fatqat_rho(program, runtime, noise),
        _aer_rho(circuit, aer_model, basis_gates=["h", "cx"]),
    )


def test_phase_damping_matches_aer(runtime):
    # Parameter conventions differ by design: fatqat's PhaseDamping(p) leaves
    # qubit coherence at (1 - p), Aer's phase_damping_error(g) at sqrt(1 - g);
    # they describe the same channel family via g = 1 - (1 - p)**2.
    p = 0.3
    program, circuit = _bell()
    noise = fq.NoiseModel()
    noise.add_channel(fq.ops.H, fq.noise.PhaseDamping(p=p))
    aer_model = _aer_model(phase_damping_error(1 - (1 - p) ** 2), ["h"])

    _assert_close(
        _fatqat_rho(program, runtime, noise),
        _aer_rho(circuit, aer_model, basis_gates=["h", "cx"]),
    )


def test_relaxation_channels_match_aer_thermal_relaxation(runtime):
    t1, t2, duration = 60e-6, 90e-6, 5e-6
    program, circuit = _bell()
    damping, dephasing = fq.noise.relaxation_channels(t1, t2, duration)
    noise = fq.NoiseModel()
    noise.add_channel(fq.ops.H, damping)
    noise.add_channel(fq.ops.H, dephasing)
    aer_model = _aer_model(thermal_relaxation_error(t1, t2, duration), ["h"])

    _assert_close(
        _fatqat_rho(program, runtime, noise),
        _aer_rho(circuit, aer_model, basis_gates=["h", "cx"]),
    )


def test_stacked_channels_compose_in_registration_order(runtime):
    # Depolarizing and amplitude damping do NOT commute (damping is not
    # unital), so this case pins the order convention for real: fatqat
    # applies stacked channels in registration order, matching Aer's
    # error_first.compose(error_second).
    p, gamma = 0.2, 0.3
    program, circuit = _bell()
    noise = fq.NoiseModel()
    noise.add_channel(fq.ops.H, fq.noise.Depolarizing(p=p))
    noise.add_channel(fq.ops.H, fq.noise.AmplitudeDamping(gammas=(gamma,)))
    composed = depolarizing_error(p, 1).compose(amplitude_damping_error(gamma))
    aer_model = _aer_model(composed, ["h"])

    ours = _fatqat_rho(program, runtime, noise)
    _assert_close(ours, _aer_rho(circuit, aer_model, basis_gates=["h", "cx"]))

    # The reversed composition must differ - otherwise this test could not
    # detect an order-convention mistake in either library.
    reversed_composed = amplitude_damping_error(gamma).compose(depolarizing_error(p, 1))
    reversed_model = _aer_model(reversed_composed, ["h"])
    reversed_rho = _aer_rho(circuit, reversed_model, basis_gates=["h", "cx"])
    assert not np.allclose(ours, reversed_rho, rtol=0, atol=_ATOL)
