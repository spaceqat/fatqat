"""QEC17 device constraints and individual physical calibration behavior."""

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.errors import BackendValidationError, UnsupportedOperationError
from fatqat.noise import Depolarizing, ThermalRelaxation
from fatqat.simulator import SCQubitQEC17Simulator
from fatqat.simulator._qec17_calibration import (
    _QEC17_COUPLER_CALIBRATIONS,
    _QEC17_QUBIT_CALIBRATIONS,
)


def _layout(program, sites):
    refs = [ref for register in program.quantum_registers for ref in register]
    return fq.ResourceLayout(dict(zip(refs, sites, strict=True)))


def _density(program, sites, *, initial_state=None, runtime="numpy", noisy=True):
    backend = SCQubitQEC17Simulator(
        method="density_matrix",
        runtime=runtime,
        noise=SCQubitQEC17Simulator.default_noise_model() if noisy else None,
    )
    return (
        backend.run(
            program,
            shots=0,
            resource_layout=_layout(program, sites),
            initial_state=initial_state,
            result_config={"counts": False, "final_state": True},
        )
        .result()
        .get_density_matrix()
    )


def test_native_gates_and_fixed_topology():
    backend = SCQubitQEC17Simulator(runtime="numpy")
    native = backend.implementation_map
    assert backend.device_sites == tuple(range(17))
    assert {operation.name for operation in native.supported_operations()} == {
        "I",
        "H",
        "HY",
        "X",
        "MX",
        "Y",
        "MY",
        "Z",
        "MZ",
        "S",
        "Sdg",
        "T",
        "XHalf",
        "MXHalf",
        "YHalf",
        "MYHalf",
        "XYHalf",
        "MXYHalf",
        "MXMYHalf",
        "XMYHalf",
        "RZ",
        "SU2",
        "CZ",
    }
    expected = {
        (0, 9),
        (0, 16),
        (1, 9),
        (1, 10),
        (1, 13),
        (2, 10),
        (2, 13),
        (3, 9),
        (3, 11),
        (3, 16),
        (4, 9),
        (4, 10),
        (4, 11),
        (4, 12),
        (5, 10),
        (5, 12),
        (5, 14),
        (6, 11),
        (6, 15),
        (7, 11),
        (7, 12),
        (7, 15),
        (8, 12),
        (8, 14),
    }
    assert native.device_operands_for(ops.CZ) == expected | {
        edge[::-1] for edge in expected
    }
    native.remove(ops.H)
    assert backend.implementation_map.supports(ops.H)


@pytest.mark.parametrize("gate", [ops.CX, ops.SX, ops.RX(0.3), ops.Tdg])
def test_rejects_non_native_operations(gate):
    program = fq.Program(2)
    program.add(gate, (0, 1) if gate is ops.CX else 0)
    with pytest.raises(UnsupportedOperationError):
        SCQubitQEC17Simulator(runtime="numpy").run(program)


def test_rejects_disconnected_edge():
    program = fq.Program(2)
    program.add(ops.CZ, (0, 1))
    with pytest.raises(UnsupportedOperationError):
        SCQubitQEC17Simulator(runtime="numpy").run(program)


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("size,dim", [(18, 2), (1, 3)])
def test_capacity_and_dimensions_with_default_or_explicit_layout(explicit, size, dim):
    program = fq.Program([fq.QuantumRegister(size, dim=dim)])
    options = {"resource_layout": _layout(program, range(size))} if explicit else {}
    with pytest.raises(BackendValidationError):
        SCQubitQEC17Simulator(runtime="numpy").run(program, **options)


def test_grid_register_uses_explicit_device_labels():
    register = fq.GridRegister(1, 2)
    program = fq.Program([register])
    program.add(ops.H, register[0])
    program.add(ops.CZ, (register[0], register[1]))
    matrix = _density(program, (16, 0), noisy=False)
    state = np.array([1, 0, 1, 0]) / np.sqrt(2)
    np.testing.assert_allclose(matrix, np.outer(state, state))


def test_snapshot_covers_each_site_and_edge_once_with_physical_values():
    assert [item.qubit for item in _QEC17_QUBIT_CALIBRATIONS] == list(range(17))
    edges = [item.edge for item in _QEC17_COUPLER_CALIBRATIONS]
    assert len(edges) == len(set(edges)) == 24
    for item in _QEC17_QUBIT_CALIBRATIONS:
        ThermalRelaxation(t1=item.t1, t2=item.t2_echo)
        assert item.t2_star > 0
        assert 0 <= item.xeb_fidelity <= 1
        assert 0 <= item.f00 <= 1
        assert 0 <= item.f11 <= 1
    assert all(0 <= item.cz_fidelity <= 1 for item in _QEC17_COUPLER_CALIBRATIONS)


@pytest.mark.parametrize("calibration", _QEC17_QUBIT_CALIBRATIONS)
def test_driven_gate_uses_its_physical_qubit_fidelity(calibration):
    program = fq.Program(1)
    program.add(ops.X, 0)
    actual = _density(program, (calibration.qubit,))
    # For a one-qubit depolarizing channel after X|0>, P(1) is F.
    fidelity = calibration.xeb_fidelity
    np.testing.assert_allclose(actual, np.diag([1 - fidelity, fidelity]), atol=1e-14)


@pytest.mark.parametrize("sites", [(0, 4), (4, 0)])
def test_noise_follows_physical_layout_instead_of_logical_index(sites):
    program = fq.Program(2)
    program.add(ops.X, program.quantum_registers[0].all())
    actual = _density(program, sites)
    fidelities = {0: 0.99948, 4: 0.99900}
    probabilities = [np.array([1 - fidelities[q], fidelities[q]]) for q in sites]
    np.testing.assert_allclose(np.diag(actual), np.kron(*probabilities), atol=1e-14)


@pytest.mark.parametrize("calibration", _QEC17_COUPLER_CALIBRATIONS)
@pytest.mark.parametrize("reverse", [False, True])
def test_cz_uses_each_edges_fidelity_in_both_directions(calibration, reverse):
    sites = calibration.edge[::-1] if reverse else calibration.edge
    program = fq.Program(2)
    program.add(ops.CZ, (0, 1))
    actual = _density(program, sites, initial_state=np.ones(4) / 2)
    ideal = np.array([1, 1, 1, -1]) / 2
    probability = 4 * (1 - calibration.cz_fidelity) / 3
    expected = (1 - probability) * np.outer(ideal, ideal) + probability * np.eye(4) / 4
    np.testing.assert_allclose(actual, expected, atol=1e-14)


@pytest.mark.parametrize(
    "site,t1,t2", [(0, 38.16e-6, 7.85e-6), (16, 23.28e-6, 8.62e-6)]
)
def test_explicit_idle_obeys_individual_t1_and_t2e(site, t1, t2):
    program = fq.Program(1)
    for _ in range(50):
        program.add(ops.I, 0)
    actual = _density(program, (site,), initial_state=np.ones(2) / np.sqrt(2))
    population = 0.5 * np.exp(-1e-6 / t1)
    coherence = 0.5 * np.exp(-1e-6 / t2)
    np.testing.assert_allclose(
        actual, [[1 - population, coherence], [coherence, population]], atol=1e-14
    )


@pytest.mark.parametrize("gate", [ops.Z, ops.MZ, ops.S, ops.Sdg, ops.T, ops.RZ(0.4)])
def test_virtual_z_has_no_default_noise(gate):
    program = fq.Program(1)
    program.add(gate, 0)
    initial = np.ones(2) / np.sqrt(2)
    np.testing.assert_allclose(
        _density(program, (4,), initial_state=initial),
        _density(program, (4,), initial_state=initial, noisy=False),
        atol=1e-14,
    )


@pytest.mark.parametrize("site,f00,f11", [(4, 0.991, 0.978), (15, 0.999, 0.987)])
@pytest.mark.parametrize("bit", [0, 1])
def test_readout_uses_individual_confusion_matrix(site, f00, f11, bit):
    program = fq.Program(1, 1)
    program.measure(0, 0)
    backend = SCQubitQEC17Simulator(
        runtime="numpy", noise=SCQubitQEC17Simulator.default_noise_model()
    )
    counts = (
        backend.run(
            program,
            shots=20000,
            resource_layout=_layout(program, (site,)),
            initial_state=np.eye(2)[bit],
            simulation_config={"seed": 123},
        )
        .result()
        .get_counts()
    )
    expected_error = 1 - (f00 if bit == 0 else f11)
    assert abs(counts.get(str(1 - bit), 0) / 20000 - expected_error) < 0.003


def test_model_instances_and_backend_copy_are_independent():
    first = SCQubitQEC17Simulator.default_noise_model()
    second = SCQubitQEC17Simulator.default_noise_model()
    backend = SCQubitQEC17Simulator(method="DM", runtime="numpy", noise=first)
    first.add(Depolarizing(p=1), operation=ops.RZ)
    program = fq.Program(1)
    program.add(ops.RZ(0), 0)
    options = {"shots": 0, "result_config": {"counts": False, "final_state": True}}
    for simulator in (
        backend,
        SCQubitQEC17Simulator(method="DM", runtime="numpy", noise=second),
    ):
        np.testing.assert_allclose(
            simulator.run(program, **options).result().get_density_matrix(),
            np.diag([1, 0]),
        )


def _ghz_program():
    program = fq.Program(3, 3)
    program.add(ops.YHalf, 0)
    for control, target in ((0, 1), (1, 2)):
        program.add(ops.MYHalf, target)
        program.add(ops.CZ, (control, target))
        program.add(ops.YHalf, target)
    return program


def test_numpy_and_numba_agree_on_noisy_native_circuit():
    program = _ghz_program()
    np.testing.assert_allclose(
        _density(program, (6, 11, 4), runtime="numpy"),
        _density(program, (6, 11, 4), runtime="numba"),
        atol=1e-13,
    )


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
def test_noisy_ghz_trajectories_match_density_matrix_and_individual_readout(runtime):
    program = _ghz_program()
    sites = (6, 11, 4)
    populations = np.diag(_density(program, sites)).real
    readout = np.array([[1.0]])
    for f00, f11 in ((0.996, 0.983), (0.993, 0.982), (0.991, 0.978)):
        readout = np.kron(readout, [[f00, 1 - f11], [1 - f00, f11]])
    expected = readout @ populations
    program.measure_all()
    backend = SCQubitQEC17Simulator(
        runtime=runtime, noise=SCQubitQEC17Simulator.default_noise_model()
    )
    result = backend.run(
        program,
        shots=2000,
        resource_layout=_layout(program, sites),
        simulation_config={"seed": 321, "shot_parallelism": "serial"},
    ).result()
    counts = result.get_counts()
    observed = np.array([counts.get(f"{value:03b}", 0) / 2000 for value in range(8)])
    assert sum(counts.values()) == 2000
    assert result.metadata["runtime"] == runtime
    assert np.abs(observed - expected).sum() / 2 < 0.05


def test_default_idle_noise_rejects_conditions_but_ideal_execution_accepts_them():
    program = fq.Program(1, 1)
    program.add(ops.X, 0, condition=(0, 0))
    noisy = SCQubitQEC17Simulator(noise=SCQubitQEC17Simulator.default_noise_model())
    with pytest.raises(BackendValidationError, match="classical conditions"):
        noisy.run(program)
    np.testing.assert_allclose(_density(program, (0,), noisy=False), np.diag([0, 1]))
