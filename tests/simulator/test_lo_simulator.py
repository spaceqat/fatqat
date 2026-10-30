"""Tests optical gates on LOSimulator's per-run truncated modes."""

import math

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.errors import BackendValidationError
from fatqat.noise import Depolarizing
from fatqat.simulator import BasisState, EvolutionState, LOSimulator, Simulator

_STATE_ONLY = {"shots": 0, "result_config": {"counts": False, "final_state": True}}


def _two_mode_program(*operations):
    modes = fq.ModeRegister(2)
    program = fq.Program([modes])
    for operation in operations:
        targets = (modes[0], modes[1]) if operation.num_subsystems == 2 else modes[0]
        program.add(operation, targets)
    return program


def _unitary(program, cutoff):
    return (
        LOSimulator("unitary")
        .run(program, simulation_config={"cutoff": cutoff})
        .result()
        .get_unitary()
    )


def _fock(cutoff, *occupations):
    state = np.zeros((cutoff + 1) ** len(occupations), dtype=complex)
    state[np.ravel_multi_index(occupations, (cutoff + 1,) * len(occupations))] = 1.0
    return state


@pytest.mark.parametrize("cutoff", [1, 2, 3])
def test_optical_gates_are_unitary_and_conserve_photon_number(cutoff):
    program = _two_mode_program(ops.BeamSplitter(0.37, 1.1), ops.PhaseShifter(0.6))
    unitary = _unitary(program, cutoff)

    dim = cutoff + 1
    np.testing.assert_allclose(
        unitary.conj().T @ unitary, np.eye(dim**2), rtol=0, atol=1e-12
    )
    photons = np.add.outer(np.arange(dim), np.arange(dim)).ravel()
    changes_number = photons[:, None] != photons[None, :]
    np.testing.assert_allclose(unitary[changes_number], 0.0, rtol=0, atol=1e-12)


def test_phase_shifter_applies_number_dependent_phase():
    unitary = _unitary(_two_mode_program(ops.PhaseShifter(0.3)), cutoff=3)

    expected = np.kron(np.diag(np.exp(0.3j * np.arange(4))), np.eye(4))
    np.testing.assert_allclose(unitary, expected, rtol=0, atol=1e-12)


def _one_photon_block(unitary):
    # Rows and columns in the order (|1, 0>, |0, 1>) at cutoff 1.
    return unitary[np.ix_([2, 1], [2, 1])]


@pytest.mark.parametrize(
    ("splitter", "qubit_gate"),
    [
        (ops.BeamSplitter(0.3, -math.pi / 2), ops.RX(0.6)),
        (ops.BeamSplitter(0.3), ops.RY(0.6)),
    ],
    ids=["RX", "RY"],
)
def test_beam_splitter_matches_qubit_rotation_on_dual_rail(splitter, qubit_gate):
    qubit = fq.Program(1)
    qubit.add(qubit_gate, 0)
    expected = Simulator("unitary").run(qubit).result().get_unitary()

    block = _one_photon_block(_unitary(_two_mode_program(splitter), cutoff=1))

    np.testing.assert_allclose(block, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "method", ["statevector", "density_matrix", "fixed_photon_state"]
)
def test_hong_ou_mandel_photons_leave_together(method):
    modes = fq.ModeRegister(2)
    counts = fq.CountRegister(2)
    program = fq.Program([modes], [counts])
    program.add(ops.BeamSplitter(math.pi / 4), (modes[0], modes[1]))
    program.measure((modes[0], modes[1]), (counts[0], counts[1]))

    # Two photons infer cutoff 2, enough for both photons in one mode.
    result = (
        LOSimulator(method)
        .run(
            program,
            shots=2000,
            initial_state=BasisState({modes[0]: 1, modes[1]: 1}),
            simulation_config={"seed": 7},
        )
        .result()
    )

    observed = result.get_counts()
    assert set(observed) == {"20", "02"}
    assert observed["20"] == pytest.approx(1000, abs=100)


def test_cutoff_inference_accounts_for_fixed_dimension_registers():
    qubit = fq.QuantumRegister(1)
    modes = fq.ModeRegister(2)
    program = fq.Program([qubit, modes])
    program.add(ops.BeamSplitter(math.pi / 2), (modes[0], modes[1]))
    initial = np.kron([1.0, 0.0], _fock(2, 1, 0))

    state = LOSimulator().run(program, initial_state=initial, **_STATE_ONLY)

    expected = np.kron([1.0, 0.0], _fock(2, 0, 1))
    np.testing.assert_allclose(
        state.result().get_statevector(), expected, rtol=0, atol=1e-12
    )


def test_basis_state_cutoff_counts_only_mode_photons():
    qubit = fq.QuantumRegister(1)
    modes = fq.ModeRegister(2)
    program = fq.Program([qubit, modes])
    program.add(ops.BeamSplitter(math.pi / 2), (modes[0], modes[1]))
    # The qubit's level is not a photon, so the inferred cutoff is 1.
    initial = EvolutionState(quantum=BasisState({qubit[0]: 1, modes[0]: 1}))

    state = LOSimulator().run(program, initial_state=initial, **_STATE_ONLY)

    expected = np.kron([0.0, 1.0], _fock(1, 0, 1))
    np.testing.assert_allclose(
        state.result().get_statevector(), expected, rtol=0, atol=1e-12
    )


def test_explicit_cutoff_below_the_photon_total_warns():
    program = _two_mode_program(ops.BeamSplitter(math.pi / 4))
    modes = program.quantum_registers[0]
    basis = BasisState({modes[0]: 1, modes[1]: 1})
    options = {"initial_state": basis, **_STATE_ONLY}

    with pytest.warns(UserWarning, match="below the 2 photons"):
        LOSimulator().run(program, simulation_config={"cutoff": 1}, **options)
    # A mode level above the cutoff cannot be represented at all.
    with (
        pytest.warns(UserWarning),
        pytest.raises(BackendValidationError, match="outside its dimension 2"),
    ):
        LOSimulator().run(
            program,
            initial_state=BasisState({modes[0]: 2}),
            simulation_config={"cutoff": 1},
            **_STATE_ONLY,
        )


def test_cutoff_must_be_given_without_an_initial_state():
    program = _two_mode_program(ops.PhaseShifter(0.1))
    with pytest.raises(BackendValidationError, match="cutoff"):
        LOSimulator().run(program, **_STATE_ONLY)


def test_cutoff_cannot_be_inferred_from_a_mismatched_length():
    program = _two_mode_program(ops.PhaseShifter(0.1))
    with pytest.raises(BackendValidationError, match="cannot infer cutoff"):
        LOSimulator().run(program, initial_state=np.ones(8), **_STATE_ONLY)


@pytest.mark.parametrize("cutoff", [0, True, 1.0])
def test_invalid_cutoff_is_rejected(cutoff):
    program = _two_mode_program(ops.PhaseShifter(0.1))
    with pytest.raises(BackendValidationError, match="cutoff"):
        LOSimulator().run(program, simulation_config={"cutoff": cutoff}, **_STATE_ONLY)


def test_supplied_resource_layout_names_mode_refs():
    program = _two_mode_program(ops.BeamSplitter(math.pi / 4))
    modes = program.quantum_registers[0]
    layout = fq.ResourceLayout({modes[0]: "left", modes[1]: "right"})
    options = {"initial_state": _fock(1, 1, 0), **_STATE_ONLY}

    placed = LOSimulator().run(program, resource_layout=layout, **options).result()

    default = LOSimulator().run(program, **options).result()
    np.testing.assert_allclose(
        placed.get_statevector(), default.get_statevector(), rtol=0, atol=1e-12
    )
    # Results name the caller's mode refs, not the per-run sized copies.
    for result in (placed, default):
        axes = result.metadata["state_axes"]
        assert [axis["register_ref"] for axis in axes] == [modes[0], modes[1]]
    assert [axis["device_operand"] for axis in placed.metadata["state_axes"]] == [
        "left",
        "right",
    ]


def test_level_checks_on_modes_run_once_the_cutoff_is_known():
    modes = fq.ModeRegister(1)
    program = fq.Program([modes])
    program.add(ops.SwapLevels(0, 3), modes[0])  # no dimension to check yet

    with pytest.raises(ValueError, match="invalid for target dimension 3"):
        LOSimulator().run(program, simulation_config={"cutoff": 2}, **_STATE_ONLY)
    LOSimulator().run(program, simulation_config={"cutoff": 3}, **_STATE_ONLY)


def test_generic_simulator_rejects_dimensionless_registers():
    program = _two_mode_program(ops.PhaseShifter(0.1))
    with pytest.raises(BackendValidationError, match="LOSimulator"):
        Simulator().run(program, **_STATE_ONLY)


def test_beam_splitter_rejects_unequal_fixed_dimensions():
    program = fq.Program([fq.QuantumRegister(1), fq.QuantumRegister(1, dim=3)])
    qubit, qutrit = program.quantum_registers
    with pytest.raises(ValueError, match="equal target dimensions"):
        program.add(ops.BeamSplitter(0.1), (qubit[0], qutrit[0]))


def test_run_sweep_matches_explicit_loop():
    phi = fq.Parameter("phi")
    program = _two_mode_program(
        ops.BeamSplitter(math.pi / 4),
        ops.PhaseShifter(phi),
        ops.BeamSplitter(math.pi / 4),
    )
    values = np.linspace(0.0, math.pi, 4)
    backend = LOSimulator()
    modes = program.quantum_registers[0]
    options = {"initial_state": BasisState({modes[0]: 1}), **_STATE_ONLY}

    swept = backend.run_sweep(program, {phi: values}, **options).result()

    for result, value in zip(swept, values, strict=True):
        bound = program.assign_parameters({phi: value})
        expected = backend.run(bound, **options).result().get_statevector()
        np.testing.assert_allclose(
            result.get_statevector(), expected, rtol=0, atol=1e-12
        )
    # A Mach-Zehnder interferometer sends the photon to mode 1 at phi = 0.
    assert abs(swept[0].get_statevector()[1]) == pytest.approx(1.0)


def test_noise_selects_modes_by_device_label():
    program = _two_mode_program(ops.BeamSplitter(math.pi / 4))
    noise = fq.NoiseModel()
    noise.add(Depolarizing(p=1.0), operation=ops.BeamSplitter, targets=(0, 1))

    result = (
        LOSimulator("DM", noise=noise)
        .run(program, initial_state=_fock(1, 1, 0), **_STATE_ONLY)
        .result()
    )

    np.testing.assert_allclose(
        result.get_density_matrix(), np.eye(4) / 4, rtol=0, atol=1e-12
    )

    modes = program.quantum_registers[0]
    by_ref = fq.NoiseModel()
    by_ref.add(
        Depolarizing(p=0.1), operation=ops.BeamSplitter, targets=(modes[0], modes[1])
    )
    with pytest.raises(BackendValidationError, match="device label"):
        LOSimulator("DM", noise=by_ref).run(
            program, initial_state=_fock(1, 1, 0), **_STATE_ONLY
        )
