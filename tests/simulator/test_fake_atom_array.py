"""Tests the neutral-atom backend: capacity/native-gate constraints, the
connectivity model (Pair/Unpair-gated CZ), and the Put/loss atom lifecycle.
"""

import numpy as np
import pytest

import fatqat.operations as ops
from fatqat._backends.steps import ApplyMatrixStep, LossStep, PutStep, ResetStep
from fatqat.simulator import AtomArraySimulator, Simulator
from fatqat.simulator.fake_atom_array import fake_atom_array_implementation_map
from fatqat.errors import BackendValidationError, UnsupportedOperationError
from fatqat.noise import Loss, NoiseModel, ReadoutConfusion
from fatqat.program import Program
from fatqat.registers import QuantumRegister
from fatqat.resource_layout import ResourceLayout


def _matrix_steps(plan):
    return [step for step in plan if isinstance(step, ApplyMatrixStep)]


def _one_qubit_matrix_steps(plan):
    return [s for s in _matrix_steps(plan) if len(s.target_indices) == 1]


def _two_qubit_matrix_steps(plan):
    return [s for s in _matrix_steps(plan) if len(s.target_indices) == 2]


# --- constructor validation / default shape -----------------------------------


def test_default_capacity_is_unbounded():
    # No num_sites -> no capacity cap; a large program binds without rejection.
    backend = AtomArraySimulator()
    p = Program(25)
    p.add(ops.RX(0.1), 0)
    assert isinstance(backend._resolve_resource_layout(p), ResourceLayout)


def test_explicit_none_num_sites_is_unbounded():
    backend = AtomArraySimulator(num_sites=None)
    p = Program(25)
    p.add(ops.RX(0.1), 0)
    assert isinstance(backend._resolve_resource_layout(p), ResourceLayout)


def test_explicit_capacity_is_enforced():
    backend = AtomArraySimulator(num_sites=20)
    p_fits = Program(20)  # 20 sites fit exactly
    p_fits.add(ops.RX(0.1), 0)
    assert isinstance(backend._resolve_resource_layout(p_fits), ResourceLayout)

    p_too_big = Program(21)
    p_too_big.add(ops.RX(0.1), 0)
    with pytest.raises(BackendValidationError):
        backend._resolve_resource_layout(p_too_big)


def test_rejects_non_int_num_sites():
    with pytest.raises(TypeError):
        AtomArraySimulator(num_sites="3")


def test_rejects_bool_num_sites():
    with pytest.raises(TypeError):
        AtomArraySimulator(num_sites=True)


def test_rejects_zero_num_sites():
    with pytest.raises(ValueError):
        AtomArraySimulator(num_sites=0)


def test_rejects_negative_num_sites():
    with pytest.raises(ValueError):
        AtomArraySimulator(num_sites=-1)


# --- capacity / dimension / binding -------------------------------------------


def test_rejects_over_capacity_program():
    p = Program(21)  # explicit capacity 20
    with pytest.raises(BackendValidationError):
        AtomArraySimulator(num_sites=20)._resolve_resource_layout(p)


def test_rejects_non_qubit_dimension():
    p = Program([QuantumRegister(4, dim=3)])
    with pytest.raises(BackendValidationError, match="qubit dimensions"):
        AtomArraySimulator()._resolve_resource_layout(p)


def test_binding_is_declaration_order_identity():
    p = Program(3)
    backend = AtomArraySimulator()
    ref = p.quantum_registers[0][2]
    resource_layout = backend._resolve_resource_layout(p)
    engine_index_allocation = backend._allocate_engine_indices(p, resource_layout)
    assert engine_index_allocation.engine_index(resource_layout.device_label(ref)) == 2
    assert resource_layout.device_label(ref) == 2


def test_resource_layout_covers_all_scalar_refs():
    atoms = QuantumRegister(4, name="atoms")
    p = Program([atoms])
    resource_layout = AtomArraySimulator()._resolve_resource_layout(p)
    assert {resource_layout.device_label(atoms[i]) for i in range(4)} == {0, 1, 2, 3}


# --- native gate set ----------------------------------------------------------


def test_implementation_map_exposes_four_native_families():
    m = AtomArraySimulator().implementation_map
    assert m.supports(ops.RX) and m.supports(ops.RY) and m.supports(ops.RZ)
    assert m.supports(ops.CZ)
    assert not m.supports(ops.CX)


def test_cz_is_registered_as_a_universal_rule_not_by_edge():
    # Connectivity, not a fixed edge set, gates CZ legality now, so CZ carries a
    # single class-keyed universal rule and no device-operand edges.
    m = fake_atom_array_implementation_map()
    assert m.implementation_for(ops.CZ) is not None
    assert not m.device_operands_for(ops.CZ)


def test_cx_is_unsupported_end_to_end():
    p = Program(2)
    p.add(ops.Put, (0, 1))
    p.add(ops.Pair, (0, 1))  # even paired, CX has no native implementation
    p.add(ops.CX, (0, 1))
    with pytest.raises(UnsupportedOperationError):
        AtomArraySimulator(num_sites=2).run(
            p, result_config={"counts": False, "final_state": True}
        )


# --- connectivity: Pair/Unpair gate CZ legality --------------------------------


def test_cz_on_unpaired_atoms_raises():
    # An unpaired CZ is a compile-time program error (a missing Pair), distinct
    # from a run-time atom loss: it is rejected at lowering, not dropped.
    p = Program(2)
    p.add(ops.Put, (0, 1))
    p.add(ops.CZ, (0, 1))  # never paired -> program error
    with pytest.raises(BackendValidationError, match="paired"):
        AtomArraySimulator(num_sites=2).run(
            p, result_config={"counts": False, "final_state": True}
        )


def test_cz_executes_after_pair():
    p = Program(2)
    p.add(ops.Put, (0, 1))
    p.add(ops.Pair, (0, 1))
    p.add(ops.CZ, (0, 1))
    result = AtomArraySimulator(num_sites=2).run(
        p, result_config={"counts": False, "final_state": True}
    )
    assert result.result().get_statevector().shape == (4,)


def test_cz_after_unpair_raises():
    p = Program(2)
    p.add(ops.Put, (0, 1))
    p.add(ops.Pair, (0, 1))
    p.add(ops.CZ, (0, 1))  # paired -> kept
    p.add(ops.Unpair, (0, 1))
    p.add(ops.CZ, (0, 1))  # unpaired again -> program error
    with pytest.raises(BackendValidationError, match="paired"):
        AtomArraySimulator(num_sites=2).run(
            p, result_config={"counts": False, "final_state": True}
        )


def test_v_shape_pairing_allows_both_connected_edges():
    # pair(0,1) and pair(0,2) but not (1,2): CZ(0,1) and CZ(0,2) are legal and
    # survive -- a "V", exactly representable, unlike a clique.
    p = Program(3)
    p.add(ops.Put, (0, 1, 2))
    p.add(ops.Pair, (0, 1))
    p.add(ops.Pair, (0, 2))
    p.add(ops.CZ, (0, 1))
    p.add(ops.CZ, (0, 2))
    result = AtomArraySimulator(num_sites=3).run(
        p, result_config={"counts": False, "final_state": True}
    )
    assert result.result().get_statevector().shape == (8,)


def test_v_shape_cz_on_the_missing_edge_raises():
    # The missing edge of the "V" is not paired, so CZ(1, 2) is a program error.
    p = Program(3)
    p.add(ops.Put, (0, 1, 2))
    p.add(ops.Pair, (0, 1))
    p.add(ops.Pair, (0, 2))
    p.add(ops.CZ, (1, 2))  # (1, 2) never paired -> program error
    with pytest.raises(BackendValidationError, match="paired"):
        AtomArraySimulator(num_sites=3).run(
            p, result_config={"counts": False, "final_state": True}
        )


def test_lost_atom_on_a_paired_cz_is_dropped_silently_not_raised():
    # The mirror of the unpaired-CZ error: when a CZ *is* paired, an atom lost
    # at run time (physical, per shot) silently prevents it -- the program runs
    # and reports erasure, with no error raised. Loss is not visible at lowering
    # (it is a per-shot engine effect), so the paired CZ lowers and is kept.
    noise = NoiseModel()
    noise.add(Loss(p=1.0), operation=ops.Put)  # both atoms lost right after load
    p = Program(2, 2)
    p.add(ops.Put, (0, 1))
    p.add(ops.Pair, (0, 1))
    p.add(ops.RX(np.pi), 0)
    p.add(ops.CZ, (0, 1))  # legal (paired); an atom is gone, so it is skipped
    p.measure_all()
    plan, _facts = AtomArraySimulator(num_sites=2, noise=noise)._lower_program(p)
    assert len(_two_qubit_matrix_steps(plan)) == 1  # kept at lowering
    counts = (
        AtomArraySimulator(num_sites=2, noise=noise)
        .run(p, shots=10, simulation_config={"seed": 0})
        .result()
        .get_counts()
    )
    assert counts == {"22": 10}  # both sites erased, no exception


def test_paired_cz_matches_manual_scalar_sequence():
    atom_p = Program(2)
    atom_p.add(ops.Put, (0, 1))
    atom_p.add(ops.Pair, (0, 1))
    atom_p.add(ops.RX(np.pi), 0)  # excite so CZ has a visible effect
    atom_p.add(ops.CZ, (0, 1))
    atom_sv = (
        AtomArraySimulator(num_sites=2)
        .run(atom_p, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )

    manual_p = Program(2)
    manual_p.add(ops.RX(np.pi), 0)
    manual_p.add(ops.CZ, (0, 1))
    manual_sv = (
        Simulator()
        .run(manual_p, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )
    assert np.allclose(atom_sv, manual_sv)


def test_conditional_pair_rejected():
    p = Program(2, 1)
    p.add(ops.Put, (0, 1))
    p.add(ops.Pair, (0, 1), condition=(p.classical_registers[0][0], 0))
    with pytest.raises(BackendValidationError, match="unconditional"):
        AtomArraySimulator(num_sites=2).run(
            p, result_config={"counts": False, "final_state": True}
        )


def test_conditional_unpair_rejected():
    p = Program(2, 1)
    p.add(ops.Put, (0, 1))
    p.add(ops.Unpair, (0, 1), condition=(p.classical_registers[0][0], 0))
    with pytest.raises(BackendValidationError, match="unconditional"):
        AtomArraySimulator(num_sites=2).run(
            p, result_config={"counts": False, "final_state": True}
        )


# --- Put atom lifecycle -------------------------------------------------------


def test_atom_lifecycle_translates_to_common_facts_and_occupancy():
    program = Program(1)
    program.add(ops.Put, 0)

    _plan, facts, occupied = AtomArraySimulator(num_sites=1)._prepare_program(program)

    assert facts.execution_shape == "per_shot"
    assert facts.stochastic_final_state is False
    assert facts.deferred_measurements == ()
    assert occupied == frozenset()


def test_atom_loss_translates_to_stochastic_per_shot_execution():
    noise = NoiseModel()
    noise.add(Loss(p=0.5), operation=ops.RX)
    program = Program(1)
    program.add(ops.RX(0.1), 0)

    _plan, facts, occupied = AtomArraySimulator(
        num_sites=1, noise=noise
    )._prepare_program(program)

    assert facts.execution_shape == "per_shot"
    assert facts.stochastic_final_state is True
    assert occupied == frozenset()


@pytest.mark.parametrize(
    ("step_kind", "step_type"),
    [("put", PutStep), ("loss", LossStep)],
)
def test_atom_extension_steps_preserve_conditions(step_kind, step_type):
    noise = NoiseModel()
    program = Program(1, 1)
    if step_kind == "put":
        program.add(ops.Put, 0, condition=(0, 0))
    else:
        noise.add(Loss(p=0.5), operation=ops.RX)
        program.add(ops.RX(0.1), 0, condition=(0, 0))

    backend = AtomArraySimulator(num_sites=1, noise=noise)
    plan, _facts = backend._lower_program(program)
    extension = next(step for step in plan if isinstance(step, step_type))

    assert extension.condition == ((0, 0),)
    facts, _occupied = backend._analyze_lowered_plan((extension,))
    assert facts.execution_shape == "per_shot"
    assert facts.has_condition is True


def test_atom_lifecycle_clears_deferred_measurements():
    program = Program(1, 1)
    program.add(ops.Put, 0)
    program.measure(0, 0)

    _plan, facts = AtomArraySimulator(num_sites=1)._lower_program(program)

    assert facts.execution_shape == "per_shot"
    assert facts.deferred_measurements == ()
    assert facts.written_clbits == frozenset({0})


def test_gate_before_put_executes_while_empty():
    program = Program(1, 1)
    program.add(ops.RX(np.pi), 0)  # native X-equivalent; skipped while empty
    program.add(ops.Put, 0)
    program.measure(0, 0)

    counts = (
        AtomArraySimulator(num_sites=1)
        .run(program, shots=8, simulation_config={"seed": 0})
        .result()
        .get_counts()
    )

    assert counts == {"0": 8}


def test_plain_simulator_rejects_atom_put():
    program = Program(1)
    program.add(ops.Put, 0)

    with pytest.raises(UnsupportedOperationError):
        Simulator().run(program)


def test_no_lifecycle_program_keeps_every_qubit_present():
    # A program that uses neither Put nor loss imposes no occupancy: it behaves
    # like the plain backend, every declared qubit present.
    p = Program(1, 1)
    p.add(ops.RX(np.pi), 0)
    p.measure(0, 0)
    counts = (
        AtomArraySimulator(num_sites=1)
        .run(p, shots=4, simulation_config={"seed": 0})
        .result()
        .get_counts()
    )
    assert counts == {"1": 4}


def test_no_lifecycle_has_no_occupancy_seed():
    program = Program(1)
    program.add(ops.RX(0.1), 0)

    _plan, facts, occupied = AtomArraySimulator(num_sites=1)._prepare_program(program)

    assert facts.execution_shape == "single_pass"
    assert occupied is None


def test_gate_on_never_put_site_is_dropped():
    p = Program(2)
    p.add(ops.Put, 0)  # only site 0 loaded; site 1 never Put
    p.add(ops.RX(np.pi), 0)
    p.add(ops.RX(np.pi), 1)  # site 1 can never hold an atom -> dropped
    plan, _facts = AtomArraySimulator(num_sites=2)._lower_program(p)
    assert len(_one_qubit_matrix_steps(plan)) == 1


def test_reset_on_never_put_site_is_dropped_from_plan():
    p = Program(2)
    p.add(ops.Put, 0)
    p.add(ops.Reset, 1)  # site 1 never Put -> dropped, never lowered
    plan, facts = AtomArraySimulator(num_sites=2)._lower_program(p)
    assert not facts.has_reset
    assert not any(isinstance(step, ResetStep) for step in plan)


def test_gate_on_put_site_executes_normally():
    p = Program(2)
    p.add(ops.Put, (0, 1))
    p.add(ops.RX(0.3), 0)
    p.add(ops.RX(0.3), 1)
    atom_sv = (
        AtomArraySimulator(num_sites=2)
        .run(p, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )
    manual = Program(2)
    manual.add(ops.RX(0.3), 0)
    manual.add(ops.RX(0.3), 1)
    manual_sv = (
        Simulator()
        .run(manual, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )
    assert np.allclose(atom_sv, manual_sv)


def test_measurement_of_empty_site_reads_erasure():
    # With the lifecycle active (some Put), a never-Put site is an empty trap:
    # measuring it reads the erasure digit 2, not 0.
    p = Program(2, 1)
    p.add(ops.Put, 0)  # site 1 never Put -> empty
    p.measure(1, 0)
    counts = (
        AtomArraySimulator(num_sites=2)
        .run(p, shots=4, result_config={"counts": True}, simulation_config={"seed": 0})
        .result()
        .get_counts()
    )
    assert counts == {"2": 4}


def test_empty_site_erasure_bypasses_readout_noise():
    # An empty site reports erasure regardless of a configured readout confusion
    # matrix: the erasure short-circuits classical readout.
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]])  # would flip a real 0 -> 1
    noise = NoiseModel()
    noise.add(ReadoutConfusion(matrix), targets=1)

    p = Program(2, 1)
    p.add(ops.Put, 0)  # site 1 empty
    p.measure(1, 0)
    counts = (
        AtomArraySimulator(num_sites=2, noise=noise)
        .run(p, shots=4, result_config={"counts": True}, simulation_config={"seed": 0})
        .result()
        .get_counts()
    )
    assert counts == {"2": 4}


def test_put_restores_a_lost_site():
    noise = NoiseModel()
    noise.add(Loss(p=1.0), operation=ops.RY)
    p = Program(1, 1)
    p.add(ops.Put, 0)
    p.add(ops.RY(0.0), 0)  # loss ejects the atom
    p.add(ops.Put, 0)  # reload a fresh |0>
    p.add(ops.RX(np.pi), 0)
    p.measure(0, 0)
    counts = (
        AtomArraySimulator(num_sites=1, noise=noise)
        .run(p, shots=8, simulation_config={"seed": 0})
        .result()
        .get_counts()
    )
    assert counts == {"1": 8}


def test_put_on_occupied_site_is_a_noop():
    def build(with_second_put):
        p = Program(1, 1)
        p.add(ops.Put, 0)
        p.add(ops.RX(np.pi), 0)
        if with_second_put:
            p.add(ops.Put, 0)  # already occupied -> no-op
        p.measure(0, 0)
        return p

    def counts(p):
        return (
            AtomArraySimulator(num_sites=1)
            .run(p, shots=8, simulation_config={"seed": 0})
            .result()
            .get_counts()
        )

    assert counts(build(True)) == counts(build(False)) == {"1": 8}


def test_process_workers_preserve_initial_atom_occupancy():
    pytest.importorskip("numba")
    program = Program(1, 1)
    program.add(ops.RX(np.pi), 0)
    program.add(ops.Put, 0)
    program.measure(0, 0)
    backend = AtomArraySimulator(num_sites=1, runtime="numba")

    def counts(shot_parallelism):
        return (
            backend.run(
                program,
                shots=8,
                simulation_config={
                    "seed": 7,
                    "shot_parallelism": shot_parallelism,
                    "kernel_parallelism": "serial",
                    "max_workers": 2 if shot_parallelism == "processes" else 1,
                },
            )
            .result()
            .get_counts()
        )

    # Empty occupancy skips RX, then Put loads |0>. Dropping the explicit
    # occupancy seed in a worker would apply RX first and report 1 instead.
    assert counts("processes") == counts("serial") == {"0": 8}


# --- atom loss ----------------------------------------------------------------


def test_atom_loss_ejects_the_atom():
    noise = NoiseModel()
    noise.add(Loss(p=1.0), operation=ops.RX)
    p = Program(1, 1)
    p.add(ops.Put, 0)
    p.add(ops.RX(np.pi), 0)  # applies, then the atom is lost
    p.measure(0, 0)
    counts = (
        AtomArraySimulator(num_sites=1, noise=noise)
        .run(p, shots=10, simulation_config={"seed": 0})
        .result()
        .get_counts()
    )
    assert counts == {"2": 10}


def test_atom_loss_rejected_by_a_non_atom_backend():
    noise = NoiseModel()
    noise.add(Loss(p=0.1), operation=ops.RX)
    with pytest.raises(BackendValidationError, match="Loss.*carrier loss"):
        Simulator().validate_noise_model(noise)


def test_atom_loss_accepted_by_atom_backend():
    noise = NoiseModel()
    noise.add(Loss(p=0.1), operation=ops.RX)
    assert AtomArraySimulator().validate_noise_model(noise) is None


def test_atom_loss_run_rejected_by_plain_simulator():
    noise = NoiseModel()
    noise.add(Loss(p=0.1), operation=ops.RX)
    program = Program(1, 1)
    program.add(ops.RX(np.pi), 0)
    program.measure(0, 0)
    with pytest.raises(BackendValidationError):
        Simulator(noise=noise).run(program)


# --- stochastic-state result / method constraints -----------------------------


def _atom_loss_program_without_measurement():
    program = Program(1)
    program.add(ops.Put, 0)
    program.add(ops.RX(np.pi), 0)
    return program


def _probabilistic_atom_loss_model():
    noise = NoiseModel()
    noise.add(Loss(p=0.5), operation=ops.RX)
    return noise


@pytest.mark.parametrize("method", ["statevector", "density_matrix"])
def test_atom_loss_final_state_requires_one_shot(method):
    backend = AtomArraySimulator(
        num_sites=1, method=method, noise=_probabilistic_atom_loss_model()
    )
    with pytest.raises(BackendValidationError, match="stochastic execution"):
        backend.run(
            _atom_loss_program_without_measurement(),
            shots=4,
            result_config={"counts": False, "final_state": True},
        )


def test_atom_loss_default_result_does_not_export_a_random_state():
    result = (
        AtomArraySimulator(num_sites=1, noise=_probabilistic_atom_loss_model())
        .run(_atom_loss_program_without_measurement(), shots=4)
        .result()
    )
    assert "statevector" not in result.available_data


def test_atom_loss_final_state_allows_one_shot():
    result = (
        AtomArraySimulator(num_sites=1, noise=_probabilistic_atom_loss_model())
        .run(
            _atom_loss_program_without_measurement(),
            shots=1,
            result_config={"counts": False, "final_state": True},
        )
        .result()
    )
    assert "statevector" in result.available_data


def test_put_only_final_state_remains_deterministic_for_any_shots():
    program = Program(1)
    program.add(ops.Put, 0)
    program.add(ops.RX(np.pi), 0)
    program.add(ops.Put, 0)  # no-op on the occupied site; still deterministic
    state = (
        AtomArraySimulator(num_sites=1)
        .run(program, shots=0, result_config={"counts": False, "final_state": True})
        .result()
        .get_statevector()
    )
    assert np.allclose(np.abs(state), [0, 1])


@pytest.mark.parametrize("method", ["unitary", "superop"])
def test_operator_methods_reject_atom_lifecycle(method):
    program = Program(1)
    program.add(ops.Put, 0)
    with pytest.raises(BackendValidationError, match="cannot represent atom occupancy"):
        AtomArraySimulator(num_sites=1, method=method).run(program)


# --- misc backend surface -----------------------------------------------------


def test_empty_program_lowers_to_nothing():
    p = Program(0, 0)
    plan, facts = AtomArraySimulator(num_sites=1)._lower_program(p)
    assert plan == ()
    assert not facts.has_measurement


def test_atom_array_simulator_is_public():
    import fatqat.simulator as simulator_pkg

    assert "AtomArraySimulator" in simulator_pkg.__all__
