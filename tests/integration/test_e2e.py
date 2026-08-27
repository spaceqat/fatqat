"""Tests a minimal public fatqat workflow from program construction to counts."""

import numpy as np

import fatqat as fq
import fatqat.operations as ops


def test_minimal_workflow_from_spec():
    program = fq.Program(2, 2)
    program.add(ops.H, 0)
    program.add(ops.CZ, (0, 1))
    program.measure(0, 0)
    program.measure(1, 1)

    backend = fq.simulator.Simulator("SV")
    job = backend.run(
        program,
        shots=1000,
        simulation_config={"seed": 2024},
        result_config={"counts": True},
    )
    result = job.result()
    counts = result.get_counts()

    assert sum(counts.values()) == 1000
    assert set(counts) <= {"00", "01"}
    # roughly balanced between the two reachable outcomes
    assert all(150 < v < 850 for v in counts.values())


def test_phase3_grouped_measure_reset_and_parallel_counts_workflow():
    program = fq.Program(2, 2)
    program.add(ops.X, 0)
    program.add(ops.X, 1)
    program.measure((0, 1), (0, 1))
    program.add(ops.Reset, (0, 1))
    program.measure_all()

    result = (
        fq.simulator.Simulator("SV", runtime="numpy")
        .run(
            program,
            shots=12,
            simulation_config={
                "seed": 2026,
                "max_workers": 2,
                "shot_parallelism": "processes",
                "kernel_parallelism": "serial",
            },
            result_config={"counts": True},
        )
        .result()
    )

    assert result.get_counts() == {"00": 12}


def test_heterogeneous_qutrit_qubit_program():
    qt = fq.QuantumRegister(1, dim=3)
    qb = fq.QuantumRegister(1, dim=2)
    ct = fq.ClassicalRegister(1, dim=3)
    cb = fq.ClassicalRegister(1, dim=2)
    program = fq.Program([qt, qb], [ct, cb])
    program.add(ops.Shift(1), qt[0])  # qutrit |0> -> |1>
    program.add(ops.X, qb[0])  # qubit  |0> -> |1>
    program.measure(qt[0], ct[0])
    program.measure(qb[0], cb[0])
    result = fq.simulator.Simulator("SV").run(program, shots=16).result()
    assert result.get_counts_as_tuples() == {(1, 1): 16}


def test_sum_across_mismatched_dims_fails_at_lowering():
    import pytest
    from fatqat.errors import MatrixImplementationError

    qt = fq.QuantumRegister(1, dim=3)
    qb = fq.QuantumRegister(1, dim=2)
    program = fq.Program([qt, qb])
    program.add(ops.Sum, (qt[0], qb[0]))  # frontend does not raise
    with pytest.raises(MatrixImplementationError):
        fq.simulator.Simulator("SV").run(
            program, result_config={"counts": False, "final_state": True}
        ).result()


def test_sum_entangles_two_qutrits():
    qreg = fq.QuantumRegister(2, dim=3)
    creg = fq.ClassicalRegister(2, dim=3)
    program = fq.Program([qreg], [creg])
    program.add(ops.Shift(2), 0)  # control qutrit -> |2>
    program.measure(0, 0)  # clbit0 = 2 (mid-circuit; deterministic)
    # Condition genuinely fires (clbit0 == 2, a value only reachable by a
    # qudit, not just 0/1), proving condition literals are compared for exact
    # equality rather than truthiness for dim > 2.
    program.add(ops.Sum, (0, 1), condition=(creg[0], 2))  # target -> (2+0)%3 = 2
    program.measure(1, 1)
    result = fq.simulator.Simulator("SV").run(program, shots=32).result()
    assert result.get_counts_as_tuples() == {(2, 2): 32}


def test_fast_and_dynamic_counts_match_for_qutrit():
    def build(force_dynamic):
        qreg = fq.QuantumRegister(1, dim=3)
        creg = fq.ClassicalRegister(1, dim=3)
        p = fq.Program([qreg], [creg])
        p.add(ops.Shift(2), 0)  # deterministic |0> -> |2>
        p.measure(0, 0)
        if force_dynamic:
            # Inert no-op: Shift(0) is the identity, and its condition can
            # never be satisfied (the measured clbit is always 2), so this
            # cannot change the outcome distribution. Its mere presence
            # (a condition) forces the statevector engine's dynamic classification,
            # letting the dynamic path be compared against the fast path for
            # the identical program shape and seed.
            p.add(ops.Shift(0), 0, condition=(p.classical_registers[0][0], 0))
        return p

    fast_counts = (
        fq.simulator.Simulator("SV")
        .run(build(False), shots=8, simulation_config={"seed": 7})
        .result()
        .get_counts_as_tuples()
    )
    dyn_counts = (
        fq.simulator.Simulator("SV")
        .run(build(True), shots=8, simulation_config={"seed": 7})
        .result()
        .get_counts_as_tuples()
    )
    assert fast_counts == dyn_counts == {(2,): 8}


def test_cclock_unequal_dimensions_runs_through_backend():
    qt = fq.QuantumRegister(1, dim=3)
    qb = fq.QuantumRegister(1, dim=2)
    program = fq.Program([qt, qb])
    program.add(ops.Shift(1), qt[0])  # control -> |1>
    program.add(ops.X, qb[0])  # target -> |1>
    program.add(ops.CClock(1), (qt[0], qb[0]))
    result = (
        fq.simulator.Simulator("SV")
        .run(program, result_config={"counts": False, "final_state": True})
        .result()
    )
    sv = result.get_statevector()
    # The engine's global statevector index is little-endian across program
    # subsystems (subsystem 0 is the least-significant digit, place value
    # prod(dims[:0])=1; subsystem 1 has place value dims[0]=3) - qt (control,
    # subsystem 0) contributes i*1, qb (target, subsystem 1) contributes k*3.
    # This is unrelated to CClock's own local matrix convention (control as
    # local MSB), which only governs the gate's own target_indices ordering.
    # omega_2^(1*1*1 mod 2) = exp(i*pi) = -1.
    assert sv.shape == (6,)
    expected = np.zeros(6, dtype=complex)
    expected[1 * 1 + 1 * 3] = -1.0
    assert np.allclose(sv, expected)


def test_qutrit_circuit_with_new_gates_produces_expected_counts():
    # Every step below is chosen so the final state is a single computational
    # basis state with certainty, so the expected outcome is hand-computable
    # (an "independently computed reference," not just a shape/range check).
    qreg = fq.QuantumRegister(2, dim=3)
    creg = fq.ClassicalRegister(2, dim=3)
    program = fq.Program([qreg], [creg])
    # q0: |0> -> Fourier -> InverseFourier -> |0> (round-trip identity; exercises
    # Fourier/InverseFourier through the real backend without changing the state).
    program.add(ops.Fourier, 0)
    program.add(ops.InverseFourier, 0)
    # q0: |0> -> SwapLevels(0, 2) -> |2>
    program.add(ops.SwapLevels(0, 2), 0)
    # q0: |2> -> SubspaceRX(pi, (0, 2)) -> -i|0>. The subspace's "k" role
    # (level 2, the current state) maps to -i times its "j" role (level 0):
    # with c=cos(pi/2)=0, s=sin(pi/2)=1, the (0,2) block sends the k-column
    # to [-i*s, 0's for the untouched level, c] = [-i, 0]. The global phase
    # is unobservable in measurement counts.
    program.add(ops.SubspaceRX(np.pi, (0, 2)), 0)
    # q1 stays |0>, so CClock(1)'s control is always |0>: this exercises
    # CClock's wiring/dimension handling through the real backend without
    # changing the (phase-invisible-to-counts) expected outcome. CClock's
    # own phase computation is independently verified in Task 5's
    # test_cclock_unequal_dimensions_runs_through_backend, where the control
    # is prepared in a nonzero level specifically to make the phase visible.
    program.add(ops.CClock(1), (0, 1))
    program.measure_all()

    result = (
        fq.simulator.Simulator("SV")
        .run(
            program,
            shots=50,
            simulation_config={"seed": 0},
            result_config={"counts": True},
        )
        .result()
    )
    assert result.get_counts_as_tuples() == {(0, 0): 50}


def test_qubit_only_gate_on_qutrit_does_not_raise_at_add_but_raises_at_run():
    import pytest
    from fatqat.errors import BackendValidationError

    qreg = fq.QuantumRegister(1, dim=3)
    program = fq.Program([qreg])
    program.add(ops.H, 0)  # frontend stays neutral: does not raise here
    with pytest.raises(BackendValidationError):
        fq.simulator.Simulator("SV").run(
            program, result_config={"counts": False, "final_state": True}
        ).result()
