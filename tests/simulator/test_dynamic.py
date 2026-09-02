import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat._backends.steps import (
    ApplyMatrixStep,
    MeasurementStep,
    ResetStep,
)
from fatqat.simulator import Simulator
from fatqat.program import Program


def test_lower_terminal_measurement_is_not_dynamic():
    p = Program(2, 2)
    p.add(ops.H, 0)
    p.add(ops.CZ, (0, 1))
    p.measure(0, 0)
    p.measure(1, 1)
    _plan, facts = Simulator("SV")._lower_program(p)
    assert facts.execution_shape == "single_pass"
    assert facts.deferred_measurements == ((1, 0), (0, 1))
    assert facts.written_clbits == frozenset({0, 1})
    assert facts.stochastic_final_state is True
    assert facts.has_measurement is True
    assert facts.has_reset is False


def test_lower_measure_then_gate_on_disjoint_qubit_is_not_dynamic():
    p = Program(2, 2)
    p.add(ops.H, 0)
    p.measure(0, 0)
    p.add(ops.X, 1)  # different qubit -> still fast path
    p.measure(1, 1)
    _plan, facts = Simulator("SV")._lower_program(p)
    assert facts.execution_shape == "single_pass"
    assert facts.deferred_measurements == ((1, 0), (0, 1))


def test_lower_gate_on_measured_qubit_is_dynamic():
    p = Program(1, 1)
    p.add(ops.H, 0)
    p.measure(0, 0)
    p.add(ops.X, 0)  # gate on already-measured qubit
    _plan, facts = Simulator("SV")._lower_program(p)
    assert facts.execution_shape == "per_shot"
    assert facts.deferred_measurements == ()


def test_repeated_measurement_of_one_subsystem_is_per_shot():
    program = Program(1, 2)
    program.measure(0, 0)
    program.measure(0, 1)

    _plan, facts = Simulator("SV")._lower_program(program)

    assert facts.execution_shape == "per_shot"
    assert facts.deferred_measurements == ()
    assert facts.written_clbits == frozenset({0, 1})


def test_lower_condition_is_dynamic_and_resolves_indices():
    p = Program(2, 2)
    p.add(ops.X, 1, condition=(0, 1))
    plan, facts = Simulator("SV")._lower_program(p)
    assert facts.execution_shape == "per_shot"
    assert facts.stochastic_final_state is False
    assert facts.has_condition is True
    gate = plan[0]
    assert isinstance(gate, ApplyMatrixStep)
    assert gate.condition == ((0, 1),)


@pytest.mark.parametrize(
    "step_kind, method, expected_shape, expected_stochastic",
    [
        ("reset", "statevector", "per_shot", True),
        ("reset", "density_matrix", "single_pass", False),
        ("reset", "unitary", "operator", True),
        ("reset", "superop", "operator", False),
        ("channel", "statevector", "per_shot", True),
        ("channel", "density_matrix", "single_pass", False),
    ],
)
def test_nonunitary_semantics_are_method_owned(
    step_kind, method, expected_shape, expected_stochastic
):
    program = Program(1)
    noise = None
    if step_kind == "reset":
        program.add(ops.Reset, 0)
    else:
        program.add(ops.X, 0)
        noise = fq.NoiseModel()
        noise.add(fq.noise.Depolarizing(p=0.1), operation=ops.X)

    _plan, facts = Simulator(method, runtime="numpy", noise=noise)._lower_program(
        program
    )

    assert facts.execution_shape == expected_shape
    assert facts.stochastic_final_state is expected_stochastic
    assert facts.has_reset is (step_kind == "reset")
    assert facts.has_channel is (step_kind == "channel")


def test_per_shot_trigger_does_not_stop_later_fact_collection():
    program = Program(2, 2)
    program.add(ops.X, 1, condition=(0, 0))
    program.measure(0, 1)

    _plan, facts = Simulator("SV")._lower_program(program)

    assert facts.execution_shape == "per_shot"
    assert facts.deferred_measurements == ()
    assert facts.written_clbits == frozenset({1})
    assert facts.has_condition is True
    assert facts.has_measurement is True


def test_common_analysis_rejects_an_unclaimed_resolved_step():
    class UnknownStep:
        condition = None

    with pytest.raises(TypeError, match="UnknownStep"):
        Simulator("SV")._analyze_lowered_plan((UnknownStep(),))


def test_plan_semantics_do_not_depend_on_numeric_runtime():
    pytest.importorskip("numba")
    program = Program(1, 1)
    program.measure(0, 0)
    program.add(ops.Reset, 0)

    facts = {
        runtime: Simulator("SV", runtime=runtime)._lower_program(program)[1]
        for runtime in ("numpy", "numba")
    }

    assert facts["numba"] == facts["numpy"]


def test_lower_unknown_gate_raises():
    class FooGate(ops.Operation):
        name = "FOO"
        num_subsystems = 1

    p = Program(1)
    p.add(FooGate(), 0)
    with pytest.raises(fq.errors.UnsupportedOperationError):
        Simulator("SV")._lower_program(p)


def test_reset_and_reuse_counts():
    # Put q0 in |1>, measure -> c0=1, reset q0, measure again -> c1=0.
    p = Program(1, 2)
    p.add(ops.X, 0)
    p.measure(0, 0)
    p.add(ops.Reset, 0)
    p.measure(0, 1)
    counts = (
        Simulator("SV")
        .run(p, shots=32, simulation_config={"seed": 0})
        .result()
        .get_counts()
    )
    assert counts == {"10": 32}  # c0=1 then c1=0 in public slot order


def test_dynamic_counts_use_snapshots_not_final_index():
    # After reset the final basis state has q0=|0>, but c0 recorded the pre-reset 1.
    # A from-final-index builder would wrongly read c0=0.
    p = Program(1, 1)
    p.add(ops.X, 0)
    p.measure(0, 0)
    p.add(ops.Reset, 0)
    counts = (
        Simulator("SV")
        .run(p, shots=10, simulation_config={"seed": 0})
        .result()
        .get_counts()
    )
    assert counts == {"1": 10}


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
@pytest.mark.parametrize("method", ["SV", "DM"])
def test_measurement_feedforward_and_reset_keep_public_targets(runtime, method):
    if runtime == "numba":
        pytest.importorskip("numba")
    program = Program(2, 2)
    program.add(ops.X, 0)
    program.measure(0, 0)
    program.add(ops.X, 1, condition=(0, 1))
    program.add(ops.Reset, 0)
    program.measure(1, 1)

    backend = Simulator(method, runtime=runtime)
    counts = backend.run(program, shots=8).result().get_counts_as_tuples()
    assert counts == {(1, 1): 8}

    final = backend.run(
        program,
        shots=1,
        result_config={"counts": False, "final_state": True},
    ).result()
    if method == "SV":
        expected = np.zeros(4, dtype=complex)
        expected[1] = 1.0
        assert np.allclose(final.get_statevector(), expected)
    else:
        expected = np.zeros((4, 4), dtype=complex)
        expected[1, 1] = 1.0
        assert np.allclose(final.get_density_matrix(), expected)


def test_condition_only_statevector_default_at_many_shots():
    # Dynamic (condition) but no measurement/reset -> statevector available/default.
    p = Program(2, 2)
    p.add(ops.X, 1, condition=(0, 0))  # applies (slot 0 == 0)
    sv = Simulator("SV").run(p, shots=8).result().get_statevector()
    expected = np.zeros(4, dtype=complex)
    expected[0b01] = 1.0  # public qubit 1 -> |01>
    assert np.allclose(sv, expected)


def test_statevector_with_reset_and_many_shots_rejected():
    p = Program(1)
    p.add(ops.Reset, 0)
    with pytest.raises(fq.errors.BackendValidationError):
        Simulator("SV").run(p, shots=10, result_config={"final_state": True})


def test_conditional_reset_fires_when_guard_true():
    # q0=|1>, measure -> c0=1; reset conditioned on c0==1 fires; second read is 0.
    p = Program(1, 2)
    p.add(ops.X, 0)
    p.measure(0, 0)
    p.add(ops.Reset, 0, condition=(0, 1))
    p.measure(0, 1)
    counts = (
        Simulator("SV")
        .run(p, shots=16, simulation_config={"seed": 0})
        .result()
        .get_counts()
    )
    assert counts == {"10": 16}  # c0=1, c1=0


def test_conditional_reset_skipped_when_guard_false():
    # Same shape, guard c0==0 is false, so reset is SKIPPED and the second read
    # stays 1. This is the case a dropped reset-condition would silently break.
    p = Program(1, 2)
    p.add(ops.X, 0)
    p.measure(0, 0)
    p.add(ops.Reset, 0, condition=(0, 0))
    p.measure(0, 1)
    counts = (
        Simulator("SV")
        .run(p, shots=16, simulation_config={"seed": 0})
        .result()
        .get_counts()
    )
    assert counts == {"11": 16}  # c1=1, c0=1 -> reset did not fire


def test_condition_only_statevector_ignores_shots_value():
    # Non-stochastic dynamic program: the statevector must be produced regardless
    # of `shots` (including shots == 0), because it does not depend on sampling.
    p = Program(2, 2)
    p.add(ops.X, 1, condition=(0, 0))
    sv = (
        Simulator("SV")
        .run(p, shots=0, result_config={"final_state": True})
        .result()
        .get_statevector()
    )
    expected = np.zeros(4, dtype=complex)
    expected[0b01] = 1.0
    assert np.allclose(sv, expected)


def test_lower_grouped_measurement_emits_one_grouped_step():
    p = Program(3, 3)
    p.measure((0, 2), (1, 0))

    plan, facts = Simulator("SV")._lower_program(p)

    assert facts.has_measurement is True
    assert facts.execution_shape == "single_pass"
    assert plan == (MeasurementStep(measured_indices=(2, 0), classical_indices=(1, 0)),)


def test_lower_adjacent_single_measurements_stay_separate_steps():
    p = Program(2, 2)
    p.measure(0, 0)
    p.measure(1, 1)

    plan, facts = Simulator("SV")._lower_program(p)

    assert facts.execution_shape == "single_pass"
    assert plan == (
        MeasurementStep(measured_indices=(1,), classical_indices=(0,)),
        MeasurementStep(measured_indices=(0,), classical_indices=(1,)),
    )


def test_lower_grouped_reset_uses_all_targets():
    p = Program(3)
    p.add(ops.Reset, (0, 2))

    plan, facts = Simulator("SV")._lower_program(p)

    assert facts.execution_shape == "per_shot"
    assert facts.has_reset is True
    assert plan == (ResetStep(reset_indices=(2, 0)),)


def test_grouped_reset_resets_all_targets_in_dynamic_path():
    p = Program(2, 2)
    p.add(ops.X, 0)
    p.add(ops.X, 1)
    p.add(ops.Reset, (0, 1))
    p.measure((0, 1), (0, 1))

    counts = (
        Simulator("SV")
        .run(p, shots=8, simulation_config={"seed": 0})
        .result()
        .get_counts()
    )

    assert counts == {"00": 8}


def test_grouped_measurement_writes_all_classical_slots_in_dynamic_path():
    p = Program(2, 2)
    p.add(ops.X, 0)
    p.add(ops.X, 1)
    p.measure((0, 1), (1, 0))
    p.add(ops.X, 0)  # makes the program dynamic because q0 was measured

    counts = (
        Simulator("SV")
        .run(p, shots=8, simulation_config={"seed": 0})
        .result()
        .get_counts()
    )

    assert counts == {"11": 8}


def _random_dynamic_program():
    p = Program(1, 1)
    p.add(ops.H, 0)
    p.measure(0, 0)
    p.add(ops.Reset, 0)
    return p


def test_dynamic_seed_is_repeatable_with_per_shot_streams():
    p = _random_dynamic_program()
    backend = Simulator("SV")

    a = (
        backend.run(p, shots=64, simulation_config={"seed": 2026, "max_workers": 1})
        .result()
        .get_counts()
    )
    b = (
        backend.run(p, shots=64, simulation_config={"seed": 2026, "max_workers": 1})
        .result()
        .get_counts()
    )

    assert a == b


def test_dynamic_statevector_single_shot_stays_serial_and_available():
    p = Program(1, 1)
    p.add(ops.H, 0)
    p.measure(0, 0)
    p.add(ops.Reset, 0)
    result = (
        Simulator("SV")
        .run(
            p,
            shots=1,
            simulation_config={"seed": 2026, "max_workers": 4},
            result_config={"counts": True, "final_state": True},
        )
        .result()
    )

    assert sum(result.get_counts().values()) == 1
    assert result.get_statevector().shape == (2,)
