"""Classical readout error: lowering, both execution paths, true-vs-reported."""

import numpy as np
import pytest

import fatqat as fq
from fatqat.backends import MeasurementStep, SimulatorBackend
from fatqat.errors import BackendValidationError
from fatqat.noise import NoiseModel
from fatqat.simulator.np import NumpyDMSimulator, NumpySVSimulator

_ALWAYS_ONE = np.array([[0.0, 0.0], [1.0, 1.0]])  # report 1 whatever is true
_FLIP_30 = np.array([[0.7, 0.0], [0.3, 1.0]])  # P(report 1 | true 0) = 0.3


def _readout_model(matrix, target=None):
    noise = NoiseModel()
    noise.add_readout_error(matrix, target=target)
    return noise


def _measured_program():
    program = fq.Program(1, 1)
    program.add_measurement(0, 0)
    return program


# --- lowering ---


def test_confusions_resolved_onto_measurement_step():
    backend = SimulatorBackend(noise=_readout_model(_FLIP_30))
    program = _measured_program()
    plan, facts = backend._lower_program(program)

    (measurement,) = [s for s in plan if isinstance(s, MeasurementStep)]
    assert measurement.confusions is not None
    assert np.array_equal(measurement.confusions[0], _FLIP_30)
    assert not measurement.confusions[0].flags.writeable
    # Readout error is classical: it is not channel noise and must not
    # change result defaults or stochasticity classification.
    assert facts.has_channel is False


def test_noise_free_measurement_lowers_without_confusions():
    backend = SimulatorBackend()
    program = _measured_program()
    plan, _ = backend._lower_program(program)

    (measurement,) = [s for s in plan if isinstance(s, MeasurementStep)]
    assert measurement.confusions is None


def test_untargeted_subsystems_lower_to_none_entries():
    noise = _readout_model(_FLIP_30, target=1)
    backend = SimulatorBackend(noise=noise)
    program = fq.Program(2, 2)
    program.add_measurement((0, 1), (0, 1))
    plan, _ = backend._lower_program(program)

    (measurement,) = [s for s in plan if isinstance(s, MeasurementStep)]
    assert measurement.confusions[0] is None
    assert np.array_equal(measurement.confusions[1], _FLIP_30)


def test_dimension_mismatch_rejected_at_lowering():
    backend = SimulatorBackend(noise=_readout_model(np.eye(3)))
    program = _measured_program()  # qubit measurement, 3x3 confusion
    with pytest.raises(BackendValidationError, match="dimension"):
        backend._lower_program(program)


def test_readout_error_keeps_fast_path_on_both_methods():
    backend = SimulatorBackend(noise=_readout_model(_FLIP_30))
    program = _measured_program()
    plan, _ = backend._lower_program(program)

    assert NumpySVSimulator()._analyze_plan(plan)[0] is False
    assert NumpyDMSimulator()._analyze_plan(plan)[0] is False


# --- execution: fast path ---


@pytest.mark.parametrize("method", ["SV", "DM"])
def test_fast_path_counts_reproduce_the_confusion_rate(method):
    shots = 10_000
    counts = (
        SimulatorBackend(method=method, noise=_readout_model(_FLIP_30))
        .run(_measured_program(), shots=shots, seed=2)
        .result()
        .get_counts()
    )

    # True outcome is always 0; the report flips to 1 at rate 0.3.
    assert abs(counts.get("1", 0) / shots - 0.3) < 0.02


def test_specific_target_confuses_only_its_subsystem():
    # Always-FLIP readout pinned to q1, state |10>: q1 (true 0) reports 1,
    # and q0 (true 1) must report 1 unchanged - a leak of the confusion onto
    # q0 would flip it to 0 and produce "10", so this discriminates, which
    # an always-report-one matrix would not (1 maps to 1 either way).
    always_flip = np.array([[0.0, 1.0], [1.0, 0.0]])
    program = fq.Program(2, 2)
    program.add(fq.ops.X, 0)
    program.add_measurement((0, 1), (0, 1))
    counts = (
        SimulatorBackend(noise=_readout_model(always_flip, target=1))
        .run(program, shots=200, seed=4)
        .result()
        .get_counts()
    )

    assert counts == {"11": 200}


# --- execution: dynamic path (true collapse vs corrupted report) ---


def test_feedforward_reads_the_reported_bit_not_the_true_one():
    # q0 is |0>: the true outcome is 0, but readout always reports 1, so the
    # condition c0 == 1 fires and flips q1.
    program = fq.Program(2, 2)
    program.add_measurement(0, 0)
    program.add(fq.ops.X, 1, condition=(0, 1))
    program.add_measurement(1, 1)
    counts = (
        SimulatorBackend(method="SV", noise=_readout_model(_ALWAYS_ONE, target=0))
        .run(program, shots=100, seed=3)
        .result()
        .get_counts()
    )

    assert counts == {"11": 100}


def test_collapse_and_state_export_keep_the_true_outcome():
    # Despite the always-report-1 readout, the post-measurement state of the
    # |0> qubit must still be exactly |0>: readout error is classical only.
    program = _measured_program()
    result = (
        SimulatorBackend(method="SV", noise=_readout_model(_ALWAYS_ONE))
        .run(
            program,
            shots=1,
            seed=5,
            result_config={"counts": True, "statevector": True},
        )
        .result()
    )

    assert np.allclose(result.get_statevector(), [1.0, 0.0])
    assert result.get_counts() == {"1": 1}


def test_reused_qubit_evolves_from_the_true_state():
    # Always-flip readout: the first read of |0> reports 1 (c0 = 1) but the
    # physical qubit really stays |0>, so X drives it to |1| and the second
    # read (true 1, flipped) reports 0 (c1 = 0) -> key "01". If readout
    # error wrongly corrupted the collapse itself, the second report would
    # be 1 and the key "11".
    always_flip = np.array([[0.0, 1.0], [1.0, 0.0]])
    noise = NoiseModel()
    noise.add_readout_error(always_flip, target=0)
    program = fq.Program(1, 2)
    program.add_measurement(0, 0)
    program.add(fq.ops.X, 0)
    program.add_measurement(0, 1)
    counts = (
        SimulatorBackend(method="SV", noise=noise)
        .run(program, shots=50, seed=6)
        .result()
        .get_counts()
    )

    assert counts == {"01": 50}


def test_parallel_dynamic_shots_match_serial_with_readout_error():
    noise = _readout_model(_FLIP_30)
    program = fq.Program(1, 2)
    program.add_measurement(0, 0)
    program.add(fq.ops.X, 0, condition=(0, 1))  # forces the dynamic path
    program.add_measurement(0, 1)
    serial = (
        SimulatorBackend(options={"parallel_mode": "serial"}, noise=noise)
        .run(program, shots=8, seed=13)
        .result()
        .get_counts()
    )
    parallel = (
        SimulatorBackend(options={"max_workers": 2}, noise=noise)
        .run(program, shots=8, seed=13)
        .result()
        .get_counts()
    )

    assert parallel == serial


def test_validate_noise_reports_readout_error_as_accepted():
    report = SimulatorBackend().validate_noise(_readout_model(_FLIP_30))

    assert report.supported is True
    assert "readout_error" in report.accepted_sources
