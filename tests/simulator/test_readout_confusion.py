"""Classical readout confusion: lowering and true-vs-reported behavior.

Every bare-int ``target=`` below is a physical device-resource label (see
``NoiseModel.add(ReadoutConfusion(...), targets=...)``), never an engine index.
It is only numerically equal to the measured subsystem's engine
index because `Simulator`'s default `_resolve_resource_layout` policy
happens to assign device labels in declaration order, coinciding with
`_EngineAllocation`'s flat indices for this generic backend. That coincidence
is backend-specific, not part of the selector's meaning; see
`tests/simulator/test_fake_atom_array.py`'s readout-selector tests for a
non-trivial layout where a device label and its engine index diverge.
"""

import numpy as np
import pytest

import fatqat as fq
from fatqat._backends.steps import MeasurementStep
from fatqat.simulator import Simulator
from fatqat.errors import BackendValidationError
from fatqat.noise import NoiseModel, ReadoutConfusion

_ALWAYS_ONE = np.array([[0.0, 0.0], [1.0, 1.0]])  # report 1 whatever is true
_FLIP_30 = np.array([[0.7, 0.0], [0.3, 1.0]])  # P(report 1 | true 0) = 0.3


def _readout_model(matrix, target=None):
    noise = NoiseModel()
    noise.add(ReadoutConfusion(matrix), targets=target)
    return noise


def _measured_program():
    program = fq.Program(1, 1)
    program.measure(0, 0)
    return program


# --- lowering ---


def test_confusions_resolved_onto_measurement_step():
    backend = Simulator(noise=_readout_model(_FLIP_30))
    program = _measured_program()
    plan, facts = backend._lower_program(program)

    (measurement,) = [s for s in plan if isinstance(s, MeasurementStep)]
    assert measurement.confusions is not None
    assert np.array_equal(measurement.confusions[0], _FLIP_30)
    assert not measurement.confusions[0].flags.writeable
    # Readout confusion is classical: it is not channel noise and must not
    # change result defaults or stochasticity classification.
    assert facts.has_channel is False


def test_noise_free_measurement_lowers_without_confusions():
    backend = Simulator()
    program = _measured_program()
    plan, _ = backend._lower_program(program)

    (measurement,) = [s for s in plan if isinstance(s, MeasurementStep)]
    assert measurement.confusions is None
    # The shared measurement-lowering boundary (backend_utils) never decides
    # what a caller stores as `reported_digit_maps`; matrix's identity,
    # noise-free case must still pass `None` explicitly, since that is the
    # compatibility default a numba-compiled fast path recognizes (contrast
    # the pulse family, which always stores its literal qutrit-to-bit map).
    assert measurement.reported_digit_maps is None


def test_untargeted_subsystems_lower_to_none_entries():
    noise = _readout_model(_FLIP_30, target=1)
    backend = Simulator(noise=noise)
    program = fq.Program(2, 2)
    program.measure((0, 1), (0, 1))
    plan, _ = backend._lower_program(program)

    (measurement,) = [s for s in plan if isinstance(s, MeasurementStep)]
    assert measurement.confusions[0] is None
    assert np.array_equal(measurement.confusions[1], _FLIP_30)


def test_dimension_mismatch_rejected_at_lowering():
    backend = Simulator(noise=_readout_model(np.eye(3)))
    program = _measured_program()  # qubit measurement, 3x3 confusion
    with pytest.raises(BackendValidationError, match="dimension"):
        backend._lower_program(program)


def test_readout_confusion_keeps_single_pass_shape():
    program = _measured_program()
    _plan, facts = Simulator(
        "statevector", noise=_readout_model(_FLIP_30)
    )._lower_program(program)

    assert facts.execution_shape == "single_pass"


# --- execution: fast path ---


@pytest.mark.parametrize("method", ["SV", "DM"])
def test_fast_path_counts_reproduce_the_confusion_rate(method):
    shots = 10_000
    counts = (
        Simulator(method=method, noise=_readout_model(_FLIP_30))
        .run(_measured_program(), shots=shots, simulation_config={"seed": 2})
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
    program.measure((0, 1), (0, 1))
    counts = (
        Simulator(noise=_readout_model(always_flip, target=1))
        .run(program, shots=200, simulation_config={"seed": 4})
        .result()
        .get_counts()
    )

    assert counts == {"11": 200}


# --- execution: dynamic path (true collapse vs corrupted report) ---


def test_feedforward_reads_the_reported_bit_not_the_true_one():
    # q0 is |0>: the true outcome is 0, but readout always reports 1, so the
    # condition c0 == 1 fires and flips q1.
    program = fq.Program(2, 2)
    program.measure(0, 0)
    program.add(fq.ops.X, 1, condition=(0, 1))
    program.measure(1, 1)
    counts = (
        Simulator(method="SV", noise=_readout_model(_ALWAYS_ONE, target=0))
        .run(program, shots=100, simulation_config={"seed": 3})
        .result()
        .get_counts()
    )

    assert counts == {"11": 100}


def test_collapse_and_state_export_keep_the_true_outcome():
    # Despite the always-report-1 readout, the post-measurement state of the
    # |0> qubit must still be exactly |0>: readout confusion is classical only.
    program = _measured_program()
    result = (
        Simulator(method="SV", noise=_readout_model(_ALWAYS_ONE))
        .run(
            program,
            shots=1,
            simulation_config={"seed": 5},
            result_config={"counts": True, "final_state": True},
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
    noise.add(ReadoutConfusion(always_flip), targets=0)
    program = fq.Program(1, 2)
    program.measure(0, 0)
    program.add(fq.ops.X, 0)
    program.measure(0, 1)
    counts = (
        Simulator(method="SV", noise=noise)
        .run(program, shots=50, simulation_config={"seed": 6})
        .result()
        .get_counts()
    )

    assert counts == {"01": 50}


def test_threaded_compiled_shots_match_serial_with_readout_confusion():
    noise = _readout_model(_FLIP_30)
    program = fq.Program(1, 2)
    program.measure(0, 0)
    program.add(fq.ops.X, 0, condition=(0, 1))  # forces the dynamic path
    program.measure(0, 1)
    serial = (
        Simulator(noise=noise)
        .run(
            program,
            shots=8,
            simulation_config={
                "seed": 13,
                "shot_parallelism": "serial",
                "kernel_parallelism": "serial",
            },
        )
        .result()
        .get_counts()
    )
    parallel = (
        Simulator(noise=noise)
        .run(
            program,
            shots=8,
            simulation_config={
                "seed": 13,
                "shot_parallelism": "threads",
                "kernel_parallelism": "serial",
                "max_workers": 2,
            },
        )
        .result()
        .get_counts()
    )

    assert parallel == serial


def test_numba_compiled_multi_shot_kernel_applies_readout_confusion():
    # Regression: the compiled multi-shot kernel used to write the true measured
    # digit straight into the classical register, so a dynamic plan's readout
    # error was silently dropped under runtime="numba" - X then always-flip
    # readout reported "1" instead of "0".
    pytest.importorskip("numba")
    from fatqat.simulator._engine.nb import _plan_compilable

    always_flip = np.array([[0.0, 1.0], [1.0, 0.0]])
    noise = _readout_model(always_flip)
    program = fq.Program(1, 1)
    program.add(fq.ops.X, 0)
    program.measure(0, 0)
    program.add(fq.ops.Reset, 0)  # a reset forces the dynamic path

    plan, _ = Simulator(noise=noise)._lower_program(program)
    assert _plan_compilable(plan) is True

    counts = {}
    for runtime in ("numpy", "numba"):
        counts[runtime] = (
            Simulator(method="SV", runtime=runtime, noise=noise)
            .run(program, shots=64, simulation_config={"seed": 5})
            .result()
            .get_counts()
        )

    assert counts["numba"] == counts["numpy"] == {"0": 64}


def test_numba_compiled_multi_shot_matches_numpy_on_a_qudit_confusion_plan():
    # A non-deterministic 3x3 confusion on a qutrit, mixed with a channel, a
    # feedforward gate and a reset: the reported digits are resampled in the
    # kernel, and the extra per-confusion uniform must keep the whole per-shot
    # RNG stream aligned with the NumPy path for the counts to agree exactly.
    pytest.importorskip("numba")
    from fatqat.noise import PhaseDamping

    confusion = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.2], [0.1, 0.1, 0.7]])

    def counts_for(runtime):
        noise = NoiseModel()
        noise.add(ReadoutConfusion(confusion))
        noise.add(PhaseDamping(p=0.2), operation=fq.ops.Shift)
        qreg = fq.QuantumRegister(2, dim=3)
        creg = fq.ClassicalRegister(2, dim=3)
        program = fq.Program([qreg], [creg])
        program.add(fq.ops.Shift(1), qreg[0])
        program.measure(qreg[0], creg[0])
        program.add(fq.ops.Shift(2), qreg[1], condition=(creg[0], 1))
        program.add(fq.ops.Reset, qreg[0])
        program.measure(qreg[1], creg[1])
        return (
            Simulator(method="SV", runtime=runtime, noise=noise)
            .run(program, shots=300, simulation_config={"seed": 21})
            .result()
            .get_counts()
        )

    numpy_counts = counts_for("numpy")
    # Every digit is reachable through the confusion, so the corrupted reports
    # really are being sampled rather than passed through.
    assert len(numpy_counts) > 1
    assert counts_for("numba") == numpy_counts


def test_numba_partially_confused_measurement_only_draws_where_attached():
    # One confused subsystem out of two in a single measurement step: the
    # error-free subsystem must consume no uniform, or every later draw in the
    # shot shifts and the counts diverge from the NumPy path.
    pytest.importorskip("numba")

    noise = NoiseModel()
    noise.add(ReadoutConfusion(_FLIP_30), targets=1)  # q0 reports without error
    program = fq.Program(2, 2)
    program.add(fq.ops.H, 0)
    program.measure((0, 1), (0, 1))
    program.add(fq.ops.Reset, 0)  # forces the dynamic path

    counts = {
        runtime: Simulator(method="SV", runtime=runtime, noise=noise)
        .run(program, shots=200, simulation_config={"seed": 8})
        .result()
        .get_counts()
        for runtime in ("numpy", "numba")
    }

    assert counts["numba"] == counts["numpy"]


def test_check_noise_support_reports_readout_confusion_as_accepted():
    report = Simulator().check_noise_support(_readout_model(_FLIP_30))

    assert report.supported is True
    assert "ReadoutConfusion" in report.accepted_sources


# --- validate_for: run() direct-raise strict selector-identity validation ---


def test_run_rejects_foreign_logical_readout_selector_directly():
    program = _measured_program()
    foreign = fq.QuantumRegister(1, name="q")
    backend = Simulator(noise=_readout_model(_FLIP_30, target=foreign[0]))

    with pytest.raises(BackendValidationError):
        backend.run(program)


def test_run_rejects_unmapped_physical_readout_label_directly():
    program = _measured_program()
    backend = Simulator(noise=_readout_model(_FLIP_30, target=99))

    with pytest.raises(BackendValidationError):
        backend.run(program)


def test_run_succeeds_when_valid_readout_selector_targets_unmeasured_subsystem():
    # A valid selector (real device label) naming a subsystem that is never
    # measured is a permitted no-effect entry, not a validation error.
    program = fq.Program(2, 1)
    program.measure(0, 0)
    backend = Simulator(noise=_readout_model(_FLIP_30, target=1))

    result = backend.run(program).result()
    assert result is not None
