"""Public atom-occupancy checks and their execution boundaries."""

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.errors import BackendValidationError, UnsupportedOperationError
from fatqat.qasm import QasmExportError, to_qasm


@pytest.mark.parametrize("runtime", ["numpy", "numba"])
@pytest.mark.parametrize("method", ["statevector", "density_matrix"])
def test_check_records_loss_and_reloading(runtime, method):
    noise = fq.NoiseModel()
    noise.add(fq.noise.Loss(p=1), operation=ops.RX)
    program = fq.Program(1, 4)
    program.check_occupancy(0, 0)
    program.add(ops.Put, 0)
    program.check_occupancy(0, 1)
    program.add(ops.RX(np.pi), 0)
    program.check_occupancy(0, 2)
    program.add(ops.Put, 0)
    program.check_occupancy(0, 3)

    counts = (
        fq.simulator.AtomArraySimulator(method=method, runtime=runtime, noise=noise)
        .run(program, shots=5, simulation_config={"seed": 7})
        .result()
        .get_counts()
    )
    assert counts == {"0101": 5}


def test_check_is_non_destructive_and_can_drive_feedforward():
    program = fq.Program(1, 2)
    program.add(ops.Put, 0)
    program.add(ops.RY(np.pi / 2), 0)
    program.check_occupancy(0, 0)
    program.add(ops.RY(-np.pi / 2), 0, condition=(0, 1))
    program.measure(0, 1)

    counts = (
        fq.simulator.AtomArraySimulator()
        .run(program, shots=20, simulation_config={"seed": 7})
        .result()
        .get_counts()
    )
    assert counts == {"10": 20}


def test_check_can_control_conditional_reload_and_reuse_output():
    noise = fq.NoiseModel()
    noise.add(fq.noise.Loss(p=1), operation=ops.RX)
    program = fq.Program(1, 1)
    program.add(ops.Put, 0)
    program.add(ops.RX(np.pi), 0)
    program.check_occupancy(0, 0)
    program.add(ops.Put, 0, condition=(0, 0))
    program.check_occupancy(0, 0)

    counts = (
        fq.simulator.AtomArraySimulator(noise=noise)
        .run(program, shots=4, simulation_config={"seed": 4})
        .result()
        .get_counts()
    )
    assert counts == {"1": 4}


def test_check_accepts_multiple_sites_and_last_write_wins():
    program = fq.Program(2, 1)
    program.add(ops.Put, 0)
    program.check_occupancy((0, 1), (0, 0))
    assert fq.simulator.AtomArraySimulator().run(
        program, shots=3
    ).result().get_counts() == {"0": 3}


def test_check_preserves_site_order_and_ignores_measurement_confusion():
    noise = fq.NoiseModel()
    noise.add(fq.noise.ReadoutConfusion([[0, 1], [1, 0]]))
    program = fq.Program(2, 2)
    program.add(ops.Put, 0)
    program.check_occupancy((0, 1), (0, 1))
    assert fq.simulator.AtomArraySimulator(noise=noise).run(
        program, shots=3
    ).result().get_counts() == {"10": 3}


def test_check_validates_pairs_and_binary_output():
    program = fq.Program(2, 2)
    with pytest.raises(ValueError, match="equal nonzero"):
        program.check_occupancy((0, 1), 0)
    with pytest.raises(ValueError, match="equal nonzero"):
        program.check_occupancy((), ())
    qutrit_output = fq.ClassicalRegister(1, dim=3)
    other = fq.Program(1, [qutrit_output])
    with pytest.raises(ValueError, match="dimension 2"):
        other.check_occupancy(0, qutrit_output[0])


def test_check_is_not_silently_exported_or_run_on_other_backends():
    program = fq.Program(1, 1)
    program.check_occupancy(0, 0)
    with pytest.raises(UnsupportedOperationError, match="AtomArraySimulator"):
        fq.simulator.Simulator().run(program, shots=1)
    with pytest.raises(QasmExportError, match="occupancy"):
        to_qasm(program)
    with pytest.raises(BackendValidationError, match="occupancy check"):
        fq.Estimator(fq.simulator.AtomArraySimulator()).run(
            program, fq.Observable([("Z", 1)])
        )
    with pytest.raises(ValueError, match="LogicalProgram"):
        fq.LogicalProgram(1, 1).check_occupancy(0, 0)


def test_check_survives_copy_and_is_a_classical_dependency():
    program = fq.Program(1, 1)
    program.check_occupancy(0, 0)
    program.add(ops.RX(np.pi), 0, condition=(0, 0))
    copied = program.copy()
    assert fq.simulator.AtomArraySimulator().run(
        copied, shots=2
    ).result().get_counts() == {"0": 2}
    nodes = copied.dag().nodes
    assert nodes[0].kind == "occupancy_check"
    assert any(edge.reason == "classical" for edge in copied.dag().edges)
