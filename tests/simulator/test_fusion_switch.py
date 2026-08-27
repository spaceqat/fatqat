"""Turning gate fusion off without changing the numerical result."""

import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.errors import BackendValidationError
from fatqat.simulator import Simulator

pytest.importorskip("numba")

_STATE_ONLY = {"counts": False, "final_state": True}


def _noisy_program():
    """Gate/channel pairs on the same targets, which is what the DM fuser merges."""
    program = fq.Program(3)
    for qubit in range(3):
        program.add(ops.H, qubit)
    for qubit in range(2):
        program.add(ops.CX, (qubit, qubit + 1))
    return program


def _noise():
    model = fq.NoiseModel()
    model.add(fq.noise.Depolarizing(p=0.05), operation=ops.CX)
    return model


def _density_matrix(*, fusion, runtime="numba"):
    return (
        Simulator(method="DM", runtime=runtime, noise=_noise())
        .run(
            _noisy_program(),
            shots=0,
            simulation_config={"fusion": fusion},
            result_config=_STATE_ONLY,
        )
        .result()
        .get_density_matrix()
    )


# --- the switch itself ---------------------------------------------------


def test_fusion_is_off_by_default():
    result = Simulator(method="DM", runtime="numba").run(
        _noisy_program(), shots=0, result_config=_STATE_ONLY
    )
    assert result.result().metadata["simulation_config"]["fusion"] is False


def test_numpy_runtime_rejects_enabling_fusion():
    with pytest.raises(BackendValidationError, match="not supported.*matrix engine"):
        Simulator(method="DM", runtime="numpy").run(
            _noisy_program(), simulation_config={"fusion": True}
        )


def test_numpy_runtime_accepts_fusion_off():
    result = (
        Simulator(method="DM", runtime="numpy")
        .run(_noisy_program(), simulation_config={"fusion": False})
        .result()
    )
    assert result.metadata["simulation_config"]["fusion"] is False


def test_numba_statevector_rejects_enabling_fusion():
    with pytest.raises(BackendValidationError, match="compiled multi-shot"):
        Simulator(method="SV", runtime="numba").run(
            _noisy_program(), simulation_config={"fusion": True}
        )


@pytest.mark.parametrize("bad", ["yes", 1, 0])
def test_a_non_boolean_is_rejected(bad):
    with pytest.raises(BackendValidationError, match="fusion must be a bool"):
        Simulator(method="DM", runtime="numba").run(
            _noisy_program(), simulation_config={"fusion": bad}
        )


# --- what it changes, and what it must not -------------------------------


def test_the_two_settings_agree_on_the_answer():
    # The switch is about association order, not about what is computed.
    assert _density_matrix(fusion=True) == pytest.approx(
        _density_matrix(fusion=False), abs=1e-12
    )


def test_public_fusion_switch_selects_the_materialized_rewrite(monkeypatch):
    # The production rewrite preserves semantics, so a deterministic sentinel
    # gives the public switch an exact observable effect without relying on
    # platform-specific floating-point association differences.
    from fatqat.simulator._engine import nb

    monkeypatch.setattr(nb, "_fuse_gate_channels", lambda plan: plan[-1:])
    noise = fq.NoiseModel()
    noise.add(fq.noise.Depolarizing(p=0.0), operation=ops.X)
    program = fq.Program(1, 1)
    program.add(ops.X, 0)
    program.measure(0, 0)
    simulator = Simulator(method="DM", runtime="numba", noise=noise)

    def counts(*, fusion):
        return (
            simulator.run(
                program,
                shots=4,
                simulation_config={
                    "fusion": fusion,
                    "shot_parallelism": "serial",
                    "kernel_parallelism": "serial",
                },
                result_config={"counts": True, "final_state": False},
            )
            .result()
            .get_counts()
        )

    assert counts(fusion=False) == {"1": 4}
    assert counts(fusion=True) == {"0": 4}


def test_the_unfused_numba_run_still_only_agrees_with_numpy_to_rounding():
    # Worth stating: fusion is not the only reordering between the runtimes, so
    # fusion=False does not promise bit-identity with numpy - only that one
    # source of difference has been removed.
    unfused = _density_matrix(fusion=False)
    reference = _density_matrix(fusion=False, runtime="numpy")

    assert unfused == pytest.approx(reference, abs=1e-12)
