"""Turning gate fusion off, and what that is for.

Fusion merges adjacent plan steps into wider ones. It computes the same
quantity by a different association, so results move in the last bits. The
switch exists so that a caller comparing numbers - across runtimes, or against
an independent implementation - can take that variable out.
"""

import numpy as np
import pytest

import fatqat as fq
from fatqat import operations as ops
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


def test_fusion_uses_the_runtime_default_by_default():
    result = Simulator(method="DM", runtime="numba").run(
        _noisy_program(), shots=0, result_config=_STATE_ONLY
    )
    assert result.result().metadata["simulation_config"]["fusion"] is None


@pytest.mark.parametrize("fusion", [True, False])
def test_numpy_runtime_rejects_an_explicit_fusion_setting(fusion):
    with pytest.raises(BackendValidationError, match="only supported.*numba"):
        Simulator(method="DM", runtime="numpy").run(
            _noisy_program(), simulation_config={"fusion": fusion}
        )


def test_numpy_runtime_accepts_the_default():
    result = Simulator(method="DM", runtime="numpy").run(_noisy_program()).result()
    assert result.metadata["simulation_config"]["fusion"] is None


@pytest.mark.parametrize("bad", ["yes", 1, 0])
def test_a_non_boolean_is_rejected(bad):
    with pytest.raises(BackendValidationError, match="fusion must be a bool or None"):
        Simulator(method="DM", runtime="numba").run(
            _noisy_program(), simulation_config={"fusion": bad}
        )


# --- what it changes, and what it must not -------------------------------


def test_the_two_settings_agree_on_the_answer():
    # The switch is about association order, not about what is computed.
    assert _density_matrix(fusion=True) == pytest.approx(
        _density_matrix(fusion=False), abs=1e-12
    )


def test_disabling_fusion_changes_the_arithmetic_order():
    # If this ever passes as bit-identical, the switch has stopped reaching the
    # fuser and every other test here would still pass.
    fused = _density_matrix(fusion=True)
    unfused = _density_matrix(fusion=False)

    assert not np.array_equal(fused, unfused)
    assert np.abs(fused - unfused).max() < 1e-12


def test_the_unfused_numba_run_still_only_agrees_with_numpy_to_rounding():
    # Worth stating: fusion is not the only reordering between the runtimes, so
    # fusion=False does not promise bit-identity with numpy - only that one
    # source of difference has been removed.
    unfused = _density_matrix(fusion=False)
    reference = _density_matrix(fusion=None, runtime="numpy")

    assert unfused == pytest.approx(reference, abs=1e-12)


# --- the wiring ----------------------------------------------------------


@pytest.mark.parametrize("method", ["DM", "superop"])
@pytest.mark.parametrize("fusion, expected_calls", [(None, 1), (True, 1), (False, 0)])
def test_the_setting_reaches_the_gate_channel_fuser(method, fusion, expected_calls):
    # Pin the wiring, not just the numbers: a switch that never reached the
    # fuser would still satisfy the agreement tests above.
    from fatqat.simulator._engine import nb

    calls = []
    original = nb._fuse_gate_channels

    def _counting(plan):
        calls.append(len(plan))
        return original(plan)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(nb, "_fuse_gate_channels", _counting)
        Simulator(method=method, runtime="numba", noise=_noise()).run(
            _noisy_program(),
            shots=0,
            simulation_config={"fusion": fusion},
            result_config=_STATE_ONLY,
        ).result()

    assert len(calls) == expected_calls


@pytest.mark.parametrize("method", ["unitary", "superop"])
@pytest.mark.parametrize("fusion, expected_calls", [(None, 1), (True, 1), (False, 0)])
def test_the_setting_reaches_the_operator_payload_fuser(method, fusion, expected_calls):
    # Operator fusion has a separate size threshold and implementation from
    # gate/channel fusion.  One public switch must control both, so force the
    # size gate open and pin this second wiring path explicitly.
    from fatqat.simulator._engine import nb

    calls = []
    original = nb._fuse_operator_payloads

    def _counting(payloads, dims):
        calls.append(len(payloads))
        return original(payloads, dims)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(nb, "_MIN_SIZE_TO_FUSE", 0)
        patch.setattr(nb, "_fuse_operator_payloads", _counting)
        Simulator(method=method, runtime="numba").run(
            _noisy_program(),
            shots=0,
            simulation_config={"fusion": fusion},
            result_config=_STATE_ONLY,
        ).result()

    assert len(calls) == expected_calls
