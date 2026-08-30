"""Public three-level atom backend ownership and execution coverage."""

from copy import deepcopy
import warnings

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat._pulse_values import PulseControl
from fatqat.emulator._core.backend import _PulseBackend
from fatqat.errors import (
    BackendExecutionError,
    BackendValidationError,
    UnsupportedOperationError,
)
from fatqat.noise import (
    AmplitudeDamping,
    Depolarizing,
    NoiseModel,
    PhaseDamping,
    ThermalRelaxation,
)
from fatqat.registers import RegisterRef
from fatqat.emulator import SampledWaveform


def _backend(
    atom_3level_model,
    atom_3level_calibration,
    *,
    noise=None,
    rows=1,
    cols=2,
    spacing=2.0,
    method="density_matrix",
):
    return fq.emulator.Atom3LevelEmulator(
        atom_3level_model,
        arrangement=fq.emulator.AtomArrangement.rectangular(rows, cols, spacing),
        method=method,
        noise=noise,
    )


def test_atom_3level_public_identity_and_private_ownership(
    atom_3level_model, atom_3level_calibration
):
    assert (
        fq.emulator.Atom3LevelEmulator.__module__
        == "fatqat.emulator.atom_3level.backend"
    )
    arrangement = fq.emulator.AtomArrangement.rectangular(1, 2, 2.0)
    backend = fq.emulator.Atom3LevelEmulator(atom_3level_model, arrangement=arrangement)
    assert backend.model is atom_3level_model
    assert backend.arrangement is arrangement
    assert type(backend).__bases__ == (_PulseBackend,)
    assert not hasattr(backend, "_dele" + "gate")
    assert not any(
        name in type(backend).__dict__
        for name in ("run", "propagator", "_prepare_program", "_execute")
    )
    for name, value in (
        ("model", atom_3level_model),
        ("arrangement", arrangement),
    ):
        with pytest.raises(AttributeError):
            setattr(backend, name, value)
    assert not hasattr(backend, "noise")
    for keyword in (
        "coordinates",
        "runtime",
        "adapter",
    ):
        with pytest.raises(TypeError, match=keyword):
            fq.emulator.Atom3LevelEmulator(
                atom_3level_model,
                arrangement=arrangement,
                **{keyword: object()},
            )
    assert not hasattr(fq.emulator, "_prepare_atom_3level_runtime")
    assert not hasattr(fq.emulator, "_Atom3LevelQutipAdapter")


def test_atom_3level_backend_lowers_model_factory_direct_control(
    atom_3level_model, atom_3level_calibration
):
    backend = _backend(atom_3level_model, atom_3level_calibration)
    program = fq.Program(2)
    operation = ops.PulseOperation(
        1.0,
        (
            PulseControl(
                atom_3level_model.control.raman(0),
                SampledWaveform((0.0, 1.0), (0.0, 0.1j)),
            ),
        ),
    )
    program.add(operation)

    (block,) = backend._prepare_program(program).plan
    assert block.controls == operation.controls
    assert block.target_indices == (0,)


def test_atom3_coherent_statevector_is_flat_and_full_physical(
    atom_3level_model,
):
    backend = fq.emulator.Atom3LevelEmulator(
        atom_3level_model,
        arrangement=fq.emulator.AtomArrangement.rectangular(1, 2, 2.0),
        method="statevector",
    )
    program = fq.Program(2)
    program.add(ops.RX(np.pi / 2), 0)

    result = backend.run(program).result()
    state = result.get_statevector()

    assert result.available_data == frozenset({"statevector"})
    assert state.shape == (9,)
    assert np.linalg.norm(state) == pytest.approx(1.0)
    assert abs(state[0]) < 0.999


def test_atom_3level_result_shot_validation_preserves_exact_messages(
    atom_3level_model, atom_3level_calibration
):
    backend = _backend(atom_3level_model, atom_3level_calibration)
    program = fq.Program(2, 1)
    program.measure(0, 0)

    with pytest.raises(BackendValidationError) as exc:
        backend.run(program, shots=1.5)
    assert str(exc.value) == (
        "shots must be an int when requested results depend on it"
    )

    with pytest.raises(BackendValidationError) as exc:
        backend.run(program, shots=2, result_config={"final_state": True})
    assert str(exc.value) == (
        "density_matrix with physical measurement sampling is only supported "
        "for shots == 1"
    )


def test_atom_3level_measurement_keeps_the_qutrit_to_bit_digit_map(
    atom_3level_model, atom_3level_calibration
):
    backend = _backend(atom_3level_model, atom_3level_calibration)
    program = fq.Program(2, 1)
    program.measure(0, 0)

    plan = backend._prepare_program(program).plan

    assert plan[0].reported_digit_maps == ((0, 1, 1),)


def test_atom_3level_default_map_does_not_bind_calibration_to_model_identity(
    atom_3level_model,
    atom_3level_calibration,
    atom_3level_model_document,
):
    same_identity = deepcopy(atom_3level_model_document)
    same_identity["parameters"]["c6"] *= -1
    declared_same = fq.emulator.Atom3LevelModel.from_document(same_identity)
    assert _backend(declared_same, atom_3level_calibration).model is declared_same

    changed_identity = deepcopy(atom_3level_model_document)
    changed_identity["model"]["revision"] = "new-revision"
    changed_model = fq.emulator.Atom3LevelModel.from_document(changed_identity)
    assert _backend(changed_model, atom_3level_calibration).model is changed_model


def test_atom_3level_supplied_map_is_type_checked_copied_and_empty_is_explicit(
    atom_3level_model, atom_3level_calibration
):
    arrangement = fq.emulator.AtomArrangement.rectangular(1, 2, 2.0)
    with pytest.raises(BackendValidationError, match="PulseImplementationMap"):
        fq.emulator.Atom3LevelEmulator(
            atom_3level_model,
            arrangement=arrangement,
            gate_implementation_map=object(),
        )
    supplied = fq.emulator.atom_3level.default_atom_3level_gate_implementation_map(
        model=atom_3level_model, calibration=atom_3level_calibration
    )
    backend = fq.emulator.Atom3LevelEmulator(
        atom_3level_model,
        arrangement=arrangement,
        method="unitary",
        gate_implementation_map=supplied,
    )
    assert backend._gate_implementation_map is not supplied
    supplied.remove(ops.RX)
    program = fq.Program(2)
    program.add(ops.RX(0.2), 0)
    assert backend.run(program).result().get_unitary().shape == (9, 9)

    empty = fq.emulator.PulseImplementationMap()
    direct_only = fq.emulator.Atom3LevelEmulator(
        atom_3level_model,
        arrangement=arrangement,
        method="unitary",
        gate_implementation_map=empty,
    )
    with pytest.raises(UnsupportedOperationError, match="RX is not supported"):
        direct_only.run(program)


def test_compiled_map_transfers_while_target_c6_and_geometry_control_evolution(
    atom_3level_model,
    atom_3level_calibration,
    atom_3level_model_document,
):
    compiled = fq.emulator.atom_3level.default_atom_3level_gate_implementation_map(
        model=atom_3level_model, calibration=atom_3level_calibration
    )
    changed_document = deepcopy(atom_3level_model_document)
    changed_document["model"]["revision"] = "finer-target"
    changed_document["parameters"]["c6"] *= -1
    changed_model = fq.emulator.Atom3LevelModel.from_document(changed_document)
    source_target = fq.emulator.Atom3LevelEmulator(
        atom_3level_model,
        arrangement=fq.emulator.AtomArrangement.rectangular(1, 2, 2.0),
        method="unitary",
        gate_implementation_map=compiled,
    )
    changed_target = fq.emulator.Atom3LevelEmulator(
        changed_model,
        arrangement=fq.emulator.AtomArrangement.rectangular(1, 2, 3.0),
        method="unitary",
        gate_implementation_map=compiled,
    )
    program = fq.Program(2)
    program.add(ops.CZ, (0, 1))
    source_plan = source_target._prepare_program(program).plan
    changed_plan = changed_target._prepare_program(program).plan
    for first, second in zip(
        source_plan[0].controls, changed_plan[0].controls, strict=True
    ):
        assert first.channel == second.channel
        assert np.array_equal(first.waveform.values, second.waveform.values)
    assert not np.allclose(
        source_target.run(program).result().get_unitary(),
        changed_target.run(program).result().get_unitary(),
    )


def test_custom_map_uses_only_public_atom_structural_authoring_values(
    atom_3level_model,
):
    implementation_map = fq.emulator.PulseImplementationMap()

    def custom_x(_operation, *, device_operands):
        (site,) = device_operands
        waveform = SampledWaveform((0.0, 0.2), (0.1, 0.1))
        return fq.emulator.PulseDefinition(
            0.2,
            (
                PulseControl(atom_3level_model.control.raman(site), waveform),
                PulseControl(atom_3level_model.control.rydberg(site), waveform),
            ),
            (fq.emulator.PhaseShift(atom_3level_model.frame(site), 0.1),),
        )

    implementation_map.add(ops.X, custom_x)
    backend = fq.emulator.Atom3LevelEmulator(
        atom_3level_model,
        arrangement=fq.emulator.AtomArrangement.rectangular(1, 1, 2.0),
        gate_implementation_map=implementation_map,
    )
    program = fq.Program(1)
    program.add(ops.X, 0)
    (block,) = backend._prepare_program(program).plan
    assert len(block.controls) == 2
    assert block.control_bindings[0].engine_indices == (0,)
    assert block.control_bindings[0].engine_indices == (0,)


def test_atom_3level_noise_is_retained_validated_and_binary(
    atom_3level_model, atom_3level_calibration
):
    empty = NoiseModel()
    backend = _backend(atom_3level_model, atom_3level_calibration, noise=empty)
    assert backend._noise_model is not empty
    assert backend._noise_model._noise_sources() == ()
    assert backend.validate_noise_model(empty) is None
    readout = _readout(np.array([[0.9, 0.1], [0.1, 0.9]]))
    assert backend.validate_noise_model(readout) is None

    for channel in (
        ThermalRelaxation(t1=5.0, t2=4.0),
        AmplitudeDamping(p=0.1),
        PhaseDamping(p=0.1),
        Depolarizing(p=0.1),
    ):
        rejected = NoiseModel()
        rejected.add(channel, operation=ops.X)
        with pytest.raises(BackendValidationError, match=type(channel).__name__):
            _backend(atom_3level_model, atom_3level_calibration, noise=rejected)

    empty.add(PhaseDamping(rate=0.1), operation=ops.X)
    with pytest.raises(BackendValidationError, match="PhaseDamping"):
        backend.validate_noise_model(empty)
    driven = fq.Program(2)
    driven.add(ops.RX(0.1), 0)
    backend.run(driven).result()
    _backend(
        atom_3level_model,
        atom_3level_calibration,
        method="unitary",
    ).run(driven).result().get_unitary()

    with pytest.raises(BackendValidationError, match="2 x 2"):
        _backend(atom_3level_model, atom_3level_calibration, noise=_readout(np.eye(3)))


def _readout(matrix):
    model = NoiseModel()
    model.add(fq.noise.ReadoutConfusion(matrix))
    return model


def _capture_adapter_bindings(monkeypatch):
    from fatqat.emulator.atom_3level import qutip_adapter as atom_3level_qutip_adapter

    captured = []
    actual = atom_3level_qutip_adapter._Atom3LevelQutipAdapter

    def capture(target, **kwargs):
        captured.append((target, kwargs["engine_allocation"]))
        return actual(target, **kwargs)

    monkeypatch.setattr(atom_3level_qutip_adapter, "_Atom3LevelQutipAdapter", capture)
    return captured


def test_atom_3level_binding_is_exact_row_major_all_occupied_for_run_and_unitary(
    atom_3level_model, atom_3level_calibration, monkeypatch
):
    backend = _backend(atom_3level_model, atom_3level_calibration)
    unitary_backend = _backend(
        atom_3level_model, atom_3level_calibration, method="unitary"
    )
    captured = _capture_adapter_bindings(monkeypatch)
    run_program = fq.Program(2)
    run_program.add(ops.RZ(0.2), 0)
    backend.run(
        run_program, result_config={"counts": False, "final_state": True}
    ).result()
    propagate_program = fq.Program(2)
    propagate_program.add(ops.RZ(0.3), 1)
    unitary_backend.run(propagate_program).result().get_unitary()
    assert len(captured) == 2
    for target, allocation in captured:
        assert target in (backend._target, unitary_backend._target)
        assert allocation.device_operands == (0, 1)
        assert allocation.system_dims == (3, 3)
        assert tuple(
            interaction.distance_um for interaction in target.interactions
        ) == (2.0,)


def test_atom_3level_declaration_order_binds_multi_register_nonprefix_refs(
    atom_3level_model, atom_3level_calibration, monkeypatch
):
    left = fq.QuantumRegister(1, name="left")
    right = fq.QuantumRegister(1, name="right")
    program = fq.Program([left, right])
    # The first operation addresses the second declaration, not a prefix.
    program.add(ops.RZ(0.2), right[0])
    backend = _backend(atom_3level_model, atom_3level_calibration, method="unitary")
    captured = _capture_adapter_bindings(monkeypatch)
    propagator = backend.run(program).result().get_unitary()
    assert len(captured) == 1
    target, allocation = captured[0]
    assert target is backend._target
    assert allocation.device_operands == (0, 1)
    assert allocation.system_dims == (3, 3)
    assert np.allclose(
        propagator,
        np.kron(np.diag((1.0, np.exp(0.2j), 1.0)), np.eye(3)),
    )


def test_atom_3level_exact_binding_dimension_capacity_and_qutrit_result(
    atom_3level_model, atom_3level_calibration
):
    backend = _backend(atom_3level_model, atom_3level_calibration)
    with pytest.raises(BackendValidationError, match="exactly"):
        backend.run(fq.Program(1))
    with pytest.raises(BackendValidationError, match="exactly"):
        backend.run(fq.Program(3))
    with pytest.raises(BackendValidationError, match="dimension-two"):
        backend.run(fq.Program([fq.QuantumRegister(2, dim=3)]))
    program = fq.Program(2)
    program.add(ops.RX(np.pi), 0)
    result = backend.run(
        program, result_config={"counts": False, "final_state": True}
    ).result()
    assert result.get_density_matrix().shape == (9, 9)


def test_atom_3level_binary_measurement_reset_and_readout_unitary_inert(
    atom_3level_model, atom_3level_calibration
):
    backend = _backend(
        atom_3level_model,
        atom_3level_calibration,
        noise=_readout(np.array([[0.0, 1.0], [1.0, 0.0]])),
    )
    program = fq.Program(2, 1)
    program.add(ops.RX(np.pi), 0)
    program.measure(0, 0)
    program.add(ops.Reset, 0)
    result = backend.run(
        program, shots=1, result_config={"counts": True, "final_state": True}
    ).result()
    assert result.get_counts() == {"0": 1}
    assert result.get_density_matrix().shape == (9, 9)
    coherent = fq.Program(2)
    coherent.add(ops.RZ(0.2), 0)
    noisy_unitary = _backend(
        atom_3level_model,
        atom_3level_calibration,
        noise=backend._noise_model,
        method="unitary",
    )
    empty = _backend(atom_3level_model, atom_3level_calibration, method="unitary")
    assert np.allclose(
        noisy_unitary.run(coherent).result().get_unitary(),
        empty.run(coherent).result().get_unitary(),
    )


@pytest.mark.parametrize(
    ("first", "second", "expected_index"),
    ((0, 0, 0), (0, 1, 3), (1, 0, 1), (1, 1, 4)),
)
def test_atom_3level_all_computational_inputs_remain_physical_qutrit_states(
    atom_3level_model, atom_3level_calibration, first, second, expected_index
):
    backend = _backend(atom_3level_model, atom_3level_calibration)
    program = fq.Program(2)
    if first:
        program.add(ops.RX(np.pi), 0)
    if second:
        program.add(ops.RX(np.pi), 1)
    density = (
        backend.run(program, result_config={"counts": False, "final_state": True})
        .result()
        .get_density_matrix()
    )
    assert density.shape == (9, 9)
    assert density[expected_index, expected_index].real > 0.999999
    assert np.isclose(np.trace(density), 1.0, atol=1e-8)


def test_atom_3level_superposition_and_final_frame_unitary(
    atom_3level_model, atom_3level_calibration
):
    backend = _backend(atom_3level_model, atom_3level_calibration)
    superposition = fq.Program(2)
    superposition.add(ops.RX(np.pi / 2), 0)
    density = (
        backend.run(superposition, result_config={"counts": False, "final_state": True})
        .result()
        .get_density_matrix()
    )
    assert density[0, 0].real == pytest.approx(0.5, abs=2e-6)
    assert density[1, 1].real == pytest.approx(0.5, abs=2e-6)
    assert abs(density[0, 1]) > 0.49

    framed = fq.Program(2)
    framed.add(ops.RZ(0.37), 0)
    final = (
        _backend(atom_3level_model, atom_3level_calibration, method="unitary")
        .run(framed)
        .result()
        .get_unitary()
    )
    assert not np.allclose(final, np.eye(9))


def test_atom_3level_measurement_returns_the_physical_single_shot_posterior_before_reset(
    atom_3level_model, atom_3level_calibration
):
    backend = _backend(atom_3level_model, atom_3level_calibration)
    program = fq.Program(2, 1)
    program.add(ops.RX(np.pi / 2), 0)
    program.measure(0, 0)
    result = backend.run(
        program,
        shots=1,
        simulation_config={"seed": 23},
        result_config={"counts": True, "final_state": True},
    ).result()
    density = result.get_density_matrix()
    assert density.shape == (9, 9)
    # Canonical axis 0 is the least-significant qutrit digit, so the physical
    # posterior occupies flat index 0 or 1, never a binary-only artifact.
    counts = result.get_counts()
    assert counts in ({"0": 1}, {"1": 1})
    outcome = int(next(iter(counts)))
    assert density[outcome, outcome].real > 0.999999


def test_atom_3level_feedforward_uses_the_reported_binary_bit_and_sampling_is_seeded(
    atom_3level_model, atom_3level_calibration
):
    noise = _readout(np.array([[0.0, 1.0], [1.0, 0.0]]))
    backend = _backend(atom_3level_model, atom_3level_calibration, noise=noise)
    program = fq.Program(2, 1)
    program.add(ops.RX(np.pi), 0)
    program.measure(0, 0)
    # Physical |1> reports as 0 under confusion, so this conditional rotation
    # executes and returns the qutrit to |0>.
    program.add(ops.RX(np.pi), 0, condition=(0, 0))
    first = backend.run(
        program,
        shots=1,
        simulation_config={"seed": 17},
        result_config={"counts": True, "final_state": True},
    ).result()
    second = backend.run(
        program,
        shots=1,
        simulation_config={"seed": 17},
        result_config={"counts": True, "final_state": True},
    ).result()
    assert first.get_counts() == second.get_counts() == {"0": 1}
    assert first.get_density_matrix()[0, 0].real > 0.999


def test_atom_3level_stochastic_measurement_is_seed_reproducible(
    atom_3level_model, atom_3level_calibration
):
    backend = _backend(atom_3level_model, atom_3level_calibration)
    program = fq.Program(2, 1)
    program.add(ops.RX(np.pi / 2), 0)
    program.measure(0, 0)
    config = {"counts": True, "final_state": False}
    first = backend.run(
        program, shots=40, simulation_config={"seed": 91}, result_config=config
    ).result()
    second = backend.run(
        program, shots=40, simulation_config={"seed": 91}, result_config=config
    ).result()
    assert first.get_counts() == second.get_counts()
    assert set(first.get_counts()) == {"0", "1"}


def test_atom_3level_execution_failure_is_an_eager_failed_job_without_solver_surface(
    atom_3level_model, atom_3level_calibration, monkeypatch
):
    backend = _backend(atom_3level_model, atom_3level_calibration)

    def fail(*_args, **_kwargs):
        raise RuntimeError("private adapter failure")

    monkeypatch.setattr(backend, "_create_runner", fail)
    job = backend.run(fq.Program(2))
    with pytest.raises(
        BackendExecutionError, match="Pulse backend execution failed"
    ) as excinfo:
        job.result()
    assert not any(
        token in str(excinfo.value).lower() for token in ("adapter", "solver", "qutip")
    )


def test_atom_3level_result_metadata_keeps_common_runtime_facts(
    atom_3level_model, atom_3level_calibration
):
    backend = _backend(atom_3level_model, atom_3level_calibration)
    result = backend.run(
        fq.Program(2), result_config={"counts": False, "final_state": True}
    ).result()
    assert result.metadata["backend_name"] == "Atom3LevelEmulator"
    assert result.metadata["simulation_config"]["schedule_mode"] == "ASAP"
    assert result.metadata["result_config"] == {
        "counts": False,
        "final_state": True,
    }

    def check_public(value, path=()):
        if isinstance(value, RegisterRef):
            assert path == ("state_axes", "register_ref")
            return
        assert isinstance(value, (str, int, float, bool, type(None), tuple, list, dict))
        if isinstance(value, dict):
            for key, child in value.items():
                assert isinstance(key, str)
                assert not any(
                    token in key.lower()
                    for token in ("coordinate", "handle", "runtime", "engine")
                )
                check_public(child, (*path, key))
        elif isinstance(value, (tuple, list)):
            for child in value:
                check_public(child, path)

    check_public(result.metadata)


def test_atom_3level_load_atoms_is_ordinary_unsupported_operation(
    atom_3level_model, atom_3level_calibration
):
    backend = _backend(atom_3level_model, atom_3level_calibration)
    program = fq.Program(2)
    program.add(ops.Put, (0, 1))
    with pytest.raises(UnsupportedOperationError):
        backend.run(program)


def test_weak_blockade_neither_rejects_nor_emits_the_removed_advisory(
    atom_3level_model, atom_3level_calibration, atom_3level_model_document
):
    pair = fq.Program(2)
    pair.add(ops.CZ, (0, 1))
    weak = _backend(atom_3level_model, atom_3level_calibration, spacing=20.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _backend(
            atom_3level_model,
            atom_3level_calibration,
            spacing=20.0,
            method="unitary",
        ).run(pair).result().get_unitary()
        weak.run(pair, shots=1)
    assert not any("V/Omega" in str(warning.message) for warning in caught)

    negative_document = deepcopy(atom_3level_model_document)
    negative_document["parameters"]["c6"] = -atom_3level_model.c6_angular_per_us_um6
    negative_model = fq.emulator.Atom3LevelModel.from_document(negative_document)
    negative = _backend(negative_model, atom_3level_calibration, spacing=20.0)
    _backend(
        negative_model,
        atom_3level_calibration,
        spacing=20.0,
        method="unitary",
    ).run(pair).result().get_unitary()
    assert negative._target.interactions[0].signed_strength_rad_per_us < 0
    prepared = negative._prepare_program(pair)
    runner = negative._create_runner(
        prepared,
        execution_mode="density_matrix",
        retain_final_state=True,
    )
    assert runner.interaction_drift().full()[8, 8].real < 0
