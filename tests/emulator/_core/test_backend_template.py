"""The shared pulse backend prepares every run fact exactly once."""

from dataclasses import FrozenInstanceError
import inspect

import numpy as np
import pytest

import fatqat as fq
from fatqat._pulse_values import PulseControl
from fatqat.emulator._core.backend import _PulseBackend
from fatqat.emulator._core.engine import PulseEngine
from fatqat.emulator._core.lindblad import (
    ResolvedLindbladTerm,
    _classify_lindblad_noise,
)
from fatqat.emulator._core.outcome import (
    _PulseExecutionSummary,
    _PulseShotOutcome,
)
from fatqat.emulator._core.planning import _PreparedPulseProgram
from fatqat.emulator._core.pulse import (
    PhaseShift,
    PulseDefinition,
    PulseImplementationMap,
)
from fatqat.emulator._core.target import (
    _ControlAddress,
    _ControlBinding,
    _FrameAddress,
    _GateBinding,
    _TargetClaim,
)
from fatqat.errors import BackendExecutionError, BackendValidationError
from fatqat.job import Job
from fatqat.noise import (
    AmplitudeDamping,
    LindbladImplementationMap,
    NoiseModel,
    PauliChannel,
)
from fatqat.noise.lindblad import amplitude_damping_lindblad_rule
from fatqat.resource_layout import ResourceLayout
from fatqat.waveforms import SampledWaveform


class _CountingNoiseModel(NoiseModel):
    def __init__(self):
        super().__init__()
        self.selector_validations = 0
        self.operation_selections = 0
        self.background_selections = 0

    def _validate_for(self, program, legal_device_operands):
        self.selector_validations += 1
        return super()._validate_for(program, legal_device_operands)

    def _noise_for_occurrence(self, operation, targets, resource_layout):
        self.operation_selections += 1
        return super()._noise_for_occurrence(operation, targets, resource_layout)

    def _background_noise_for(self, target, device_label):
        self.background_selections += 1
        return super()._background_noise_for(target, device_label)

    def _copy(self):
        copied = type(self)()
        copied._noise_registrations = self._noise_registrations.copy()
        copied._readout_registrations = self._readout_registrations.copy()
        return copied


class _NoMatchNoiseModel(_CountingNoiseModel):
    def _background_noise_for(self, target, device_label):
        del target, device_label
        self.background_selections += 1
        return ()


class _FakeModel:
    @staticmethod
    def drive(label):
        return _ControlAddress("fake", "drive", (label,))

    @staticmethod
    def frame(label):
        return _FrameAddress("fake", (label,))


class _CountingTarget:
    local_dimension = 2

    def __init__(self, device_labels=("q0", "q1", "q2")):
        self.model = _FakeModel()
        self.device_labels = tuple(device_labels)
        self.hilbert_dimension = 2 ** len(self.device_labels)
        owner = object()
        self._claims = tuple(
            _TargetClaim(owner, "site", ordinal)
            for ordinal in range(len(self.device_labels))
        )
        self.bind_program_calls = 0
        self.bind_control_calls = 0
        self.bind_frame_calls = 0
        self.bind_gate_operands_calls = 0
        self.bound_controls = []
        self.validate_control_calls = 0

    def bind_program(self, program):
        self.bind_program_calls += 1
        refs = tuple(
            register[index]
            for register in program.quantum_registers
            for index in range(register.size)
        )
        if len(refs) > len(self.device_labels):
            raise BackendValidationError("program exceeds fake target")
        return ResourceLayout(
            {ref: self.device_labels[ordinal] for ordinal, ref in enumerate(refs)}
        )

    def bind_control(self, reference):
        self.bind_control_calls += 1
        if not isinstance(reference, _ControlAddress) or reference.family != "fake":
            raise BackendValidationError("foreign fake control")
        try:
            ordinal = self.device_labels.index(reference.operands[0])
        except (ValueError, IndexError):
            raise BackendValidationError("unknown fake target") from None
        binding = _ControlBinding(
            reference.kind,
            (self.device_labels[ordinal],),
            (self._claims[ordinal],),
        )
        self.bound_controls.append(binding)
        return binding

    def bind_frame(self, reference):
        self.bind_frame_calls += 1
        if not isinstance(reference, _FrameAddress) or reference.family != "fake":
            raise BackendValidationError("foreign fake frame")
        ordinal = self.device_labels.index(reference.operands[0])
        return _GateBinding(
            (self._claims[ordinal],),
            (self.device_labels[ordinal],),
        )

    def validate_pulse_controls(self, controls, bindings, block_duration):
        del block_duration
        self.validate_control_calls += 1
        assert len(controls) == len(bindings)

    def reported_digit_map(self, device_label):
        self.device_labels.index(device_label)
        return (0, 1)

    def bind_gate_operands(self, device_operands):
        self.bind_gate_operands_calls += 1
        ordinals = tuple(self.device_labels.index(value) for value in device_operands)
        return _GateBinding(
            tuple(self._claims[ordinal] for ordinal in ordinals),
            tuple(device_operands),
        )


class _ArrayOperator:
    def __init__(self, value):
        self._value = np.asarray(value, dtype=complex)

    def full(self):
        return self._value


class _TemplateRunner:
    def __init__(self, target, execution_mode, retain_final_state):
        self._target = target
        self._execution_mode = execution_mode
        self._retain_final_state = retain_final_state
        self.solver_metadata_calls = 0
        self.propagator_calls = 0

    def initial_state(self):
        if self._execution_mode == "density_matrix":
            value = np.zeros(
                (self._target.hilbert_dimension,) * 2,
                dtype=complex,
            )
            value[0, 0] = 1.0
            return value
        value = np.zeros(self._target.hilbert_dimension, dtype=complex)
        value[0] = 1.0
        return value

    @staticmethod
    def copy_state(state):
        return np.array(state, copy=True)

    @staticmethod
    def evolve(run, context, enabled):
        del run, context, enabled

    def propagator(self, run, *, apply_final_frame=True):
        del run, apply_final_frame
        self.propagator_calls += 1
        return _ArrayOperator(np.eye(self._target.hilbert_dimension))

    @staticmethod
    def execute_boundary(step, context):
        if hasattr(step, "classical_indices"):
            for index in step.classical_indices:
                context.classical_memory[index] = 0

    def finish_shot(self, context):
        if self._execution_mode == "density_matrix":
            kind = "density_matrix"
        elif self._execution_mode == "trajectory":
            kind = "statevector"
        else:
            kind = "statevector"
        return _PulseShotOutcome(
            np.array(context.state, copy=True) if self._retain_final_state else None,
            kind,
            tuple(context.classical_memory),
        )

    def solver_metadata(self):
        self.solver_metadata_calls += 1
        return {"mode": self._execution_mode}


class _TemplateBackend(_PulseBackend):
    _coherent_execution_mode = "density_matrix"

    def __init__(self, target, *, noise=None, gate_map=None, lindblad_map=None):
        self.source_validations = 0
        self.classifications = 0
        self.runner_calls = 0
        self.execution_mode_calls = 0
        self.last_runner = None
        self.runner_prepared = None
        super().__init__(
            target.model,
            noise=noise,
            gate_implementation_map=(
                PulseImplementationMap() if gate_map is None else gate_map
            ),
            lindblad_implementation_map=(
                LindbladImplementationMap() if lindblad_map is None else lindblad_map
            ),
        )
        self._set_target(target)

    def _validate_source_program(self, program):
        self.source_validations += 1
        assert isinstance(program, fq.Program)

    def _classify_noise(self, noise_model):
        self.classifications += 1
        return _classify_lindblad_noise(
            noise_model,
            self._lindblad_implementation_map,
            local_dimension=self._target.local_dimension,
            backend_name=type(self).__name__,
            supports_readout_confusion=False,
        )

    def _resolve_execution_mode(self, facts):
        self.execution_mode_calls += 1
        del facts
        return "density_matrix"

    def _create_runner(
        self,
        prepared,
        *,
        execution_mode,
        retain_final_state,
    ):
        self.runner_calls += 1
        self.runner_prepared = prepared
        self.last_runner = _TemplateRunner(
            self._target,
            execution_mode,
            retain_final_state,
        )
        return self.last_runner


def _lindblad_map():
    implementation_map = LindbladImplementationMap()
    implementation_map.register(
        AmplitudeDamping,
        amplitude_damping_lindblad_rule,
    )
    return implementation_map


def _gate_map(target, *, with_frame=False):
    implementation_map = PulseImplementationMap()

    def realize(_operation, *, device_operands):
        channel = target.model.drive(device_operands[0])
        return PulseDefinition(
            0.5,
            (
                PulseControl(
                    channel,
                    SampledWaveform((0.0, 0.5), (0.2, 0.2)),
                ),
            ),
            post_actions=(
                (PhaseShift(target.model.frame(device_operands[0]), 0.25),)
                if with_frame
                else ()
            ),
        )

    implementation_map.add(fq.ops.X, realize)
    implementation_map.add(fq.ops.Y, realize)
    return implementation_map


def test_preparation_builds_one_complete_immutable_value_exactly_once():
    target = _CountingTarget()
    noise = _CountingNoiseModel()
    noise.add(
        AmplitudeDamping(rate=0.2),
        operation=fq.ops.X,
    )
    for label in target.device_labels:
        noise.add(AmplitudeDamping(rate=0.1), targets=label)
    backend = _TemplateBackend(
        target,
        noise=noise,
        gate_map=_gate_map(target),
        lindblad_map=_lindblad_map(),
    )
    program = fq.Program(2)
    program.add(fq.ops.X, 0)

    prepared = backend._prepare_program(program)

    assert isinstance(prepared, _PreparedPulseProgram)
    assert prepared.plan[0].control_bindings[0].engine_indices == (0,)
    assert prepared.engine_allocation.device_operands == ("q0", "q1", "q2")
    refs = tuple(
        register[index]
        for register in program.quantum_registers
        for index in range(register.size)
    )
    assert prepared.resource_layout.device_labels_for(refs) == ("q0", "q1")
    assert len(prepared.plan[0].noise) == 1
    assert len(prepared.background_noise) == 3
    assert prepared.facts.has_nonzero_evolution
    assert prepared.facts.has_resolved_lindblad
    assert prepared.facts.has_supported_background_lindblad_registration
    assert target.bind_program_calls == 1
    assert target.bind_control_calls == 1
    assert target.bind_gate_operands_calls == 1
    assert target.validate_control_calls == 1
    captured_noise = backend._noise_model
    assert noise.selector_validations == 0
    assert noise.operation_selections == 0
    assert noise.background_selections == 0
    assert captured_noise.selector_validations == 1
    assert captured_noise.operation_selections == 1
    assert captured_noise.background_selections == 3
    assert backend.source_validations == 1
    assert backend.classifications == 1
    with pytest.raises(FrozenInstanceError):
        prepared.plan = ()


def test_direct_control_stores_the_one_binding_and_translates_target_ordinal():
    target = _CountingTarget()
    backend = _TemplateBackend(target)
    control = PulseControl(
        target.model.drive("q1"),
        SampledWaveform((0.0, 0.5), (0.0, 0.2)),
    )
    operation = fq.ops.PulseOperation(0.5, (control,))
    program = fq.Program(2)
    program.add(operation)

    prepared = backend._prepare_program(program)
    block = prepared.plan[0]
    assert target.bind_control_calls == 1
    assert target.bound_controls[0].device_operands == ("q1",)
    assert block.control_bindings[0].kind == "drive"
    assert block.control_bindings[0].engine_indices == (1,)
    assert block.target_indices == (1,)


def test_direct_control_can_target_an_unreferenced_modeled_subsystem():
    target = _CountingTarget()
    backend = _TemplateBackend(target)
    operation = fq.ops.PulseOperation(
        0.5,
        (
            PulseControl(
                target.model.drive("q2"),
                SampledWaveform((0.0, 0.5), (0.0, 0.2)),
            ),
        ),
    )
    program = fq.Program(2)
    program.add(operation)
    prepared = backend._prepare_program(program)

    assert prepared.plan[0].control_bindings[0].engine_indices == (2,)


def test_background_targets_bind_declared_and_unreferenced_physical_targets():
    target = _CountingTarget()
    noise = _CountingNoiseModel()
    for label in target.device_labels:
        noise.add(AmplitudeDamping(rate=0.1), targets=label)
    backend = _TemplateBackend(
        target,
        noise=noise,
        lindblad_map=_lindblad_map(),
    )
    prepared = backend._prepare_program(fq.Program(1))
    assert tuple(term.engine_indices for term in prepared.background_noise) == (
        (0,),
        (1,),
        (2,),
    )


def test_resolved_terms_use_local_not_full_hilbert_dimension():
    target = _CountingTarget()
    noise = _CountingNoiseModel()
    noise.add(AmplitudeDamping(rate=0.1), targets="q0")
    backend = _TemplateBackend(
        target,
        noise=noise,
        lindblad_map=_lindblad_map(),
    )
    prepared = backend._prepare_program(fq.Program(1))
    assert all(
        term.local_operator.shape == (target.local_dimension,) * 2
        for term in prepared.background_noise
    )


def test_operation_scoped_probability_rejects_without_implicit_conversion():
    target = _CountingTarget()
    noise = _CountingNoiseModel()
    noise.add(
        AmplitudeDamping(p=0.2),
        operation=fq.ops.X,
    )
    backend = _TemplateBackend(
        target,
        noise=noise,
        gate_map=_gate_map(target),
        lindblad_map=_lindblad_map(),
    )
    program = fq.Program(1)
    program.add(fq.ops.X, 0)
    with pytest.raises(BackendValidationError, match="finite probability mode"):
        backend._prepare_program(program)


def test_operation_scoped_rate_keeps_target_binding_and_fact_scope_separate():
    target = _CountingTarget()
    noise = _CountingNoiseModel()
    noise.add(AmplitudeDamping(rate=0.2), operation=fq.ops.X)
    backend = _TemplateBackend(
        target,
        noise=noise,
        gate_map=_gate_map(target),
        lindblad_map=_lindblad_map(),
    )
    program = fq.Program(1)
    program.add(fq.ops.X, 0)

    prepared = backend._prepare_program(program)

    term = prepared.plan[0].noise[0]
    assert term.engine_indices == (0,)
    assert abs(term.local_operator[0, 1]) ** 2 == pytest.approx(0.2)
    assert prepared.facts.has_resolved_lindblad
    assert not prepared.facts.has_supported_background_lindblad_registration


def test_supported_noise_for_an_absent_operation_is_a_valid_no_op():
    target = _CountingTarget()
    noise = _CountingNoiseModel()
    noise.add(AmplitudeDamping(rate=0.2), operation=fq.ops.X)
    backend = _TemplateBackend(
        target,
        noise=noise,
        gate_map=_gate_map(target),
        lindblad_map=_lindblad_map(),
    )
    program = fq.Program(1)
    program.add(fq.ops.Y, 0)

    prepared = backend._prepare_program(program)

    assert prepared.plan[0].noise == ()
    assert not prepared.facts.has_resolved_lindblad


def test_supported_background_registration_can_set_capability_without_resolution():
    target = _CountingTarget()
    noise = _NoMatchNoiseModel()
    noise.add(AmplitudeDamping(rate=0.2), targets="q0")
    backend = _TemplateBackend(
        target,
        noise=noise,
        lindblad_map=_lindblad_map(),
    )

    prepared = backend._prepare_program(fq.Program(1))

    assert prepared.background_noise == ()
    assert not prepared.facts.has_resolved_lindblad
    assert prepared.facts.has_supported_background_lindblad_registration


def test_missing_or_empty_lindblad_implementation_rejects_explicitly():
    target = _CountingTarget()
    noise = _CountingNoiseModel()
    noise.add(AmplitudeDamping(rate=0.1), targets="q0")
    with pytest.raises(BackendValidationError, match="no registered"):
        _TemplateBackend(target, noise=noise)._prepare_program(fq.Program(1))

    empty_map = LindbladImplementationMap()
    empty_map.register(AmplitudeDamping, lambda channel, **kwargs: ())
    backend = _TemplateBackend(
        target,
        noise=noise,
        lindblad_map=empty_map,
    )
    with pytest.raises(BackendValidationError, match="no Lindblad operators"):
        backend._prepare_program(fq.Program(1))


def test_pauli_channel_policy_is_explicit_even_with_a_registered_rule():
    noise = NoiseModel()
    noise.add(PauliChannel({"X": 0.1}), operation=fq.ops.X)
    implementation_map = LindbladImplementationMap()
    implementation_map.register(
        PauliChannel,
        lambda channel, *, physical_dimension: (np.eye(physical_dimension),),
    )

    report = _classify_lindblad_noise(
        noise,
        implementation_map,
        local_dimension=2,
        backend_name="TemplateBackend",
        supports_readout_confusion=True,
    )

    assert not report.supported
    assert "pulse-family policy" in report.warnings[0]
    assert "registered Lindblad implementation" in report.warnings[0]


def test_invalid_local_operator_shape_rejects_at_shared_resolution_boundary():
    target = _CountingTarget()
    noise = _CountingNoiseModel()
    noise.add(AmplitudeDamping(rate=0.1), targets="q0")
    invalid_map = LindbladImplementationMap()
    invalid_map.register(
        AmplitudeDamping,
        lambda channel, **kwargs: (np.eye(3),),
    )
    backend = _TemplateBackend(
        target,
        noise=noise,
        lindblad_map=invalid_map,
    )
    with pytest.raises(BackendValidationError, match=r"expected \(2, 2\)"):
        backend._prepare_program(fq.Program(1))


@pytest.mark.parametrize("read_only_view", [False, True])
def test_resolved_lindblad_term_owns_its_operator(read_only_view):
    source = np.array([[0.0, 1.0], [0.0, 0.0]])
    supplied = source.view()
    if read_only_view:
        supplied.flags.writeable = False

    term = ResolvedLindbladTerm(supplied, (0,))
    source[0, 1] = 7.0

    assert term.local_operator[0, 1] == 1.0
    assert not term.local_operator.flags.writeable


def test_invalid_shots_raise_directly_after_preparation_without_a_runner():
    backend = _TemplateBackend(_CountingTarget())
    program = fq.Program(1, 1)
    program.measure(0, 0)
    with pytest.raises(BackendValidationError, match="shots"):
        backend.run(program, shots=0)
    assert backend.runner_calls == 0
    assert backend.classifications == 1


def test_constructor_copies_maps_and_captures_noise_registrations():
    target = _CountingTarget()
    noise = _CountingNoiseModel()
    gate_map = _gate_map(target)
    lindblad_map = _lindblad_map()
    original_lindblad_rule = lindblad_map.get(AmplitudeDamping)
    backend = _TemplateBackend(
        target,
        noise=noise,
        gate_map=gate_map,
        lindblad_map=lindblad_map,
    )

    gate_map.remove(fq.ops.X)
    lindblad_map.register(AmplitudeDamping, lambda channel, **kwargs: ())

    assert backend._noise_model is not noise
    assert backend._gate_implementation_map.supports(fq.ops.X)
    assert (
        backend._lindblad_implementation_map.get(AmplitudeDamping)
        is original_lindblad_rule
    )

    noise.add(AmplitudeDamping(rate=0.2), targets="q0")
    prepared = backend._prepare_program(fq.Program(1))
    assert prepared.background_noise == ()
    assert backend.validate_noise(noise).supported


def test_explicit_empty_maps_remain_empty_and_constructor_types_are_checked():
    target = _CountingTarget()
    backend = _TemplateBackend(
        target,
        gate_map=PulseImplementationMap(),
        lindblad_map=LindbladImplementationMap(),
    )
    assert not backend._gate_implementation_map.supported_operations()
    assert not backend._lindblad_implementation_map.supported_channels()

    with pytest.raises(BackendValidationError, match="noise must"):
        _PulseBackend.__init__(
            backend,
            target.model,
            noise=object(),
            gate_implementation_map=PulseImplementationMap(),
            lindblad_implementation_map=LindbladImplementationMap(),
        )
    with pytest.raises(BackendValidationError, match="gate_implementation_map"):
        _PulseBackend.__init__(
            backend,
            target.model,
            noise=None,
            gate_implementation_map=object(),
            lindblad_implementation_map=LindbladImplementationMap(),
        )
    with pytest.raises(BackendValidationError, match="lindblad_implementation_map"):
        _PulseBackend.__init__(
            backend,
            target.model,
            noise=None,
            gate_implementation_map=PulseImplementationMap(),
            lindblad_implementation_map=object(),
        )


def test_public_noise_validation_checks_type_then_calls_classifier_once():
    backend = _TemplateBackend(_CountingTarget())
    with pytest.raises(BackendValidationError, match="noise_model"):
        backend.validate_noise(object())
    assert backend.classifications == 0

    report = backend.validate_noise(NoiseModel())
    assert report.supported
    assert backend.classifications == 1


def test_shared_workflows_are_final_and_only_four_family_hooks_remain():
    expected_hooks = {
        "_validate_source_program",
        "_classify_noise",
        "_resolve_execution_mode",
        "_create_runner",
    }
    nonfinal_private_methods = {
        name
        for name, member in inspect.getmembers(_PulseBackend, inspect.isfunction)
        if name.startswith("_")
        and not name.startswith("__")
        and not getattr(member, "__final__", False)
    }
    assert nonfinal_private_methods == expected_hooks
    for name in (
        "run",
        "propagator",
        "_prepare_program",
        "_execute",
        "_assemble_result",
        "validate_noise",
    ):
        assert getattr(getattr(_PulseBackend, name), "__final__", False)
    assert not hasattr(_PulseBackend, "_create_runner_from_bindings")
    assert "_coherent_execution_mode" not in _PulseBackend.__dict__
    assert _TemplateBackend.__dict__["_coherent_execution_mode"] == "density_matrix"
    assert not {
        name
        for name in _TemplateBackend.__dict__
        if getattr(getattr(_PulseBackend, name, None), "__final__", False)
    }


def test_execution_summary_drives_common_metadata_once():
    backend = _TemplateBackend(_CountingTarget())

    result = backend.run(
        fq.Program(1),
        shots=7,
        result_config={"counts": False, "final_state": True},
    ).result()

    assert result.metadata["solver"] == {"mode": "density_matrix"}
    assert result.metadata["simulation_config"]["schedule_mode"] == "ASAP"
    assert result.metadata["result_config"] == {
        "counts": False,
        "final_state": True,
    }
    assert backend.last_runner.solver_metadata_calls == 1
    assert backend.execution_mode_calls == 1


def test_execution_summary_is_a_minimal_immutable_owned_handoff():
    outcome = _PulseShotOutcome(
        np.array([1.0, 0.0]),
        "statevector",
        (),
    )
    source_outcomes = [outcome]
    source_metadata = {"mode": "fake"}
    summary = _PulseExecutionSummary(
        source_outcomes,
        "statevector",
        source_metadata,
    )

    source_outcomes.clear()
    source_metadata["mode"] = "changed"

    assert summary.outcomes == (outcome,)
    assert summary.solver_metadata == {"mode": "fake"}
    with pytest.raises(TypeError):
        summary.solver_metadata["mode"] = "changed"
    with pytest.raises(FrozenInstanceError):
        summary.final_state_kind = "density_matrix"


def test_unexpected_execution_failure_returns_a_failed_eager_job(monkeypatch):
    backend = _TemplateBackend(_CountingTarget())

    def fail(*_args, **_kwargs):
        raise RuntimeError("solver failed")

    monkeypatch.setattr(PulseEngine, "run", fail)
    job = backend.run(fq.Program(1))

    assert isinstance(job, Job)
    with pytest.raises(BackendExecutionError) as exc:
        job.result()
    assert isinstance(exc.value.__cause__, RuntimeError)
    assert str(exc.value.__cause__) == "solver failed"


def _frame_gate_map(target):
    implementation_map = PulseImplementationMap()

    def frame(_operation, *, device_operands):
        return PulseDefinition(
            0.0,
            (),
            post_actions=(PhaseShift(target.model.frame(device_operands[0]), 0.25),),
        )

    implementation_map.add(fq.ops.RZ, frame)
    return implementation_map


def test_propagator_empty_and_frame_only_paths_use_fixed_coherent_mode():
    target = _CountingTarget(("q0",))
    noise = NoiseModel()
    noise.add(AmplitudeDamping(rate=0.2), targets="q0")
    backend = _TemplateBackend(
        target,
        noise=noise,
        gate_map=_frame_gate_map(target),
        lindblad_map=_lindblad_map(),
    )

    assert np.allclose(backend.propagator(fq.Program(1)), np.eye(2))
    assert backend.runner_calls == 0

    program = fq.Program(1)
    program.add(fq.ops.RZ(0.2), 0)
    assert np.allclose(backend.propagator(program), np.eye(2))
    assert backend.runner_calls == 1
    assert backend.last_runner.propagator_calls == 1
    assert backend.execution_mode_calls == 0


def _measurement_program():
    program = fq.Program(1, 1)
    program.measure(0, 0)
    return program


def _reset_program():
    program = fq.Program(1)
    program.add(fq.ops.Reset, 0)
    return program


def _conditioned_program():
    program = fq.Program(1, 1)
    program.add(fq.ops.X, 0, condition=(0, 1))
    return program


@pytest.mark.parametrize(
    ("build_program", "message"),
    (
        (_measurement_program, "measurement"),
        (_reset_program, "reset"),
        (_conditioned_program, "conditioned"),
    ),
)
def test_propagator_rejects_noncoherent_facts_before_runner(
    build_program,
    message,
):
    target = _CountingTarget(("q0",))
    backend = _TemplateBackend(target, gate_map=_gate_map(target))
    with pytest.raises(BackendValidationError, match=message):
        backend.propagator(build_program())
    assert backend.runner_calls == 0


def test_propagator_rejects_elapsed_resolved_noise_before_runner():
    target = _CountingTarget(("q0",))
    noise = NoiseModel()
    noise.add(AmplitudeDamping(rate=0.2), operation=fq.ops.X)
    backend = _TemplateBackend(
        target,
        noise=noise,
        gate_map=_gate_map(target),
        lindblad_map=_lindblad_map(),
    )
    program = fq.Program(1)
    program.add(fq.ops.X, 0)

    with pytest.raises(BackendValidationError, match="dissipative Lindblad"):
        backend.propagator(program)
    assert backend.runner_calls == 0
