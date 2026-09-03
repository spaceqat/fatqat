"""Whole-plan engine ownership and boundary tests with a fake runner."""

import numpy as np
import pytest

from fatqat._backends.steps import MeasurementStep, ResetStep
from fatqat._pulse_values import PulseControl
from fatqat.emulator._core.engine import (
    PulseEngine,
    _TerminalTrajectoryBatchRunner,
)
from fatqat.emulator._core.pulse import (
    PhaseShift,
    PulseBlock,
)
from fatqat.emulator._core.scheduling import schedule_pulse_run
from fatqat.emulator._core.target import _PreparedControlBinding, _TargetClaim
from fatqat.errors import BackendValidationError
from fatqat.emulator import SampledWaveform

_OWNER = object()
_CLAIMS = {
    "q0": _TargetClaim(_OWNER, "subsystem", 0),
    "q1": _TargetClaim(_OWNER, "subsystem", 1),
}


def _physical_block(
    model,
    duration,
    *,
    subsystem_id="q0",
    condition=None,
    post_actions=(),
    target_indices=None,
):
    controls = (
        PulseControl(
            model.control.drive(subsystem_id),
            SampledWaveform((0.0, duration), (0.0, 0.0)),
        ),
    )
    claim = _CLAIMS[subsystem_id]
    binding = _PreparedControlBinding("drive", (0 if subsystem_id == "q0" else 1,))
    return PulseBlock(
        duration,
        controls,
        (binding,),
        (claim,),
        post_actions=post_actions,
        condition=condition,
        target_indices=target_indices,
    )


class _FakeRunner:
    def __init__(self):
        self.context_ids = []
        self.runs = []
        self.boundaries = []

    def initial_state(self):
        return {"runs": 0}

    @staticmethod
    def copy_state(state):
        return dict(state)

    def evolve(self, run, context, enabled):
        self.context_ids.append(id(context))
        self.runs.append((run.starts, run.end_time, enabled))
        context.state["runs"] += 1
        for block, active in zip(run.blocks, enabled):
            if not active:
                continue
            for action in block.post_actions:
                if isinstance(action, PhaseShift):
                    context.frame_angles[action.frame] = (
                        context.frame_angles.get(action.frame, 0.0) + action.angle_rad
                    )

    def propagator(self, run, *, apply_final_frame=True):
        raise AssertionError("the execution fake does not construct propagators")

    def execute_boundary(self, step, context):
        self.context_ids.append(id(context))
        self.boundaries.append((type(step), context.time))

    def finish_shot(self, context):
        self.context_ids.append(id(context))
        return (
            context.time,
            dict(context.frame_angles),
            tuple(context.classical_memory),
            context.state["runs"],
        )

    @staticmethod
    def runtime_details():
        return {"mode": "fake"}


def test_engine_owns_boundaries_guard_reservation_and_persistent_shot_context(model):
    frame = model.frame("q0")
    first = _physical_block(model, 1.0, post_actions=(PhaseShift(frame, 0.25),))
    guarded = _physical_block(
        model,
        2.0,
        condition=((0, 1),),
        post_actions=(PhaseShift(frame, 9.0),),
    )
    last = _physical_block(model, 1.0)
    plan = (
        first,
        MeasurementStep((0,), (0,), reported_digit_maps=((0, 1, 1),)),
        guarded,
        ResetStep((0,)),
        last,
    )
    runner = _FakeRunner()
    outcomes = PulseEngine(runner).run(
        plan, shots=1, n_clbits=1, rng=np.random.default_rng(4)
    )

    assert runner.runs == [
        ((0.0,), 1.0, (True,)),
        ((1.0,), 3.0, (False,)),
        ((3.0,), 4.0, (True,)),
    ]
    assert runner.boundaries == [(MeasurementStep, 1.0), (ResetStep, 3.0)]
    assert len(set(runner.context_ids)) == 1
    assert outcomes == ((4.0, {frame: 0.25}, (0,), 3),)


def test_engine_replays_each_shot_with_distinct_state_and_shared_rng_stream(model):
    runner = _FakeRunner()
    outcomes = PulseEngine(runner).run(
        (_physical_block(model, 1.0),),
        shots=2,
        n_clbits=0,
        rng=np.random.default_rng(7),
    )
    assert outcomes[0] == outcomes[1]
    assert len(set(runner.context_ids)) == 2


def test_engine_evolves_static_terminal_measurement_plan_once_then_samples(model):
    runner = _FakeRunner()
    measurement = MeasurementStep((0,), (0,))
    outcomes = PulseEngine(runner).run(
        (_physical_block(model, 1.0), measurement),
        shots=3,
        n_clbits=1,
        rng=np.random.default_rng(7),
    )

    assert runner.runs == [((0.0,), 1.0, (True,))]
    assert runner.boundaries == [(MeasurementStep, 1.0)] * 3
    assert [outcome[-2] for outcome in outcomes] == [(0,), (0,), (0,)]


def test_engine_reuses_one_evolution_for_independent_measurement_groups(model):
    runner = _FakeRunner()
    measurements = (
        MeasurementStep((0,), (0,)),
        MeasurementStep((1,), (1,)),
    )
    outcome_groups = PulseEngine(runner).run_terminal_measurement_groups(
        (_physical_block(model, 1.0),),
        measurements,
        shots=2,
        n_clbits=2,
        rng=np.random.default_rng(3),
        measurement_rngs=(np.random.default_rng(5), np.random.default_rng(7)),
    )

    assert runner.runs == [((0.0,), 1.0, (True,))]
    assert runner.boundaries == [(MeasurementStep, 1.0)] * 4
    assert [[outcome[-1] for outcome in group] for group in outcome_groups] == [
        [1, 1],
        [1, 1],
    ]


def test_engine_keeps_measurement_followed_by_pulse_dynamic(model):
    runner = _FakeRunner()
    measurement = MeasurementStep((0,), (0,))
    PulseEngine(runner).run(
        (measurement, _physical_block(model, 1.0)),
        shots=2,
        n_clbits=1,
        rng=np.random.default_rng(7),
    )

    assert runner.runs == [
        ((0.0,), 1.0, (True,)),
        ((0.0,), 1.0, (True,)),
    ]
    assert runner.boundaries == [(MeasurementStep, 0.0)] * 2


def test_engine_places_conditioned_and_independent_blocks_together(model):
    runner = _FakeRunner()
    measurement = MeasurementStep((0,), (0,))
    PulseEngine(runner).run(
        (
            measurement,
            _physical_block(model, 1.0, condition=((0, 1),)),
            _physical_block(model, 1.0, subsystem_id="q1"),
        ),
        shots=1,
        n_clbits=1,
        rng=np.random.default_rng(7),
    )

    # The measurement is the only boundary.  The false guarded q0 block still
    # reserves its q0 slot, while the independent q1 block shares the run.
    assert runner.runs == [((0.0, 0.0), 1.0, (False, True))]
    assert runner.boundaries == [(MeasurementStep, 0.0)]


def test_engine_keeps_disjoint_post_measurement_pulse_on_fast_path(model):
    runner = _FakeRunner()
    measurement = MeasurementStep((0,), (0,))
    PulseEngine(runner).run(
        (
            measurement,
            _physical_block(model, 1.0, target_indices=(1,)),
        ),
        shots=2,
        n_clbits=1,
        rng=np.random.default_rng(7),
    )

    assert runner.runs == [((0.0,), 1.0, (True,))]
    assert runner.boundaries == [(MeasurementStep, 1.0)] * 2


class _TrajectoryRunner(_FakeRunner):
    def __init__(self):
        super().__init__()
        self.batch_calls = []
        self.finished = []
        self.rng_ids = []
        self.finished_contexts = []

    def run_trajectory_batch(self, scheduled_run, *, ntraj, seeds):
        self.batch_calls.append((scheduled_run, ntraj, seeds))
        return tuple({"seed": seed} for seed in seeds)

    def execute_boundary(self, step, context):
        super().execute_boundary(step, context)
        self.rng_ids.append(id(context.rng))
        context.classical_memory[0] = context.state["seed"] % 2

    def finish_shot(self, context):
        self.context_ids.append(id(context))
        self.finished_contexts.append(context)
        result = (
            context.state["seed"],
            tuple(context.classical_memory),
            context.time,
        )
        self.finished.append(result)
        return result


class _RegionalTrajectoryRunner(_TrajectoryRunner):
    """Expose shot context continuity through runner payloads."""

    def initial_state(self):
        return {"regions": []}

    def evolve(self, run, context, enabled):
        context.state["regions"].append(
            (
                context.time,
                run.end_time,
                tuple(context.classical_memory),
                enabled,
                int(context.rng.integers(0, 10_000)),
            )
        )

    def execute_boundary(self, step, context):
        context.classical_memory[0] = int(context.rng.integers(0, 2))

    def finish_shot(self, context):
        return (
            tuple(context.state["regions"]),
            tuple(context.classical_memory),
            context.time,
        )


def test_trajectory_execution_preserves_context_across_regions(model):
    plan = (
        _physical_block(model, 1.0),
        MeasurementStep((0,), (0,)),
        _physical_block(model, 2.0, condition=((0, 1),)),
    )

    def execute():
        return PulseEngine(_RegionalTrajectoryRunner()).run_trajectories(
            plan,
            shots=2,
            n_clbits=1,
            rng=np.random.default_rng(20260830),
        )

    first = execute()
    second = execute()

    assert first == second
    assert first[0] != first[1]
    for regions, memory, final_time in first:
        assert len(regions) == 2
        assert regions[0][:4] == (0.0, 1.0, (0,), (True,))
        assert regions[1][:3] == (1.0, 3.0, memory)
        expected_enabled = (memory[0] == 1,)
        assert regions[1][3] == expected_enabled
        assert final_time == 3.0


def test_trajectory_execution_preserves_batched_seed_and_outcome_order(model):
    blocks = (
        _physical_block(model, 1.0),
        _physical_block(model, 2.0, subsystem_id="q1"),
    )
    measurement = MeasurementStep((0,), (0,), reported_digit_maps=((0, 1, 1),))
    runner = _TrajectoryRunner()
    rng = np.random.default_rng(17)

    outcomes = PulseEngine(runner, schedule_mode="ALAP").run_trajectories(
        (*blocks, measurement), shots=3, n_clbits=1, rng=rng
    )

    assert len(runner.batch_calls) == 1
    scheduled, ntraj, seeds = runner.batch_calls[0]
    assert scheduled == schedule_pulse_run(blocks, boundary_time=0.0, mode="ALAP")
    assert ntraj == 3
    assert len(seeds) == 3
    assert len({id(context) for context in runner.finished_contexts}) == 3
    assert len(set(runner.rng_ids)) == 1
    assert runner.boundaries == [(MeasurementStep, scheduled.end_time)] * 3
    assert outcomes == tuple((seed, (seed % 2,), scheduled.end_time) for seed in seeds)
    assert runner.finished == list(outcomes)

    repeated = _TrajectoryRunner()
    PulseEngine(repeated, schedule_mode="ALAP").run_trajectories(
        (*blocks, measurement),
        shots=3,
        n_clbits=1,
        rng=np.random.default_rng(17),
    )
    assert repeated.batch_calls[0][2] == seeds


@pytest.mark.parametrize("shots", [0, -1, True, 1.5])
def test_trajectory_execution_rejects_invalid_shots(model, shots):
    with pytest.raises(BackendValidationError, match="shots"):
        PulseEngine(_TrajectoryRunner()).run_trajectories(
            (_physical_block(model, 1.0),),
            shots=shots,
            n_clbits=0,
            rng=np.random.default_rng(1),
        )


@pytest.mark.parametrize("n_clbits", [-1, True, 1.5])
def test_trajectory_execution_rejects_invalid_classical_width(model, n_clbits):
    with pytest.raises(BackendValidationError, match="classical width"):
        PulseEngine(_TrajectoryRunner()).run_trajectories(
            (_physical_block(model, 1.0),),
            shots=1,
            n_clbits=n_clbits,
            rng=np.random.default_rng(1),
        )


def test_trajectory_execution_rejects_unknown_plan_steps():
    with pytest.raises(BackendValidationError, match="unknown"):
        PulseEngine(_TrajectoryRunner()).run_trajectories(
            (object(),), shots=1, n_clbits=0, rng=np.random.default_rng(1)
        )


def test_runtime_protocol_guard_checks_presence_not_signature(model):
    class WrongSignature(_TrajectoryRunner):
        # pylint: disable-next=arguments-differ
        def run_trajectory_batch(self, scheduled_run):
            return (scheduled_run,)

    runner = WrongSignature()
    assert isinstance(runner, _TerminalTrajectoryBatchRunner)
    with pytest.raises(TypeError):
        PulseEngine(runner).run_trajectories(
            (_physical_block(model, 1.0),),
            shots=1,
            n_clbits=0,
            rng=np.random.default_rng(1),
        )


@pytest.mark.parametrize("returned", [[], ({"seed": 1}, {"seed": 2})])
def test_trajectory_execution_validates_batched_return_shape(model, returned):
    runner = _TrajectoryRunner()
    runner.run_trajectory_batch = lambda *_args, **_kwargs: returned
    with pytest.raises(BackendValidationError, match="return"):
        PulseEngine(runner).run_trajectories(
            (_physical_block(model, 1.0),),
            shots=1,
            n_clbits=0,
            rng=np.random.default_rng(1),
        )
