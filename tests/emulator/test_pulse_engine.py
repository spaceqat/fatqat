"""Whole-plan engine ownership and boundary tests with a fake runner."""

import json
from pathlib import Path

import numpy as np

from fatqat.backends import MeasurementStep, ResetStep
from fatqat.emulator.engine import PulseEngine
from fatqat.emulator.resolved import (
    PhaseShift,
    PulseBlock,
    SampledControl,
)
from fatqat.emulator.superconducting import load_physics_model

_FIXTURES = Path(__file__).parent / "fixtures"


def _model():
    return load_physics_model(
        json.loads((_FIXTURES / "sc_transmon_exchange.json").read_text())
    )


def _physical_block(
    model,
    duration,
    *,
    subsystem_id="q0",
    condition=None,
    post_actions=(),
    target_indices=None,
):
    return PulseBlock(
        model,
        duration,
        (
            SampledControl(
                model.drive_control(subsystem_id),
                [0.0, duration],
                [0.0, 0.0],
            ),
        ),
        (model.resource(subsystem_id),),
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


def test_engine_owns_boundaries_guard_reservation_and_persistent_shot_context():
    model = _model()
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


def test_engine_replays_each_shot_with_distinct_state_and_shared_rng_stream():
    model = _model()
    runner = _FakeRunner()
    outcomes = PulseEngine(runner).run(
        (_physical_block(model, 1.0),),
        shots=2,
        n_clbits=0,
        rng=np.random.default_rng(7),
    )
    assert outcomes[0] == outcomes[1]
    assert len(set(runner.context_ids)) == 2


def test_engine_evolves_static_terminal_measurement_plan_once_then_samples():
    model = _model()
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


def test_engine_keeps_measurement_followed_by_pulse_dynamic():
    model = _model()
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


def test_engine_places_conditioned_and_independent_blocks_together():
    model = _model()
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


def test_engine_keeps_disjoint_post_measurement_pulse_on_fast_path():
    model = _model()
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
