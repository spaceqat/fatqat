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


def _physical_block(model, duration, *, condition=None, post_actions=()):
    return PulseBlock(
        model,
        duration,
        (
            SampledControl(
                model.drive_control("q0"),
                [0.0, duration],
                [0.0, 0.0],
            ),
        ),
        (model.resource("q0"),),
        post_actions=post_actions,
        condition=condition,
    )


class _FakeRunner:
    def __init__(self):
        self.context_ids = []
        self.runs = []
        self.boundaries = []

    def initial_state(self):
        return {"runs": 0}

    def evolve(self, run, context, enabled):
        self.context_ids.append(id(context))
        self.runs.append((run.starts_ns, run.end_ns, enabled))
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
        self.boundaries.append((type(step), context.time_ns))

    def finish_shot(self, context):
        self.context_ids.append(id(context))
        return (
            context.time_ns,
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
    outcomes = PulseEngine(runner).execute(
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
    outcomes = PulseEngine(runner).execute(
        (_physical_block(model, 1.0),),
        shots=2,
        n_clbits=0,
        rng=np.random.default_rng(7),
    )
    assert outcomes[0] == outcomes[1]
    assert len(set(runner.context_ids)) == 2
