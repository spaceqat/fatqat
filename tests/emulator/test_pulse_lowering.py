"""Unplaced pulse lowering and shared-boundary preservation."""

import json
from pathlib import Path

import numpy as np
import pytest

import fatqat as fq
from fatqat.backends import MeasurementStep, ResetStep
from fatqat.backends.backend_utils import _LoweringContext
from fatqat.emulator.backend import PulseBackend
from fatqat.emulator.resolved import PulseBlock
from fatqat.emulator.superconducting import (
    load_calibration_spec,
    load_physics_model,
)
from fatqat.errors import BackendValidationError
from fatqat.noise import NoiseModel

_FIXTURES = Path(__file__).parent / "fixtures"


def _backend(noise=None):
    model = load_physics_model(
        json.loads((_FIXTURES / "sc_transmon_exchange.json").read_text())
    )
    calibration = load_calibration_spec(
        json.loads((_FIXTURES / "sc_transmon_exchange_calibration.json").read_text()),
        model,
    )
    return PulseBackend(model, calibration, noise=noise)


def test_lowering_produces_unplaced_blocks_and_preserves_boundaries_and_guards():
    backend = _backend()
    program = fq.Program(2, 1)
    program.add(fq.ops.RX(0.4), 0)
    program.measure(0, 0)
    program.add(fq.ops.RZ(0.2), 1, condition=(0, 0))
    program.add(fq.ops.Reset, 1, condition=(0, 0))
    plan, facts = backend._lower_program(program)

    assert [type(step) for step in plan] == [
        PulseBlock,
        MeasurementStep,
        PulseBlock,
        ResetStep,
    ]
    assert plan[0].start_time is None
    assert plan[1].reported_digit_maps == ((0, 1, 1),)
    assert plan[2].condition == ((0, 0),)
    assert plan[3].condition == ((0, 0),)
    assert facts.has_measurement


def test_lowering_rejects_absent_edges_and_reversed_cz_orientation():
    backend = _backend()
    disconnected_document = json.loads(
        (_FIXTURES / "sc_transmon_exchange.json").read_text()
    )
    disconnected_document["parameters"]["couplings"] = []
    disconnected = load_physics_model(disconnected_document)
    calibration_document = json.loads(
        (_FIXTURES / "sc_transmon_exchange_calibration.json").read_text()
    )
    calibration_document["recipes"]["cz"]["edges"] = []
    disconnected_backend = PulseBackend(
        disconnected, load_calibration_spec(calibration_document, disconnected)
    )
    iswap = fq.Program(2)
    iswap.add(fq.ops.iSwap, (0, 1))
    with pytest.raises(BackendValidationError, match="no declared coupling"):
        disconnected_backend.run(iswap)

    reversed_cz = fq.Program(2)
    reversed_cz.add(fq.ops.CZ, (1, 0))
    with pytest.raises(BackendValidationError, match="orientation"):
        backend.run(reversed_cz)


# --- private lowering context: resource layout and engine allocation must
# travel together, exactly as the matrix family already requires -----------


def test_lower_program_no_longer_accepts_a_half_specified_context():
    backend = _backend()
    program = fq.Program(1)
    program.add(fq.ops.RZ(0.2), 0)
    layout = backend._resolve_resource_layout(program)
    allocation = backend._allocate_engine_indices(program)

    # The old two-independent-optionals signature accepted either half
    # alone; the seam now only accepts a single paired `_LoweringContext`,
    # so passing either half by its old keyword is a TypeError, not a
    # silently-accepted partial context.
    with pytest.raises(TypeError):
        backend._lower_program(program, resource_layout=layout)
    with pytest.raises(TypeError):
        backend._lower_program(program, engine_index_allocation=allocation)

    # The paired form still works and is equivalent to the omitted-context
    # (resolve-both-here) default.
    context = _LoweringContext(
        resource_layout=layout, engine_index_allocation=allocation
    )
    plan, facts = backend._lower_program(program, context=context)
    default_plan, default_facts = backend._lower_program(program)
    assert [type(step) for step in plan] == [type(step) for step in default_plan]
    assert facts == default_facts


# --- shared measurement-lowering boundary: confusion validation parity -----


def test_pulse_measurement_confusion_must_match_the_reported_bit_dimension():
    noise = NoiseModel()
    noise.add_readout_error(np.eye(3), target="q0")
    backend = _backend(noise)
    program = fq.Program(1, 1)
    program.measure(0, 0)

    # Routed through the shared boundary helper (backend_utils._resolve_confusions):
    # pulse's literal (0, 1, 1) reported-digit map implies reported dimension
    # 2, so a 3x3 confusion is rejected with the same "reported classical
    # dimension" message the matrix family raises for an analogous mismatch
    # (see tests/backend/test_readout_error.py::test_dimension_mismatch_rejected_at_lowering).
    with pytest.raises(BackendValidationError, match="reported classical dimension"):
        backend._lower_program(program)


def test_pulse_measurement_accepts_a_correctly_shaped_confusion_matrix():
    always_flip = np.array([[0.0, 1.0], [1.0, 0.0]])
    noise = NoiseModel()
    noise.add_readout_error(always_flip, target="q0")
    backend = _backend(noise)
    program = fq.Program(1, 1)
    program.measure(0, 0)

    plan, _facts = backend._lower_program(program)
    (measurement,) = [s for s in plan if isinstance(s, MeasurementStep)]
    # Pulse always stores its literal qutrit-to-bit map, unlike the matrix
    # family's `None` identity default, whether or not confusion is present.
    assert measurement.reported_digit_maps == ((0, 1, 1),)
    assert np.array_equal(measurement.confusions[0], always_flip)
