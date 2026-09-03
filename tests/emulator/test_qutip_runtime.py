"""Cross-family behavioral checks for the shared QuTiP runtime policy."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.emulator._atom_3level import Atom3LevelEmulator


def _transmon_case(scale: float, _atom_model: Any) -> tuple[Any, Any]:
    model = fq.emulator.TransmonModel.from_document(
        {
            "format": {"id": "sc.transmon_exchange", "version": 1},
            "model": {"id": "scaled-time", "revision": "test"},
            "system": {
                "subsystem_type": "transmon",
                "subsystems": ["q0"],
                "control_edges": [],
            },
            "units": {"frequency": "GHz", "anharmonicity": "GHz"},
            "parameters": {
                "subsystems": {
                    "q0": {
                        "frequency": 5.0 / scale,
                        "anharmonicity": -0.22 / scale,
                    }
                }
            },
        }
    )
    return (
        fq.emulator.TransmonEmulator(model, method="density_matrix"),
        model.control.drive("q0"),
    )


def _atom2_case(_scale: float, _atom_model: Any) -> tuple[Any, Any]:
    model = fq.emulator.Atom2LevelModel.from_document(
        fq.emulator.load_model_document("atom2level.reference")
    )
    backend = fq.emulator.Atom2LevelEmulator(
        model,
        arrangement=fq.emulator.AtomArrangement.chain(1, spacing=1.0),
        method="density_matrix",
    )
    return backend, model.control.drive()


def _atom3_case(_scale: float, atom_model: Any) -> tuple[Any, Any]:
    backend = Atom3LevelEmulator(
        atom_model,
        arrangement=fq.emulator.AtomArrangement.chain(1, spacing=1.0),
        method="density_matrix",
    )
    return backend, atom_model.control.raman(0)


@pytest.mark.parametrize(
    "build_case",
    (_transmon_case, _atom2_case, _atom3_case),
    ids=("transmon", "atom2", "atom3"),
)
def test_time_rescaled_sampled_drives_remain_equivalent(
    build_case: Callable[[float, Any], tuple[Any, Any]],
    atom_3level_model: Any,
) -> None:
    normalized_grid = np.array((0.0, 0.45, 0.49, 0.5, 0.51, 0.55, 1.0))
    states = []

    for scale in (1e-6, 1e6):
        backend, channel = build_case(scale, atom_3level_model)
        waveform = fq.emulator.SampledWaveform(
            scale * normalized_grid,
            (2.0 * np.pi / scale) * np.sin(np.pi * normalized_grid) ** 2,
        )
        program = fq.Program(1)
        program.add(
            ops.PulseOperation(
                scale,
                (fq.emulator.PulseControl(channel, waveform),),
            )
        )

        result = backend.run(program).result()
        states.append(result.get_density_matrix())
        assert result.metadata["runtime_details"]["solver_options"][
            "max_step"
        ] == pytest.approx(scale / 200.0)

    assert 1.0 - states[0][0, 0].real > 0.5
    for state in states[1:]:
        assert np.allclose(state, states[0], rtol=0.0, atol=2e-4)
