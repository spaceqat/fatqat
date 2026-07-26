import pytest

import fatqat as fq
from fatqat import operations as ops
from fatqat.backends import SimulatorBackend


def _random_dynamic_program():
    p = fq.Program(2, 2)
    p.add(ops.H, 0)
    p.add(ops.CX, (0, 1))
    p.measure((0, 1), (0, 1))
    p.add(fq.ops.Reset, (0, 1))
    return p


@pytest.mark.parametrize(
    "parallel_mode,seed",
    [
        ("multiprocessing", 2026),
        ("loky", 7),
        ("auto", 99),
    ],
)
def test_parallel_counts_match_serial_for_same_seed(parallel_mode, seed):
    p = _random_dynamic_program()

    serial = (
        SimulatorBackend("SV")
        .run(p, shots=40, simulation_config={"seed": seed, "max_workers": 1})
        .result()
        .get_counts()
    )
    parallel = (
        SimulatorBackend("SV")
        .run(
            p,
            shots=40,
            simulation_config={
                "seed": seed,
                "max_workers": 2,
                "parallel_mode": parallel_mode,
            },
        )
        .result()
        .get_counts()
    )

    assert parallel == serial


def test_parallel_mode_serial_wins_over_max_workers():
    p = _random_dynamic_program()

    counts = (
        SimulatorBackend("SV")
        .run(
            p,
            shots=12,
            simulation_config={"seed": 11, "max_workers": 2, "parallel_mode": "serial"},
        )
        .result()
        .get_counts()
    )

    assert sum(counts.values()) == 12


def test_unknown_parallel_mode_rejected_at_run():
    # Per-run simulation configuration is validated before execution, rather
    # than being swallowed into a failed Job.
    with pytest.raises(
        fq.errors.BackendValidationError, match="unsupported parallel_mode"
    ):
        SimulatorBackend("SV").run(
            _random_dynamic_program(),
            simulation_config={"max_workers": 2, "parallel_mode": "not-a-mode"},
        )
