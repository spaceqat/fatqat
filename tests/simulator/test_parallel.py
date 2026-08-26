import fatqat as fq
from fatqat import operations as ops
from fatqat.simulator import Simulator


def _random_dynamic_program():
    program = fq.Program(2, 2)
    program.add(ops.H, 0)
    program.add(ops.CX, (0, 1))
    program.measure((0, 1), (0, 1))
    program.add(ops.Reset, (0, 1))
    return program


def test_real_process_shots_return_counts_with_serial_children():
    import numba

    from fatqat.simulator._engine.parallel import _loky_executor

    result = (
        Simulator("SV", runtime="numba")
        .run(
            _random_dynamic_program(),
            shots=8,
            simulation_config={
                "seed": 5,
                "shot_parallelism": "processes",
                "kernel_parallelism": "serial",
                "max_workers": 2,
            },
        )
        .result()
    )

    assert sum(result.get_counts().values()) == 8
    assert set(result.get_counts()) <= {"00", "11"}
    assert _loky_executor(2).submit(numba.get_num_threads).result() == 1
