from functools import partial
from threading import Lock

import fatqat as fq
import fatqat.operations as ops
from fatqat.simulator import Simulator
from fatqat.simulator._engine.np import NumpySVEngine


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


def test_auto_execution_without_process_support():
    class LocalEngine(NumpySVEngine):
        _supports_process_shots = False

        def _process_engine_factory(self):
            raise RuntimeError("this runtime cannot create process workers")

    backend = Simulator("SV", runtime="numpy")
    backend._engine = LocalEngine()
    config = {"seed": 5, "shot_parallelism": "auto", "max_workers": 2}
    program = _random_dynamic_program()
    result = backend.run(program, shots=64, simulation_config=config).result()
    assert sum(result.get_counts().values()) == 64
    assert set(result.get_counts()) <= {"00", "11"}


def test_workers_preserve_configuration_without_copying_runtime_state():
    class ConfiguredEngine(NumpySVEngine):
        def __init__(self, *, readout_flip):
            super().__init__()
            self._readout_flip = readout_flip
            # A live runtime resource must be recreated in each worker.
            self._runtime_lock = Lock()

        def _process_engine_factory(self):
            return partial(type(self), readout_flip=self._readout_flip)

        def measure_subsystems(self, indices, rng):
            with self._runtime_lock:
                digits = super().measure_subsystems(indices, rng)
            return tuple(digit ^ self._readout_flip for digit in digits)

    backend = Simulator("SV", runtime="numpy")
    backend._engine = ConfiguredEngine(readout_flip=1)
    program = fq.Program(1, 1)
    program.measure(0, 0)
    program.add(ops.Reset, 0)
    result = backend.run(
        program,
        shots=8,
        simulation_config={
            "seed": 5,
            "shot_parallelism": "processes",
            "max_workers": 2,
        },
    ).result()

    assert result.get_counts() == {"1": 8}
