from dataclasses import replace
from functools import partial
from threading import Lock

import fatqat as fq
import fatqat.operations as ops
from fatqat.simulator import Simulator
from fatqat.simulator._engine.engines.numpy_engine import NumpySVEngine


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
            execution_config={
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
        _capabilities = replace(
            NumpySVEngine().capabilities, supports_process_shots=False
        )

        def _process_engine_factory(self):
            raise RuntimeError("this runtime cannot create process workers")

    backend = Simulator("SV", runtime="numpy")
    backend._engine = LocalEngine()
    config = {"seed": 5, "shot_parallelism": "auto", "max_workers": 2}
    program = _random_dynamic_program()
    result = backend.run(program, shots=64, execution_config=config).result()
    assert sum(result.get_counts().values()) == 64
    assert set(result.get_counts()) <= {"00", "11"}


def test_workers_preserve_configuration_without_copying_runtime_state():
    class ConfiguredEngine(NumpySVEngine):
        def __init__(self, *, initial_digit):
            super().__init__()
            self._initial_digit = initial_digit
            # A live runtime resource must be recreated in each worker.
            self._runtime_lock = Lock()

        def _process_engine_factory(self):
            return partial(type(self), initial_digit=self._initial_digit)

        def initialize(self, system_dims, n_clbits=0, *, initial_state=None):
            with self._runtime_lock:
                super().initialize(system_dims, n_clbits, initial_state=initial_state)
                self.state[:] = 0
                self.state[self._initial_digit] = 1

    backend = Simulator("SV", runtime="numpy")
    backend._engine = ConfiguredEngine(initial_digit=1)
    program = fq.Program(1, 1)
    program.measure(0, 0)
    program.add(ops.Reset, 0)
    result = backend.run(
        program,
        shots=8,
        execution_config={
            "seed": 5,
            "shot_parallelism": "processes",
            "max_workers": 2,
        },
    ).result()

    assert result.get_counts() == {"1": 8}
