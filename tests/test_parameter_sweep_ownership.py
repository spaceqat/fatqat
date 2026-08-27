"""Protect the shared validate-once parameter sweep seam."""

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.observable import Observable


@pytest.mark.parametrize("family", ["simulator", "estimator"])
def test_sweep_normalizes_once_and_never_calls_public_binding(monkeypatch, family):
    angle = fq.Parameter("angle")
    program = fq.Program(1)
    program.add(ops.RY(angle), 0)
    bindings = {angle: np.array([0.1, 0.4, 0.7])}

    if family == "simulator":
        from fatqat.simulator import simulator as family_module

        executor = fq.simulator.Simulator("SV")

        def run_sweep():
            return executor.run_sweep(
                program,
                bindings,
                shots=0,
                result_config={"counts": False, "final_state": True},
            )

    else:
        from fatqat import estimator as family_module

        executor = fq.Estimator(fq.simulator.Simulator("SV"))
        observable = Observable([("Z", 1.0)])

        def run_sweep():
            return executor.run_sweep(program, observable, bindings)

    original_normalizer = family_module._normalize_parameter_batch
    calls = 0

    def count_normalization(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_normalizer(*args, **kwargs)

    monkeypatch.setattr(
        fq.Program,
        "assign_parameters",
        lambda *_args, **_kwargs: pytest.fail("public binding must not be called"),
    )
    monkeypatch.setattr(
        family_module,
        "_normalize_parameter_batch",
        count_normalization,
    )

    results = run_sweep().result()

    assert len(results) == 3
    assert calls == 1
