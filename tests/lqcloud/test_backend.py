import pytest

import fatqat as fq

from fatqat.lqcloud import LQCloudBackend
from fatqat.compiler.dialects.sc_native import NativeMeasure

pytest.importorskip("lqcloud")


def _circuit():
    circuit = fq.LogicalProgram(3, 4)
    circuit.add(fq.operations.H, 0)
    circuit.add(fq.operations.CCX, (0, 1, 2))
    circuit.add(fq.operations.iSwap, (0, 2))
    circuit.measure(2, 3)
    return circuit


@pytest.mark.parametrize("name", ["AGate-100", "QZ01-surface_code", "MQ02"])
def test_compile_submit_and_result_with_real_sdk(cloud, name):
    backend = LQCloudBackend(name)
    compiled = fq.compiler.compile_to_sc(_circuit(), backend, seed=7)
    assert cloud["submitted"] == []
    assert all(
        g.name in {"h", "rz", "cz", "barrier", "measure", "reset"}
        for g, _, _ in compiled.program.data
    )
    original = tuple(compiled.program.data)
    job = backend.run(compiled, shots=10)
    result = job.result(timeout=1)
    assert isinstance(result, fq.Result)
    assert result.get_counts() == {"0001": 10}
    assert result.available_data == frozenset({"counts"})
    assert result.metadata["job_id"] == job.job_id == "test-job"
    assert tuple(compiled.program.data) == original
    payload = cloud["submitted"][0]
    assert payload["qpu_name"] == name
    circuit = payload["command"]["circuit"]
    assert circuit["initial_layout"] == list(compiled.physical_layout)
    assert circuit["n_clbits"] == 4
    measured = [i for i in circuit["instructions"] if i["name"] == "measure"]
    native = [i for i in compiled.output.operations if isinstance(i, NativeMeasure)]
    assert [(i["qubits"][0], i["clbits"][0]) for i in measured] == [(native[0].site, 3)]
    assert any(i["name"] == "x21" for i in circuit["instructions"]) == (
        name != "QZ01-surface_code"
    )


def test_wrong_target_and_unmeasured_program_never_submit(cloud):
    first = LQCloudBackend("MQ02")
    second = LQCloudBackend("AGate-100")
    compiled = fq.compiler.compile_to_sc(_circuit(), first)
    with pytest.raises(ValueError, match="target"):
        second.run(compiled)
    unmeasured = fq.compiler.compile_to_sc(fq.LogicalProgram(1), first)
    with pytest.raises(ValueError, match="measurement"):
        first.run(unmeasured)
    with pytest.raises(TypeError):
        first.run(_circuit())
    assert cloud["submitted"] == []


@pytest.mark.parametrize("shots", [0, 50001, True, 1.5])
def test_invalid_shots_never_submit(cloud, shots):
    backend = LQCloudBackend("MQ02")
    compiled = fq.compiler.compile_to_sc(_circuit(), backend)
    with pytest.raises((ValueError, TypeError)):
        backend.run(compiled, shots=shots)
    assert cloud["submitted"] == []


@pytest.mark.parametrize("counts", [{"1": 10}, {"000x": 10}, {"0001": -1}, {"0001": 9}])
def test_malformed_counts_are_not_presented_as_success(cloud, counts):
    backend = LQCloudBackend("MQ02")
    compiled = fq.compiler.compile_to_sc(_circuit(), backend)
    cloud["result"] = {"counts": counts}
    with pytest.raises(ValueError, match="counts"):
        backend.run(compiled, shots=10).result(timeout=1)


def test_timeout_does_not_resubmit_or_cancel(cloud):
    from lqcloud import JobTimeoutError

    backend = LQCloudBackend("MQ02")
    job = backend.run(fq.compiler.compile_to_sc(_circuit(), backend), shots=10)
    cloud["status"] = "running"
    with pytest.raises(JobTimeoutError):
        job.result(timeout=1)
    assert job.status() == "RUNNING"
    cloud["status"] = "completed"
    assert job.result(timeout=1).get_counts() == {"0001": 10}
    assert len(cloud["submitted"]) == 1


def test_zero_timeout_cannot_silently_wait_forever(cloud):
    backend = LQCloudBackend("MQ02")
    job = backend.run(fq.compiler.compile_to_sc(_circuit(), backend), shots=10)
    with pytest.raises(ValueError, match="timeout"):
        job.result(timeout=0)


def test_qudit_inputs_are_not_reinterpreted_as_qubits(cloud):
    backend = LQCloudBackend("MQ02")
    q = fq.QuantumRegister(1, dim=3)
    c = fq.ClassicalRegister(1, dim=3)
    program = fq.LogicalProgram([q], [c])
    program.measure(q[0], c[0])
    with pytest.raises(fq.compiler.PassError, match="qubit"):
        fq.compiler.compile_to_sc(program, backend)


def test_cancel_and_failed_execution(cloud):
    from lqcloud import JobError

    backend = LQCloudBackend("MQ02")
    job = backend.run(fq.compiler.compile_to_sc(_circuit(), backend), shots=10)
    cloud["status"] = "failed"
    with pytest.raises(JobError):
        job.result(timeout=1)
    assert job.cancel()


def test_qasm_entry_and_early_emit(cloud):
    backend = LQCloudBackend("MQ02")
    text = fq.qasm.to_qasm(_circuit())
    compiled = fq.compiler.compile_qasm_to_sc(text, backend)
    assert compiled.output.IR_ID == "sc.lqcloud.native.v1"
    source = fq.compiler.compile_to_sc(
        _circuit(), backend, emit="gate.logical.source.v1"
    )
    assert type(source.output) is fq.LogicalProgram
    with pytest.raises(TypeError):
        backend.run(source)
