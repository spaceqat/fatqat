"""HAI power tests with a fake SDK; no cloud requests or runner API calls."""

import importlib.util
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture(name="hai")
def fake_hai(monkeypatch):
    for key in (
        "HAI_INSTANCE_ID",
        "HAI_REGION",
        "TENCENTCLOUD_SECRET_ID",
        "TENCENTCLOUD_SECRET_KEY",
    ):
        monkeypatch.setenv(key, "hai-test" if key == "HAI_INSTANCE_ID" else "canary")
    modules = {}
    for name in (
        "tencentcloud",
        "tencentcloud.common",
        "tencentcloud.common.credential",
        "tencentcloud.common.exception",
        "tencentcloud.common.exception.tencent_cloud_sdk_exception",
        "tencentcloud.hai",
        "tencentcloud.hai.v20230812",
        "tencentcloud.hai.v20230812.hai_client",
        "tencentcloud.hai.v20230812.models",
    ):
        modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, modules[name])
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, modules[name])

    class SDKError(Exception):
        def get_code(self):
            return self.args[0]

    model = modules["tencentcloud.hai.v20230812.models"]
    for name in (
        "DescribeInstancesRequest",
        "StartInstanceRequest",
        "StopInstanceRequest",
    ):
        setattr(model, name, SimpleNamespace)
    modules["tencentcloud.common.credential"].Credential = lambda *_: None
    modules[
        "tencentcloud.common.exception.tencent_cloud_sdk_exception"
    ].TencentCloudSDKException = SDKError
    instance = SimpleNamespace(
        InstanceId="hai-test", InstanceState="RUNNING", PublicIpAddresses=["8.8.8.8"]
    )
    calls, states, failures = [], [], []

    def describe(request):
        assert request.InstanceIds == ["hai-test"]
        if states:
            instance.InstanceState = states.pop(0)
        return SimpleNamespace(InstanceSet=[instance])

    def start(request):
        assert request.InstanceId == "hai-test"
        calls.append("start")
        if failures:
            raise SDKError(failures.pop(0))
        return SimpleNamespace(TaskId="start-task")

    def stop(request):
        assert request.InstanceId == "hai-test"
        assert request.StopMode == "STOP_CHARGE"
        calls.append("stop")
        if failures:
            raise SDKError(failures.pop(0))
        return SimpleNamespace(TaskId="stop-task")

    client = SimpleNamespace(
        DescribeInstances=describe, StartInstance=start, StopInstance=stop
    )
    modules["tencentcloud.hai.v20230812.hai_client"].HaiClient = lambda *_: client
    spec = importlib.util.spec_from_file_location(
        "hai_power",
        Path(__file__).resolve().parents[2] / ".github/scripts/hai_power.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clock = [0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        module.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    return SimpleNamespace(
        module=module,
        instance=instance,
        calls=calls,
        states=states,
        failures=failures,
        client=client,
        clock=clock,
        error=SDKError,
    )


def test_public_ip_comes_from_current_instance_response(hai):
    assert hai.module.public_ip() == "8.8.8.8"
    hai.instance.PublicIpAddresses = ["invalid", "10.0.0.1", "1.1.1.1"]
    assert hai.module.public_ip() == "1.1.1.1"
    assert not hai.calls


@pytest.mark.parametrize(
    "addresses",
    [
        None,
        [],
        [""],
        ["invalid"],
        ["127.0.0.1"],
        ["10.0.0.1"],
        ["224.0.0.1"],
        ["2606:4700::1%bad\nready=true"],
        [134744072],
    ],
)
def test_missing_or_invalid_public_ip_fails_without_cached_fallback(hai, addresses):
    assert hai.module.public_ip() == "8.8.8.8"
    hai.instance.PublicIpAddresses = addresses
    with pytest.raises(RuntimeError, match="no usable PublicIpAddresses"):
        hai.module.public_ip()


def test_ip_requires_running_state_and_matching_instance_id(hai):
    hai.instance.InstanceState = "STOPPED"
    with pytest.raises(RuntimeError, match="no longer RUNNING"):
        hai.module.public_ip()
    hai.instance.InstanceId = "hai-other"
    with pytest.raises(RuntimeError, match="Cannot find HAI instance"):
        hai.module.public_ip()
    hai.client.DescribeInstances = lambda _: SimpleNamespace(InstanceSet=[])
    with pytest.raises(RuntimeError, match="Cannot find HAI instance"):
        hai.module.public_ip()


def test_start_exports_only_current_validated_address(hai, monkeypatch, tmp_path):
    target = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(target))
    monkeypatch.setattr(sys, "argv", ["hai_power.py", "start"])
    runpy.run_path(hai.module.__file__, run_name="__main__")
    assert target.read_text(encoding="utf-8") == "public_ip=8.8.8.8\n"
    hai.instance.PublicIpAddresses = []
    target.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no usable PublicIpAddresses"):
        runpy.run_path(hai.module.__file__, run_name="__main__")
    assert target.read_text(encoding="utf-8") == ""


@pytest.mark.parametrize("stopped_state", ["STOPPED", "STOPPED_NO_CHARGE"])
def test_start_retries_capacity_errors_and_returns_on_running(hai, stopped_state):
    hai.states.extend([stopped_state] * 4 + ["STARTING", "RUNNING"])
    hai.failures.append("ResourceInsufficient.BundleInventoryShortage")
    hai.module.start_instance()
    assert hai.calls == ["start", "start"]


def test_stop_retries_and_verifies_stopped_no_charge(hai):
    hai.states.extend(["RUNNING"] * 4 + ["STOPPING", "STOPPED_NO_CHARGE"])
    hai.failures.append("OperationDenied.InstanceOperationInProgress")
    hai.module.stop_instance()
    assert hai.calls == ["stop", "stop"]


@pytest.mark.parametrize("states", [["STOPPED"], ["RUNNING", "STOPPING", "STOPPED"]])
def test_plain_stopped_is_not_confirmed_no_charge_and_never_restarts(hai, states):
    hai.states.extend(states)
    with pytest.raises(RuntimeError, match="STOPPED_NO_CHARGE.*not confirmed"):
        hai.module.stop_instance()
    assert hai.calls == (["stop"] if states[0] == "RUNNING" else [])
    assert hai.clock[0] <= 40


def test_already_stopped_no_charge_succeeds_without_power_calls(hai):
    hai.instance.InstanceState = "STOPPED_NO_CHARGE"
    hai.module.stop_instance()
    assert hai.calls == []


@pytest.mark.parametrize(
    "action,state,budget", [("start", "STARTING", 7200), ("stop", "STOPPING", 900)]
)
def test_power_wait_has_finite_deadline(hai, action, state, budget):
    hai.instance.InstanceState = state
    with pytest.raises(TimeoutError):
        getattr(hai.module, f"{action}_instance")()
    assert budget <= hai.clock[0] <= budget + 20


@pytest.mark.parametrize("action,state", [("start", "STOPPED"), ("stop", "RUNNING")])
def test_non_retryable_power_error_propagates(hai, action, state):
    hai.instance.InstanceState = state
    hai.failures.append("UnauthorizedOperation")
    with pytest.raises(hai.error):
        getattr(hai.module, f"{action}_instance")()
