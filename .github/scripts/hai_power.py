"""Control the dedicated FATQAT HAI instance from the GitHub-hosted job."""

import ipaddress
import os
from pathlib import Path
import sys
import time

from tencentcloud.common import credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import (
    TencentCloudSDKException,
)
from tencentcloud.hai.v20230812 import hai_client, models

INSTANCE_ID = os.environ["HAI_INSTANCE_ID"]

client = hai_client.HaiClient(
    credential.Credential(
        os.environ["TENCENTCLOUD_SECRET_ID"],
        os.environ["TENCENTCLOUD_SECRET_KEY"],
    ),
    os.environ["HAI_REGION"],
)

STOPPED = {"STOPPED", "STOPPED_NO_CHARGE"}
TRANSITIONAL = {"STARTING", "STOPPING", "PENDING", "REBOOTING"}
RETRYABLE = {
    "ResourceInsufficient.BundleInventoryShortage",
    "OperationDenied.InstanceOperationInProgress",
    "FailedOperation.OperateInstanceRepeatedly",
    "UnsupportedOperation.InstanceStateRunning",
}


def describe_instance():
    req = models.DescribeInstancesRequest()
    req.InstanceIds = [INSTANCE_ID]
    instances = client.DescribeInstances(req).InstanceSet

    if len(instances) != 1 or instances[0].InstanceId != INSTANCE_ID:
        raise RuntimeError(f"Cannot find HAI instance {INSTANCE_ID}")

    return instances[0]


def instance_state():
    return describe_instance().InstanceState


def public_ip():
    """Return a current routable address only after the instance is running."""
    instance = describe_instance()
    if instance.InstanceState != "RUNNING":
        raise RuntimeError("HAI is no longer RUNNING when querying its public IP")
    for value in instance.PublicIpAddresses or []:
        if not isinstance(value, str):
            continue
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            continue
        if (
            address.is_global
            and not address.is_multicast
            and not getattr(address, "scope_id", None)
        ):
            return str(address)
    raise RuntimeError("RUNNING HAI has no usable PublicIpAddresses")


def start_instance():
    deadline = time.monotonic() + 7200  # Up to 2 hours
    next_attempt = 0
    previous_state = None

    while time.monotonic() < deadline:
        state = instance_state()
        now = time.monotonic()

        if state != previous_state:
            print(f"HAI state: {state}", flush=True)
            previous_state = state

        if state == "RUNNING":
            print("HAI instance is running.", flush=True)
            return

        if state in STOPPED and now >= next_attempt:
            req = models.StartInstanceRequest()
            req.InstanceId = INSTANCE_ID

            try:
                response = client.StartInstance(req)
                print(f"Start task accepted: {response.TaskId}", flush=True)
            except TencentCloudSDKException as exc:
                if exc.get_code() not in RETRYABLE:
                    raise
                print(f"Retryable start error: {exc.get_code()}", flush=True)

            next_attempt = now + 60

        elif state not in STOPPED | TRANSITIONAL:
            raise RuntimeError(f"Cannot start instance in state {state}")

        time.sleep(20)

    raise TimeoutError("HAI instance did not start within 2 hours")


def stop_instance():
    deadline = time.monotonic() + 900  # 15 minutes
    next_attempt = 0
    previous_state = None

    while time.monotonic() < deadline:
        state = instance_state()
        now = time.monotonic()

        if state != previous_state:
            print(f"HAI state: {state}", flush=True)
            previous_state = state

        if state == "STOPPED_NO_CHARGE":
            print("HAI instance is STOPPED_NO_CHARGE.", flush=True)
            return
        if state == "STOPPED":
            raise RuntimeError(
                "HAI is STOPPED; STOPPED_NO_CHARGE (stop without charges) is not confirmed"
            )

        if state == "RUNNING" and now >= next_attempt:
            req = models.StopInstanceRequest()
            req.InstanceId = INSTANCE_ID
            req.StopMode = "STOP_CHARGE"

            try:
                response = client.StopInstance(req)
                print(f"Stop task accepted: {response.TaskId}", flush=True)
            except TencentCloudSDKException as exc:
                if exc.get_code() not in RETRYABLE:
                    raise
                print(f"Retryable stop error: {exc.get_code()}", flush=True)

            next_attempt = now + 60

        elif state not in TRANSITIONAL | {"RUNNING"}:
            raise RuntimeError(f"Cannot stop instance in state {state}")

        time.sleep(20)

    raise TimeoutError("HAI instance did not stop within 15 minutes")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"start", "stop"}:
        sys.exit("Usage: hai_power.py start|stop")

    if sys.argv[1] == "start":
        start_instance()
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
            output.write(f"public_ip={public_ip()}\n")
    else:
        stop_instance()
