
"""Control the dedicated FATQAT HAI GPU runner instance."""

import json
import os
import sys
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

from tencentcloud.common import credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import (
    TencentCloudSDKException,
)
from tencentcloud.hai.v20230812 import hai_client, models

INSTANCE_ID = os.environ["HAI_INSTANCE_ID"]
RUNNER_NAME = os.environ.get("HAI_RUNNER_NAME", "fatqat-t4")

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


def instance_state():
    req = models.DescribeInstancesRequest()
    req.InstanceIds = [INSTANCE_ID]
    instances = client.DescribeInstances(req).InstanceSet

    if len(instances) != 1:
        raise RuntimeError(f"Cannot find HAI instance {INSTANCE_ID}")

    return instances[0].InstanceState


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


def wait_for_runner():
    """Wait for the existing GitHub runner to reconnect."""
    token = os.environ["GH_RUNNER_READ_TOKEN"]
    repository = os.environ["GITHUB_REPOSITORY"]
    url = (
        f"https://api.github.com/repos/{repository}/actions/runners"
        f"?name={quote(RUNNER_NAME)}"
    )

    for _ in range(30):  # 10 minutes
        request = Request(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2026-03-10",
            },
        )

        with urlopen(request, timeout=20) as response:
            data = json.load(response)

        for runner in data.get("runners", []):
            if (
                runner["name"] == RUNNER_NAME
                and runner["status"] == "online"
                and not runner["busy"]
            ):
                print(f"Runner {RUNNER_NAME} is ready.", flush=True)
                return

        print("Waiting for GitHub runner to become ready...", flush=True)
        time.sleep(20)

    raise TimeoutError("GitHub runner did not connect within 10 minutes")


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

        if state in STOPPED:
            print("HAI instance is stopped.", flush=True)
            return

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
        wait_for_runner()
    else:
        stop_instance()
