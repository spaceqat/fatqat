# GPU CI runner setup

## 1. Workflow

The [GPU Tests workflow](../../.github/workflows/gpu.yml) independently requests
`gpu-ci` Environment approval whenever a PR is opened, updated or reopened.
All PRs are included, without file filters or a source-path allowlist.
CPU `Tests` and `Lint` keep their existing behavior and run independently;
GPU CI neither waits for nor reads their results.

Review the selected PR revision in the prepare job summary before approving.
Pending approval, rejection or cancellation before startup makes no HAI start
or stop call. After approval, the workflow rechecks the fixed revision, starts
the existing HAI instance, checks CUDA, runs GPU tests, shuts down and reports
the result. A changed PR revision cannot reuse an old approval.

If GitHub is still calculating mergeability, the controller queries up to five
times, two seconds apart. If it remains unknown, the job fails with a rerun
hint; conflicts and closed or outdated PRs do not proceed.

## 2. Runner setup

Use the existing dedicated Linux x86_64 NVIDIA GPU host with a compatible
driver, CUDA 12.0 at `/usr/local/cuda-12.0`, Python 3.12 and outbound HTTPS to
GitHub. The commands below assume service user `ubuntu`. Verify `nvidia-smi`.

On a new host, create the environment as that user, outside the checkout:

```bash
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh
"$HOME/.local/bin/uv" venv --python 3.12 "$HOME/fatqat-ci-venv"
UV_HTTP_TIMEOUT=300 "$HOME/.local/bin/uv" pip install \
  --python "$HOME/fatqat-ci-venv/bin/python" \
  "numpy==2.5.3" "cupy-cuda12x==13.6.0" "pytest==8.4.2" \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

Reuse an existing configured environment. The mirror is optional; omit
`--index-url` to use PyPI. Do not install multiple CuPy distributions.

In **Settings > Actions > Runners > New self-hosted runner**, follow the
Linux/x64 download, checksum and registration instructions in
`$HOME/actions-runner`. Register as `ubuntu` without `sudo`, using runner name
`fatqat-t4` and label `gpu`; keep one runner matching
`self-hosted`, `linux`, `x64`, `gpu`.

```bash
cd "$HOME/actions-runner"
sudo ./svc.sh install ubuntu
sudo ./svc.sh start
sudo ./svc.sh status
```

Confirm it is online in GitHub and reconnects after reboot. Do not also run
`./run.sh` while the service is active.

## 3. Repository configuration

Under **Settings > Secrets and variables > Actions**, configure:

| Type | Name | Value |
| --- | --- | --- |
| Secret | `TENCENTCLOUD_SECRET_ID` | CI CAM user's SecretId |
| Secret | `TENCENTCLOUD_SECRET_KEY` | Matching SecretKey |
| Secret | `GH_RUNNER_READ_TOKEN` | Existing token authorized to list repository runners |
| Variable | `HAI_REGION` | Instance region, e.g. `ap-shanghai` |
| Variable | `HAI_INSTANCE_ID` | Existing `hai-...` instance ID |
| Variable | `HAI_RUNNER_NAME` | Registered runner name, e.g. `fatqat-t4` |

For `GH_RUNNER_READ_TOKEN`, select the target repository and grant
**Administration: read-only**. Complete organization approval if required.

Grant the CAM user `hai:DescribeInstances`, `hai:StartInstance` and
`hai:StopInstance` for the CI instance. Verify both start and stop permissions.
Cloud secrets stay at repository scope and are used only by trusted
GitHub-hosted control jobs. Do not store them on the GPU host.
This workflow uses `GH_RUNNER_READ_TOKEN` only to query runner readiness.

Configure **Settings > Environments > gpu-ci > Required reviewers** for the
maintainers approving GPU execution. Allow the default branch under deployment
restrictions; missing reviewer protection blocks PR runs. Shutdown does not
require another review, and manual smoke tests do not require PR approval.

Place the workflow and controller on the upstream default branch. Ensure repository and organization Actions policies allow `pull_request_target`.

## 4. Running tests

GPU tests live in `tests/gpu_tests/`. In **Actions > GPU Tests > Run workflow**,
select **main** for the manual smoke test. The independent CUDA preflight
imports CuPy, requires a visible GPU and verifies a computation before pytest;
a failed preflight stops test execution.

Manual runs use `$HOME/fatqat-ci-venv`. For a local GPU run from the repository
root, use the same environment and paths:

```bash
export CUDA_PATH=/usr/local/cuda-12.0
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$HOME/fatqat-ci-venv/bin/python" -m pytest tests/gpu_tests --confcutdir=tests/gpu_tests -v -rs
```

Approved PR runs install the fixed FATQAT merge revision, `test-full`
dependencies and pinned CuPy into a fresh virtual environment before preflight
and pytest. Normal CPU testing remains `python -m pytest`; the CuPy module
can skip when a GPU environment is unavailable.

## 5. Operations

Follow the `GPU tests` status on the PR head commit to the workflow run.
Check the startup, CUDA preflight, pytest and shutdown jobs; success requires
all of them to succeed. Pending status is posted after approval and revalidation.

Shutdown is requested even after startup or test failure once startup has been
attempted. Confirm the HAI instance reaches `STOPPED_NO_CHARGE`, not just that
the job finishes. After cancellation, runner loss or cleanup failure, inspect
the instance and stop it when no valid job is using it; forced cancellation
can interrupt cleanup, and there is no independent shutdown watchdog.

Manual and PR runs share one HAI instance and the same concurrency group.
A run awaiting GPU approval occupies that slot. Approve, reject or cancel
outdated requests so later runs can proceed.

PR code executes on a persistent self-hosted runner after maintainer approval;
a virtual environment does not isolate it from the host.

Run controller unit tests with `python -m pytest tests/test_gpu_pr.py`.
