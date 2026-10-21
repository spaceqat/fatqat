# GPU CI runner setup

The [GPU Tests workflow](../../.github/workflows/gpu.yml) starts an existing
Tencent Cloud HAI instance, waits for its runner, checks CUDA independently
of pytest, runs `tests/gpu_tests/`, and requests shutdown with `STOP_CHARGE`.
Start/stop jobs use GitHub-hosted CPU runners; the GPU job reuses a prepared
Python environment. The workflow is manually dispatched.

## 1. Prepare the GPU host

Use a dedicated Linux x86_64 instance with an NVIDIA GPU, a compatible driver,
and the CUDA Toolkit installed. The commands below assume user `ubuntu`,
CUDA 12.0 at `/usr/local/cuda-12.0`, and Python 3.12. Adapt the workflow too
if these paths change. Check `nvidia-smi` before proceeding.

On a new host, create the environment as the service user:

```bash
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh
"$HOME/.local/bin/uv" venv --python 3.12 "$HOME/fatqat-ci-venv"
UV_HTTP_TIMEOUT=300 "$HOME/.local/bin/uv" pip install --python "$HOME/fatqat-ci-venv/bin/python" "numpy==2.5.3" "cupy-cuda12x==13.6.0" "pytest==8.4.2" --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

Reuse an existing configured environment rather than recreating it. Keep it
outside the runner checkout directory. The mirror is optional; omit
`--index-url` to use PyPI. PyTorch is not required. Do not install multiple
CuPy distributions in the same environment. See the
[CuPy installation guide](https://docs.cupy.dev/en/v13.6.0/install.html).

## 2. Register the runner

In the target repository, open **Settings → Actions → Runners → New
self-hosted runner → Linux / x64**. Follow the download and checksum
instructions in `$HOME/actions-runner`. Run the registration command as
`ubuntu`, without `sudo`; use a fresh token from that repository.

Use runner name `fatqat-t4`, add label `gpu`, and accept the default group
and work directory. Keep one dedicated runner matching
`self-hosted`, `linux`, `x64`, and `gpu`, or tighten the workflow labels.

```bash
cd "$HOME/actions-runner"
sudo ./svc.sh install ubuntu
sudo ./svc.sh start
sudo ./svc.sh status
```

Confirm the service is active and the runner is online in GitHub. It must
reconnect automatically after a reboot; do not also start `./run.sh`.
If release downloads fail, upload the official archive using SCP/SFTP and
verify its checksum on the host. The host still needs outbound HTTPS access
to GitHub Actions. See [runner setup](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners).

## 3. Configure cloud access and repository settings

Create a CI-only CAM user with programmatic access. Grant
`hai:DescribeInstances`, `hai:StartInstance`, and `hai:StopInstance` for the
CI instance using the narrowest effective resource scope. Verify both start
and stop authorization. Keep cloud credentials out of the repository and
off the GPU host.

Inspect all attached policies before deployment. A temporary
`hai:StopInstance` permission with `resource: "*"` is not restricted to the
CI instance and requires explicit maintainer approval. Do not retain a
wildcard debugging policy without review; resolve resource-scoped
authorization with Tencent Cloud and narrow the permission when possible.

Under **Settings → Secrets and variables → Actions**, configure:

| Type | Name | Value |
| --- | --- | --- |
| Secret | `TENCENTCLOUD_SECRET_ID` | CI CAM user's SecretId |
| Secret | `TENCENTCLOUD_SECRET_KEY` | Matching SecretKey |
| Secret | `GH_RUNNER_READ_TOKEN` | Token authorized to list this repository's runners |
| Variable | `HAI_REGION` | Instance region, e.g. `ap-shanghai` |
| Variable | `HAI_INSTANCE_ID` | Existing `hai-...` instance ID, not an IP address |
| Variable | `HAI_RUNNER_NAME` | Registered runner name, e.g. `fatqat-t4` |

For the runner-read token, select the target owner and repository and grant
**Administration: read-only**. The creator needs the corresponding repository
access; organization approval may be required. Track token expiration.
This is not the short-lived runner registration token.

The CPU control jobs install the Tencent Cloud SDK and run
[`hai_power.py`](../../.github/scripts/hai_power.py). Cloud credentials are
passed only to their control steps, not to the GPU test job.

## 4. Run tests

For the normal test suite, use the usual development environment and command:

```bash
python -m pytest
```

The CuPy test module skips itself if CuPy cannot be imported, CUDA device
detection reports an unavailable runtime, or no device is visible. Once a
test runs, computation errors and failed assertions remain failures. New
GPU-only test modules should implement an availability guard or share one;
do not add GPU exclusions to the normal CPU workflow.

To run GPU CI, select **Actions → GPU Tests → Run workflow → main**. The
`Verify CUDA before pytest` step uses the same Python and CUDA paths as the
tests. It imports CuPy, requires a visible GPU, performs a GPU computation,
and checks the result without pytest or skip handling. Any failure stops
the GPU job before test execution.

After preflight succeeds, the workflow runs only:

```bash
"$HOME/fatqat-ci-venv/bin/python" -m pytest tests/gpu_tests --confcutdir=tests/gpu_tests -v -rs
```

For a local GPU run, first set these paths and run the command above from
the repository root:

```bash
export CUDA_PATH=/usr/local/cuda-12.0
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

`--confcutdir` isolates this standalone smoke test from parent `conftest.py`
files. The Lint workflow installs CuPy for static import checks only.

## 5. Verify the lifecycle and handle failures

Start a validation run with the instance stopped. Confirm all of the following:

| Stage | Expected result |
| --- | --- |
| Start HAI GPU | Instance becomes `RUNNING`; the named runner becomes online and idle |
| Verify CUDA before pytest | `CUDA_PREFLIGHT_PASSED` |
| Run GPU tests only | `test_cupy_sum_of_squares PASSED`, not skipped |
| Stop HAI GPU | Stop request accepted; final state `STOPPED_NO_CHARGE` |

Confirm the final state in HAI too. A successful stop job on an already
stopped instance does not verify permission to shut down a running one.
Capacity shortages are retried within a two-hour startup window, but
capacity is not reserved while the instance is stopped.

The cleanup job uses `always()`, but there is no independent shutdown
watchdog. After cleanup failure or force-cancellation, check the instance
and stop it when no valid job is using it. Closing SSH does not stop cloud
billing; retained storage may still be charged.

### Validate both available and unavailable GPU environments

| Scenario | Expected behavior |
| --- | --- |
| Run the normal test suite without CuPy or a visible CUDA GPU | The GPU module skips; other tests continue |
| Run the GPU workflow with a working CUDA environment | Preflight succeeds and GPU tests execute rather than skip |
| Run the standalone preflight with no visible CUDA GPU | Preflight exits with an error before pytest starts |
| A GPU computation raises an error or returns an incorrect result | The check or test fails; the error is not converted into a skip |

## 6. Handoff and security

Merging the PR does not transfer Secrets, Variables, PAT permissions, runner
registration, or cloud-account ownership. Configure these in the target
repository and confirm the billing owner and CAM permissions.

Before migrating the same instance, disable the fork's GPU workflow and
wait for all its runs, including cleanup, to finish. Stop and uninstall the
old runner service, remove its registration using the fork's instructions,
register it to the upstream repository, and reinstall/start the service.
Reuse the CUDA/Python environment and verify the full lifecycle upstream.
Do not let both repositories control the instance: workflow concurrency is
scoped to one repository.

Run only trusted code on this persistent runner. Shutdown does not wipe its
disk, and `persist-credentials: false` is not an isolation boundary. External
PR execution requires a separate approval and isolation design. This workflow
has no automatic PR trigger or PR-selection input; selecting `main` tests
`main`, not an unmerged PR.