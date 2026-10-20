# GPU CI runner setup

The manually triggered [GPU Tests workflow](../../.github/workflows/gpu.yml)
starts an existing Tencent Cloud HAI instance, waits for its GitHub runner,
runs only `tests/gpu_tests/`, and requests shutdown with `STOP_CHARGE`.
The GPU environment is provisioned once and reused; the GPU job does not
install dependencies. The two control jobs run on GitHub-hosted CPU runners.

## 1. Validated configuration

[GPU Tests #6](https://github.com/zengzhphy/fatqat/actions/runs/39364530125)
passed on **2026-10-20**, using commit
`79764d01aa980840b83cd065cee9d45c9b65337a` in `zengzhphy/fatqat`.
The run started from `STOPPED_NO_CHARGE`, executed the
[CuPy smoke test](test_cupy.py), and finished at `STOPPED_NO_CHARGE`.
All three jobs passed; total elapsed time was about 2 minutes 45 seconds.
These observations validate this run, not future GPU availability.

| Component | Validated configuration |
| --- | --- |
| Cloud instance | Tencent Cloud HAI GPU Basic, Shanghai, NVIDIA Tesla T4 |
| Operating system / service user | Ubuntu 20.04, x86_64 / `ubuntu` |
| NVIDIA driver | 525.105.17 |
| System CUDA Toolkit | 12.0 at `/usr/local/cuda-12.0` |
| Python | 3.12.15 at `$HOME/fatqat-ci-venv/bin/python` |
| Python packages | `numpy==2.5.3`, `cupy-cuda12x==13.6.0`, `pytest==8.4.2` |
| GitHub runner | 2.337.0 at `$HOME/actions-runner`, name `fatqat-t4` |

These are observed versions, not minimum requirements. A replacement host
must be validated separately. Confirm `nvidia-smi` works and the toolkit is
installed; `nvidia-smi` alone does not validate Python GPU computation.
Adapt both this guide and the workflow if the CUDA path or service user changes.

## 2. Provision the GPU environment once

Run as the non-root service user on a new GPU host. Keep this environment
outside the runner checkout directory and do not replace the system Python.
On an existing configured host, reuse it instead of recreating it.

```bash
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh
"$HOME/.local/bin/uv" venv --python 3.12.15 "$HOME/fatqat-ci-venv"
UV_HTTP_TIMEOUT=300 "$HOME/.local/bin/uv" pip install --python "$HOME/fatqat-ci-venv/bin/python" "numpy==2.5.3" "cupy-cuda12x==13.6.0" "pytest==8.4.2" --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

The last command uses the mirror used during setup; omit `--index-url` to
use PyPI. These top-level pins are not a full dependency lock. PyTorch is
not required for this test. Do not install another CuPy distribution
alongside `cupy-cuda12x`.

Set CUDA lookup paths and verify actual GPU execution:

```bash
export CUDA_PATH=/usr/local/cuda-12.0
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$HOME/fatqat-ci-venv/bin/python" - <<'PY'
import cupy as cp

assert cp.cuda.runtime.getDeviceCount() > 0, "No CUDA GPU detected"
with cp.cuda.Device(0):
    x = cp.asarray([1.0, 2.0, 3.0], dtype=cp.float32)
    result_gpu = cp.sum(x * x)
    cp.cuda.get_current_stream().synchronize()
    result = float(result_gpu.get())
assert result == 14.0, result
print("CUPY_TEST_PASSED, result =", result)
PY
```

Expected: `CUPY_TEST_PASSED, result = 14.0`. The workflow sets the CUDA paths
again; shell exports do not configure an already running system service.
See [CuPy installation](https://docs.cupy.dev/en/v13.6.0/install.html) and
[uv installation](https://docs.astral.sh/uv/getting-started/installation/).

## 3. Register the runner in the target repository

Open **Settings → Actions → Runners → New self-hosted runner → Linux / x64**
in the repository that will execute the workflow. Follow its current Download
instructions in `$HOME/actions-runner`, including SHA-256 verification.
The recorded runner version is not a permanent version pin.

Run the page's `./config.sh --url ... --token ...` command as `ubuntu`, without
`sudo`. Use name `fatqat-t4`, the default group and `_work` directory, and add
label `gpu`. The workflow matches `self-hosted`, `linux`, `x64`, and `gpu`.
Use the same name in `HAI_RUNNER_NAME` below. Keep only one matching dedicated
GPU runner for this deployment, or tighten the workflow's labels.

After registration, install the service:

```bash
cd "$HOME/actions-runner"
sudo ./svc.sh install ubuntu
sudo ./svc.sh start
sudo ./svc.sh status
```

Confirm the service is active and the runner is online and idle in GitHub.
Do not also run `./run.sh`. The service runs independently of SSH and must
reconnect after the instance restarts.

If release downloads fail on the cloud host, download the official Linux
archive locally, upload with SCP/SFTP, and verify its checksum on the host.
This only solves file transfer: outbound HTTPS connectivity to GitHub Actions
and code-download endpoints is still required.

References: [runner registration](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners),
[service management](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/configure-the-application),
[network requirements](https://docs.github.com/en/actions/reference/runners/self-hosted-runners).

## 4. Configure cloud access and repository settings

Create a CI-only Tencent Cloud CAM user with programmatic access. Authorize
`hai:DescribeInstances`, `hai:StartInstance`, and `hai:StopInstance` for the
dedicated HAI instance, using the narrowest effective resource scope.
Keep its SecretId and SecretKey out of source control and the GPU host.

**Authorization issue observed during setup:** a resource-scoped
`StopInstance` call was denied, with an error resource containing the literal
`{InstanceId}`. A temporary `hai:StopInstance` policy with `resource: "*"`
was proposed for diagnosis. Run #6 proves that its effective credentials
could stop the instance; it does not establish which CAM policy was active.
Before handoff, the operator must inspect the effective policies. If wildcard
shutdown access remains necessary, disclose and obtain approval for that
broader scope; do not describe it as single-instance access. Resolve the
resource-scoping issue with Tencent Cloud and narrow the policy when possible.

In the **target repository**, open **Settings → Secrets and variables → Actions**:

| Type | Name | Value / purpose |
| --- | --- | --- |
| Secret | `TENCENTCLOUD_SECRET_ID` | CI-only CAM user's SecretId |
| Secret | `TENCENTCLOUD_SECRET_KEY` | Matching SecretKey |
| Secret | `GH_RUNNER_READ_TOKEN` | Token authorized to list this repository's runners |
| Variable | `HAI_REGION` | `ap-shanghai` for the current deployment |
| Variable | `HAI_INSTANCE_ID` | Actual `hai-...` instance ID, not an IP address |
| Variable | `HAI_RUNNER_NAME` | `fatqat-t4` |

For `GH_RUNNER_READ_TOKEN`, an authorized repository administrator can create
a fine-grained PAT with the target repository owner, only the target
repository, and **Administration: read-only**. Organization approval may be
required. This token is separate from the short-lived runner registration
token and the Tencent Cloud API credentials. Track its expiration.

The workflow provides cloud credentials only to the CPU control steps, not
to the GPU test step. Its controller is
[`.github/scripts/hai_power.py`](../../.github/scripts/hai_power.py).
Each CPU control job installs the pinned Tencent Cloud SDK packages; they
do not need to be installed on the GPU host.

References: [GitHub secrets](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets),
[runner API permissions](https://docs.github.com/en/rest/actions/self-hosted-runners),
[HAI CAM permissions](https://cloud.tencent.com/document/product/598/99248),
[StartInstance](https://cloud.tencent.com/document/product/1721/102449),
[StopInstance](https://cloud.tencent.com/document/product/1721/102448).

## 5. Execute and verify

With the workflow on the default branch, select
**Actions → GPU Tests → Run workflow → main**. Start from a stopped instance
to validate the complete cycle; do not manually power it on first.

| Job | Required observation |
| --- | --- |
| Start HAI GPU | Start request accepted; instance becomes `RUNNING`; named runner becomes online and idle |
| CuPy GPU smoke test | `tests/gpu_tests/test_cupy.py::test_cupy_sum_of_squares PASSED` and `1 passed` |
| Stop HAI GPU | Stop request accepted from a running instance; final state is `STOPPED_NO_CHARGE` |

Also confirm the final instance state in the HAI console. A green stop job
when the instance was already stopped does not test shutdown authorization.
The controller currently accepts both `STOPPED` and `STOPPED_NO_CHARGE` as
stopped; verify `STOPPED_NO_CHARGE` when validating release of compute.

The controller retries recognized startup errors, including capacity
shortages, within a two-hour startup window. Runner readiness has 30 polling
attempts, with 20-second sleeps and API request time added. Shutdown has a
15-minute controller window. The workflow serializes runs within this
repository; it is not a cross-repository lock.

For a local GPU test, use the environment and exports in section 2, then run
from a checkout's repository root:

```bash
"$HOME/fatqat-ci-venv/bin/python" -m pytest tests/gpu_tests --confcutdir=tests/gpu_tests -v
```

The [CPU workflow](../../.github/workflows/tests.yml) excludes this directory
with `python -m pytest --ignore=tests/gpu_tests`. A plain `python -m pytest`
still collects GPU files. The [Lint workflow](../../.github/workflows/lint.yml)
installs CuPy for import checks but does not execute GPU computations.
`--confcutdir` keeps the standalone GPU test from loading parent `conftest.py`
files, including the root's `scipy_doctest` dependency.

## 6. Upstream handoff and operating limits

Merging a PR transfers files, not runner registration, repository settings,
PAT permissions, or ownership of the Tencent Cloud account. For upstream use:

1. Have a maintainer approve the deployment, billing owner and effective CAM
   policy. Populate the upstream Secrets and Variables. The runner-read token
   must target `spaceqat/fatqat`, not just the personal fork.
2. Disable the fork's GPU workflow and wait for all its runs, including
   cleanup, to finish before migrating the shared instance.
3. On the GPU host, stop and uninstall the old runner service. Use the fork's
   runner removal instructions, register the existing installation to the
   upstream repository, and reinstall/start its service as `ubuntu`.
   Reuse the CUDA and Python environment.
4. Once the files are merged and configuration is ready, validate the full
   stopped → test → stopped cycle from the upstream repository. Do not enable
   simultaneous control of this instance from both repositories.

**Scope:** manually dispatched trusted-code tests, with automatic instance
startup/shutdown. There is no automatic external-PR trigger or PR-selection
input. Running on `main` tests `main`, not an unmerged contributor's PR.
Approval, explicit selection of the reviewed revision, and suitable isolation
must be designed before executing external PR code. Cloud shutdown does not
wipe the persistent runner; `persist-credentials: false` is not a sandbox.

**Failure handling:** the cleanup job uses `always()`, but there is no
independent shutdown watchdog. Do not treat this as a guarantee against
orphaned running instances. If cleanup fails, or a run is force-cancelled,
inspect the HAI console and stop the dedicated instance when no valid job
is using it. Failure-path and cancellation recovery are not validated by the
successful run #6 alone.

**Capacity and cost:** stopping releases compute while retaining the cloud
disk. Restart can fail when capacity is unavailable, and the public IP may
change. This workflow reuses a fixed instance ID; it does not guarantee the
same physical GPU or a bounded capacity wait. Closing SSH or stopping the
runner service does not stop cloud billing; retained storage may be charged.
See [HAI shutdown behavior](https://cloud.tencent.com/document/product/1721/102027).

Do not commit credentials, passwords, private keys or runner credential files.
Keep saved keys and account ownership available through an approved handoff
process, rather than public documentation.
