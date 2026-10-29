# GPU CI runner setup

The [GPU Tests workflow](../../.github/workflows/gpu.yml) starts an existing
Tencent Cloud HAI instance, waits for its runner, checks CUDA independently
of pytest, runs `tests/gpu_tests/`, and requests shutdown with `STOP_CHARGE`.
Start/stop jobs use GitHub-hosted CPU runners; the GPU job reuses a prepared
Python environment for manual main-branch smoke tests. Successful PR runs of
`Tests` start GPU selection; relevant changes require separate GPU approval.

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

This workflow uses `GH_RUNNER_READ_TOKEN` only to query runner readiness.
PR/run metadata and commit statuses use job-scoped `GITHUB_TOKEN` permissions.
The CPU control jobs install the Tencent Cloud SDK and run
[`hai_power.py`](../../.github/scripts/hai_power.py). Cloud credentials are
passed only to their control steps, not to the GPU test job.

For automatic PR GPU tests, a repository administrator must create the
**Settings → Environments → `gpu-ci`** environment and configure **Required
reviewers** with the maintainers who approve GPU execution. Allow the default
branch under any deployment branch restrictions. Disabling administrator
bypass is recommended. The workflow checks that reviewers are configured;
missing protection or an API error blocks the PR path. No new PAT is needed.
See [GitHub's environment configuration](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments).

Keep the existing cloud secrets at repository scope: `gpu-ci` protects only
the approval job, while automatic shutdown must run without a second review.
The manual default-branch smoke test does not require this environment.

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

For manual smoke tests, the workflow runs:

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

### Automatic PR GPU tests

Contributors open ordinary PRs. `Tests` and `Lint` keep their existing triggers
and fork-workflow policy; no GPU approval is added to CPU testing. Only a
successful PR run of `Tests` proceeds to GPU selection. Failed, cancelled,
incomplete and non-PR runs stop without GPU approval or cloud calls.

The trusted prepare job queries this PR's changed files, including additions,
modifications, deletions and both paths of renames. The centralized `GPU_PATHS`
rules in [`gpu_pr.py`](../../.github/scripts/gpu_pr.py) cover `tests/gpu_tests/`,
the GPU workflow, `gpu_pr.py`, `hai_power.py`, `tests/test_gpu_pr.py` and
`pyproject.toml`. Markdown/reStructuredText documents are excluded. The current
suite is a CuPy smoke test: ordinary CPU implementation/tests and documentation
changes do not request GPU. Extend the rules when adding GPU implementations
or dependency files. Selection uses the diff, not the existence of GPU tests.
File queries are paginated and checked for completeness; errors, changed
versions or GitHub's 3,000-file limit stop selection rather than imply no GPU.

Unrelated PRs end without approval, cloud operations or a pending GPU status.
Relevant PRs wait at the `gpu-ci` Environment for **Approve and deploy** from
a required reviewer. The selection job summary shows the PR number and full
head/base/merge SHAs. Rejection or cancellation before approval does not start
or stop HAI. After approval, the controller revalidates that fixed version
before any startup call; changed versions cannot reuse the approval. The
remaining startup, preflight, tests, cleanup and status reporting are automatic.

The GPU workflow and `gpu_pr.py` must be present on the target repository's
default branch for `workflow_run` to trigger. The controller checks the source
workflow ID, run attempt, successful conclusion and current PR head. For fork
runs with no `pull_requests` entries, it queries open PRs by fork owner and
branch, then matches repository ID and head SHA. Missing, closed, ambiguous,
conflicting or unconfirmed PRs are skipped. A new head must pass `Tests` first.

Selection pins three separate values: the PR head SHA for eligibility and
commit statuses, the current target branch's base SHA, and the current test
merge SHA returned by GitHub. The merge must have exactly the base and head
as its parents. After approval and again when the runner becomes available,
selection runs again;
changes to any of these SHAs skip testing. Checkout fetches the pinned merge
SHA and verifies both `HEAD` and its parents before installation or execution.
This tests the merged result, like ordinary PR Tests. It does **not** prove
that the selected current merge is the exact merge commit tested by the
preceding CPU run; the base branch may have advanced. Selection logs state
this limitation and record all three SHAs.

PR tests create a fresh virtual environment under `RUNNER_TEMP`, install the
selected FATQAT checkout with its `test-full` dependencies and pinned CuPy,
run the CUDA preflight, and execute `tests/gpu_tests`. Manual dispatch on the
default branch continues to use `$HOME/fatqat-ci-venv` and the original smoke
test. The HAI controller and single-instance concurrency group are shared.

A run awaiting GPU approval occupies the shared GPU workflow concurrency
slot. Approve, reject or cancel outdated requests to allow later GPU runs
to proceed.

The `GPU tests` status on the PR head commit links to the GPU workflow run.
Pending is posted only after approval and revalidation, during startup.
Rejected, cancelled or stale GPU requests report error if reporting can run;
unrelated PRs receive no GPU status. Success requires the start job, entire GPU job,
pytest step and stop job all to succeed. Failed pytest reports failure when
startup and shutdown succeed and the GPU job is not cancelled; installation,
preflight, a skipped pytest step, cancellation, post-job or shutdown failures
report error. This refers to the workflow step, not individual pytest test cases.
If GitHub rejects a status update, the reporting
job fails visibly; operators must inspect the run rather than assume a pending
status means tests are still running.

Local tests mock GitHub and check workflow gating without starting cloud
resources. Environment review/rejection, cancellation and the HAI lifecycle
still require GitHub/real-runner acceptance testing:

```bash
python -m pytest tests/test_gpu_pr.py
```

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

The start step records `attempted=true` before invoking HAI. Cleanup uses
`always()` and that flag, so startup errors and later installation, CUDA,
pytest or timeout failures still request shutdown. Ineligible requests that
never attempt startup do not call either cloud control command. Local tests
check these conditions; they cannot prove cleanup completes after GitHub
force-cancellation. There is no independent shutdown watchdog.
After cleanup failure or force-cancellation, check the instance
and stop it when no valid job is using it. Closing SSH does not stop cloud
billing; retained storage may still be charged.

### Validate both available and unavailable GPU environments

| Scenario | Expected behavior |
| --- | --- |
| Run the normal test suite without CuPy or a visible CUDA GPU | The GPU module skips; other tests continue |
| Run the GPU workflow with a working CUDA environment | Preflight succeeds and GPU tests execute rather than skip |
| Run the standalone preflight with no visible CUDA GPU | Preflight exits with an error before pytest starts |
| A GPU computation raises an error or returns an incorrect result | The check or test fails; the error is not converted into a skip |

### Validate PR selection and GPU approval

Use an open PR and check the following cases. These are acceptance
criteria, not a record of completed runs.

| Scenario | Expected behavior |
| --- | --- |
| Ordinary Tests fails | No GPU approval request or HAI startup |
| Tests passes and only unrelated files change | No GPU approval, cloud calls or pending GPU status |
| Tests passes with GPU-related changes, but approval is pending | GPU execution waits; no HAI startup |
| GPU approval is rejected or the run is cancelled before startup | No HAI start or stop call |
| The selected revision changes while awaiting approval | The old approval cannot start tests for the new revision |
| Approved GPU tests pass | CUDA preflight and pytest execute, the result is reported, and shutdown completes |
| Approved GPU tests fail | The PR receives a failing result and shutdown is still requested |
| Manual default-branch smoke test | No PR approval is required; preflight, tests and shutdown execute |

Actual Environment approval, cancellation and runner scheduling must be
verified on GitHub; local mocked tests do not validate those services.

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

### Persistent-runner execution

This design deliberately executes PR code directly on the existing persistent
self-hosted runner after successful CPU CI and separate `gpu-ci` approval.
Only the PR-code checkout enables `allow-unsafe-pr-checkout: true`: this is
an explicit maintainer-reviewed choice, not a security isolation mechanism.
A fresh Python virtual environment and `persist-credentials: false` do not
isolate that code from the host. Malicious PR code can alter host files, retain
processes, access runner credentials, or compromise later jobs; CPU test success
and environment approval do not prove that code is safe.
Maintainers must approve this execution model before enabling PR runs,
review external code before approving GPU execution, and reimage a
suspect runner.

Tencent Cloud secrets and `GH_RUNNER_READ_TOKEN` are supplied only to trusted
GitHub-hosted control jobs. The GPU job receives read-only repository, Actions
and PR permissions, with no status-write permission or cloud secrets. Status
reporting runs separately on a GitHub-hosted runner using default-branch code.
Do not store cloud credentials on the self-hosted machine. Shutdown preserves
its disk, and forced cancellation or runner loss can still require manual
shutdown; the existing workflow has no independent watchdog.
