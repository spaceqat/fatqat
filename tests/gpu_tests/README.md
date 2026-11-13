# GPU CI over SSH

## Workflow and approval

[GPU Tests](../../.github/workflows/gpu.yml) has two `ubuntu-latest` jobs:
`approve` waits for approval; `gpu` uses system OpenSSH on the existing HAI instance.
PR events are `opened`, `synchronize` and `reopened`; there are no file filters.
CPU `Tests` and `Lint` run independently; the GPU workflow does not query them.

Each PR independently waits in `GPU approval`, using the `gpu-ci` Environment
with Required reviewers. An unapproved PR does not block other PRs. The run and
execution job names are `GPU PR #<number>` or `GPU manual`. Review the event's PR
head before approving. After approval and acquiring the GPU execution slot, the
controller checks the PR before startup, selects the current merge revision and
records the full head, base and merge SHAs in the logs and run summary.
The tested base is the merge commit's first parent; the second must be the
approved event head. A later PR head cannot reuse approval of an earlier event.

Manual dispatch is restricted to the default branch; its `approve` job succeeds
using `gpu-ci-manual` without Required reviewers. It tests the event's fixed
commit. Only `gpu` jobs share `fatqat-hai-gpu`, with `cancel-in-progress: false`
and `queue: max`: approved PRs and manual runs execute serially. Each round
finishes shutdown and cleanup before the next starts. Rejected approval starts
no GPU job or HAI instance.

## Repository configuration

Configure these at repository level under **Settings > Secrets and variables >
Actions**; `gpu` has no Environment and cannot read Environment-only values:

| Type | Name | Value |
| --- | --- | --- |
| Secret | `TENCENTCLOUD_SECRET_ID` | Existing CI CAM user's SecretId |
| Secret | `TENCENTCLOUD_SECRET_KEY` | Matching SecretKey |
| Secret | `GPU_SSH_PRIVATE_KEY` | CI SSH private key, without a passphrase |
| Secret | `GPU_SSH_KNOWN_HOSTS` | Verified server host key, as shown below |
| Variable | `HAI_REGION` | Existing instance region |
| Variable | `HAI_INSTANCE_ID` | Existing HAI instance ID |
| Variable | `GPU_SSH_USER` | `ubuntu` |
| Variable | `GPU_SSH_PORT` | `22` |

The known-hosts secret must contain exactly one ed25519 entry:

```text
fatqat-hai-gpu ssh-ed25519 <server-host-public-key>
```

Verify the host key through the HAI console or another trusted channel.
The alias remains fixed when the public IP changes. The controller queries
`PublicIpAddresses` for the configured instance after it reaches `RUNNING`;
missing or invalid addresses fail the run without a cached-IP fallback.

Grant CAM `hai:DescribeInstances`, `hai:StartInstance` and `hai:StopInstance`
for this instance. No security-group write permission is needed.
Configure **Settings > Environments** with `gpu-ci` requiring reviewers and
`gpu-ci-manual` having none; allow the default branch in deployment restrictions.
The controller validates the approval job's selected Environment and fails on
inaccessible or malformed data. Its job token needs `contents: read`,
`actions: read` and `pull-requests: read` (declared in YAML).

Place the workflow and control scripts on the upstream default branch.
No new PAT is needed. `GH_RUNNER_READ_TOKEN`, `HAI_RUNNER_NAME` and the old
self-hosted runner registration are unused and are not automatically removed.
Before startup, CI normalizes private-key line endings and uses local
`ssh-keygen` to verify it is readable without a passphrase, without printing keys.

## HAI prerequisites

The existing `ubuntu` account must accept the CI public key in `authorized_keys`.
SSH must already be reachable on the configured port; CI does not open ports.
The NVIDIA Tesla T4 driver must pass `nvidia-smi`, with CUDA installed at
`/usr/local/cuda-12.0` and Python at `/home/ubuntu/fatqat-ci-venv/bin/python`.
That existing environment must support venv creation and the manual CuPy/pytest
smoke test. Use `cupy-cuda12x==13.6.0`; do not install multiple CuPy distributions.
Package downloads require outbound HTTPS from HAI.

## Execution

1. Check out trusted upstream control code into `control/`, check configuration
   and pin the test revision after Environment approval.
2. Install the pinned Tencent Cloud SDK.
3. Record the startup attempt, start HAI and query its current public IP.
4. Wait for SSH with strict host-key verification, then print `hostname`,
   `whoami` and `nvidia-smi`.
5. Fetch the fixed source into a separate bare Git repository. Verify the PR
   merge ref, commit and parents, archive it, and recheck the PR revision.
6. Upload only the archive and trusted test driver into a fresh
   `/home/ubuntu/fatqat-gpu.*` directory, then recheck before installation.
7. Prepare the remote environment, run independent CUDA preflight, then pytest.
8. Request `STOP_CHARGE` through HAI API and verify `STOPPED_NO_CHARGE`.
9. Remove local SSH key and known-hosts files even after failure.

PR runs create a separate venv using the existing Python, install the uploaded
FATQAT source and `test-full` dependencies, and pin CuPy to `13.6.0`.
Manual default-branch runs reuse the existing venv for the original smoke test,
but use test files from this run's selected source. Both modes explicitly set
`CUDA_PATH`, `PATH` and `LD_LIBRARY_PATH`. Preflight requires a visible GPU and
checks an actual CuPy computation outside pytest before running:

```bash
python -m pytest tests/gpu_tests --confcutdir=tests/gpu_tests -v -rs
```

A JUnit report ensures at least one test executed: partial skips are allowed,
but no tests or all tests skipped cannot succeed. The remote working directory
is removed when the driver exits. Upload or subsequent PR recheck failure also
triggers a bounded SSH cleanup of this run's directory before shutdown. If SSH
cleanup fails, HAI API shutdown still runs. No old runner `_work` directory is used.
Cloud credentials, GitHub tokens and the SSH private key stay on GitHub's host.
Only trusted control scripts execute there; PR installation and tests run on HAI.
Remote output streams through temporary disk files with Actions command parsing
suspended; only an 8 KiB diagnostic tail stays in memory.

## Troubleshooting

View the GitHub Actions GPU check or open the `GPU Tests` workflow run and its
`GPU PR #<number>` or `GPU manual` job. Success requires revision checks,
transfer, preflight, pytest, shutdown and local key cleanup to succeed. A failed
required step remains a job failure even if cleanup succeeds.
Cancellation, tests not run or all skipped cannot succeed. A plain `STOPPED`
state fails shutdown because no-charge status is unconfirmed; CI never restarts
the instance to resolve that state. GitHub API errors include method/path;
unexpected controller errors include a traceback without environment dumps.
Unknown mergeability is retried five times, two seconds apart; rerun if it
remains unknown. A closed, conflicting or changed PR stops the run.

`TCP/SSH readiness timed out` means the ten-minute readiness budget expired.
Check the current IP, existing network rules and sshd. `public-key authentication
failed` points to the private key, username or `authorized_keys`. `host key
mismatch` requires verifying the server identity before updating the secret.
`Remote command/test failed` and `Remote GPU test deadline exceeded` identify
remote execution failures; inspect the installation, CUDA and pytest output.

Startup retains its two-hour retry budget. Remote setup and tests share a
30-minute deadline, with a separate SSH client timeout. Shutdown has a 15-minute
API polling budget and a 17-minute step limit; the 240-minute job exceeds all
step budgets combined. Shutdown runs after every attempted startup, independently
of SSH. After forced cancellation, manually verify HAI is stopped and stop it
if needed; forced cancellation can interrupt cleanup.

Offline GPU CI infrastructure tests live in `.github/ci_tests`, outside the
default `tests/` collection. Ordinary `python -m pytest` runs FATQAT tests and
does not collect these controller tests. Run them explicitly from the repo root:
`python -m pytest .github/ci_tests -v`.
They mock GitHub, HAI and SSH connections, require no Secrets or GPU, and use
local OpenSSH tools plus Bash, `timeout`, `sha256sum` and `tar` for shell tests.
The existing Ubuntu `Tests` job runs this suite separately; Windows and macOS
keep the FATQAT test suite. Black and Pylint also check `.github/ci_tests`.
