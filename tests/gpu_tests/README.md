# GPU CI runner setup

Setup guide for the manually triggered [GPU Tests workflow](../../.github/workflows/gpu.yml). It runs only `tests/gpu_tests/` on a self-hosted NVIDIA GPU runner. The runner environment is provisioned once; the workflow does not install dependencies or start/stop cloud instances.

## 1. Validated environment

The following configuration passed the [CuPy smoke test](test_cupy.py) through GitHub Actions on **2026-10-15** ([successful run](https://github.com/zengzhphy/fatqat/actions/runs/38685866785)). These are observed versions, not minimum hardware requirements.

| Component | Validated configuration |
| --- | --- |
| Cloud instance | Tencent Cloud HAI GPU Basic, one NVIDIA Tesla T4 |
| Operating system / user | Ubuntu 20.04, x86_64 / `ubuntu` |
| NVIDIA driver | 525.105.17 |
| System CUDA Toolkit | 12.0 at `/usr/local/cuda-12.0` |
| Python | 3.12.15 at `$HOME/fatqat-ci-venv/bin/python` |
| Python packages | `numpy==2.5.3`, `cupy-cuda12x==13.6.0`, `pytest==8.4.2` |
| GitHub runner | 2.337.0, installed at `$HOME/actions-runner` |

Provision a Linux GPU instance with an NVIDIA driver and CUDA Toolkit already installed. Confirm `nvidia-smi` works and the toolkit path exists before proceeding. `nvidia-smi` alone does not verify that the toolkit or Python GPU libraries are usable. If the image uses another CUDA version or location, adapt both this setup and the workflow and repeat the GPU test; the recipe below assumes the configuration above. See [CuPy's installation requirements](https://docs.cupy.dev/en/v13.6.0/install.html).

## 2. Prepare the Python environment

Run these commands **on the cloud host as the non-root user that will run the service** (`ubuntu` here). Keep the environment outside the runner's checkout directory. Do not replace the system Python.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and create the environment on a new host:

```bash
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh
"$HOME/.local/bin/uv" venv --python 3.12.15 "$HOME/fatqat-ci-venv"
UV_HTTP_TIMEOUT=300 "$HOME/.local/bin/uv" pip install --python "$HOME/fatqat-ci-venv/bin/python" "numpy==2.5.3" "cupy-cuda12x==13.6.0" "pytest==8.4.2" --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

The final command uses the Tsinghua PyPI mirror used during setup. Omit `--index-url` to use PyPI instead. On an already configured host, reuse the environment rather than recreating it. These top-level pins record the validated versions; they are not a complete dependency lock.

PyTorch was used for an initial hardware check but is **not required by this CuPy test**. Do not install another CuPy distribution (`cupy` or `cupy-cuda11x`) alongside `cupy-cuda12x`.

Set the [CUDA lookup paths](https://docs.cupy.dev/en/v13.6.0/install.html#working-with-custom-cuda-installation), then verify actual GPU execution:

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

Expected: `CUPY_TEST_PASSED, result = 14.0`. These exports affect only this shell; the workflow sets them again for its job. The existing-host test passed; a clean replacement host must pass the same checks before use.

## 3. Register and start the GitHub runner

In the **target repository**, enable Actions if necessary, then open **Settings → Actions → Runners → New self-hosted runner → Linux / x64**. Repository administration permission is required.

In `$HOME/actions-runner`, follow the page's Download commands, verify its SHA-256 checksum, and extract the archive. Use the current supported runner provided by GitHub; 2.337.0 is the version used for the recorded run, not a permanent pin.

Run the page's `./config.sh --url ... --token ...` command as `ubuntu`, without `sudo`. Select the default group and `_work` directory, give the runner a unique name (for example `fatqat-t4`), and add the label **`gpu`**. The optional `tencent-hai` label is descriptive; the workflow requires `self-hosted`, `linux`, `x64`, and `gpu`.

Use a fresh registration token, which expires after one hour. Never commit tokens, passwords, private keys, or runner credential files. See [GitHub's registration instructions](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners).

After registration, install and start the [system service](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/configure-the-application):

```bash
cd "$HOME/actions-runner"
sudo ./svc.sh install ubuntu
sudo ./svc.sh start
sudo ./svc.sh status
```

Expect `active (running)` and an online, idle runner in GitHub. Do not also run `./run.sh` in parallel. The service runs independently of the SSH session.

**Network fallback:** if GitHub release downloads fail on the cloud host, download the same official Linux archive on a local computer, upload it to `$HOME/actions-runner` using SCP/SFTP, and verify its checksum on the host before extracting. This solves package transfer only: the host still needs outbound HTTPS access to GitHub Actions and code-download endpoints. Follow [GitHub's endpoint requirements](https://docs.github.com/en/actions/reference/runners/self-hosted-runners).

## 4. Run and maintain the tests

Keep the workflow on the repository's default branch, then select **Actions → GPU Tests → Run workflow**. Expected output:

```text
tests/gpu_tests/test_cupy.py::test_cupy_sum_of_squares PASSED
1 passed
```

For a local rerun on the GPU host, from a checkout's **repository root**, use the environment and CUDA exports from section 2:

```bash
"$HOME/fatqat-ci-venv/bin/python" -m pytest tests/gpu_tests --confcutdir=tests/gpu_tests -v
```

The [CPU workflow](../../.github/workflows/tests.yml) uses `python -m pytest --ignore=tests/gpu_tests`. The [Lint workflow](../../.github/workflows/lint.yml) installs `cupy-cuda12x==13.6.0` for import checks but does not execute GPU computations. A plain `python -m pytest` still collects GPU files, so use the exclusion on non-GPU machines. `--confcutdir` prevents this standalone smoke test from loading parent `conftest.py` files, including the root's `scipy_doctest` dependency; see [pytest's reference](https://docs.pytest.org/en/stable/reference/reference.html).

**Current scope:** manual triggering only; no automatic PR trigger, cloud startup, shutdown, or ephemeral isolation. Use this persistent runner for trusted code only. Before enabling external PR execution, arrange appropriate approval and isolation; `persist-credentials: false` is not a sandbox.

**Upstream handoff:** a repository-scoped runner registered to a fork cannot accept upstream jobs. A maintainer must arrange upstream registration and matching labels/environment. Merging the workflow does not transfer runner registration.

**Cost control:** closing SSH or stopping the runner service does not stop cloud-instance billing. When no job is running and the host is no longer needed, shut down the instance through the HAI console; retained storage can still incur charges. See [HAI billing](https://cloud.tencent.com/document/product/1721/101029). After powering it on again, confirm the service and GPU are available before triggering another job.
