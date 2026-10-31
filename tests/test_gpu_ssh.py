"""Offline OpenSSH connection, transfer, output and remote test boundary tests."""

import base64
import importlib.util
import io
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gpu_ssh", ROOT / ".github/scripts/gpu_ssh.py"
)
gpu_ssh = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gpu_ssh)
HOST_KEY = base64.b64encode(
    b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20" + b"k" * 32
).decode()


@pytest.fixture(name="key_pairs", scope="module")
def temporary_key_pairs(tmp_path_factory, ssh_tools):
    directory = tmp_path_factory.mktemp("gpu-ssh-keys")
    pairs = {}
    for name, passphrase in (("plain", ""), ("encrypted", "test-only-passphrase")):
        key = directory / name
        subprocess.run(
            [ssh_tools, "-q", "-t", "ed25519", "-N", passphrase, "-f", str(key)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=15,
        )
        pairs[name] = (
            key.read_text(encoding="utf-8"),
            key.with_suffix(".pub").read_text(encoding="utf-8"),
        )
    return pairs


@pytest.fixture(name="ssh_tools", scope="module")
def local_openssh():
    # Git OpenSSH supports mode bits in Windows pytest temp directories; the
    # native Windows port rejects pytest's inherited OWNER RIGHTS ACL.
    executable = shutil.which("ssh-keygen")
    with pytest.MonkeyPatch.context() as patch:
        if os.name == "nt":
            executable = str(
                Path(shutil.which("git")).parents[1] / "usr/bin/ssh-keygen.exe"
            )
            assert Path(executable).is_file()
            patch.setenv(
                "PATH", str(Path(executable).parent) + os.pathsep + os.environ["PATH"]
            )
        assert executable is not None
        yield executable


@pytest.fixture(name="ssh_environment")
def configured_environment(tmp_path, key_pairs):
    return {
        "RUNNER_TEMP": str(tmp_path),
        "GPU_SSH_USER": "ubuntu",
        "GPU_SSH_PORT": "22",
        "GPU_PUBLIC_IP": "8.8.8.8",
        "PATH": os.environ["PATH"],
        **{
            name: os.environ[name]
            for name in ("SYSTEMROOT", "TEMP", "TMP")
            if name in os.environ
        },
        "GPU_SSH_PRIVATE_KEY": key_pairs["plain"][0],
        "GPU_SSH_KNOWN_HOSTS": f"fatqat-hai-gpu ssh-ed25519 {HOST_KEY}",
        "TENCENTCLOUD_SECRET_ID": "cloud-id-canary",
        "TENCENTCLOUD_SECRET_KEY": "cloud-key-canary",
        "HAI_REGION": "region",
        "HAI_INSTANCE_ID": "hai-test",
        "GH_TOKEN": "github-token-canary",
        "SSH_AUTH_SOCK": "agent-canary",
    }


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r", "mixed"])
def test_generated_private_key_is_normalized_and_validated(
    ssh_environment, key_pairs, newline, capsys
):
    private, public = key_pairs["plain"]
    if newline == "mixed":
        lines = private.splitlines()
        supplied = "".join(
            line + ("\r\n" if i % 2 else "\r") for i, line in enumerate(lines)
        )
    else:
        supplied = private.replace("\n", newline)
    ssh_environment["GPU_SSH_PRIVATE_KEY"] = supplied
    assert gpu_ssh.main(["configure"], ssh_environment) == 0
    key = gpu_ssh.local_directory(ssh_environment) / "id_ed25519"
    assert key.read_bytes() == private.encode("utf-8")
    output = capsys.readouterr()
    assert output.out == output.err == ""
    assert (
        private not in output.out + output.err and public not in output.out + output.err
    )
    result = subprocess.run(
        ["ssh-keygen", "-y", "-P", "", "-f", str(key)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
        timeout=15,
    )
    assert result.stdout.decode().split()[:2] == public.split()[:2]


@pytest.mark.parametrize("kind", ["encrypted", "invalid"])
def test_unreadable_private_key_fails_configuration_without_logging_key(
    ssh_environment, key_pairs, kind, capsys
):
    private, public = key_pairs["encrypted"]
    if kind == "invalid":
        private = "invalid-private-key-canary\n"
    ssh_environment["GPU_SSH_PRIVATE_KEY"] = private
    assert gpu_ssh.main(["configure"], ssh_environment) == 1
    output = capsys.readouterr()
    assert "invalid or requires a passphrase" in output.err
    assert (
        private not in output.out + output.err and public not in output.out + output.err
    )


@pytest.mark.parametrize("exit_code", [0, 7])
def test_local_process_output_arrives_before_exit_with_commands_suspended(
    ssh_environment, tmp_path, monkeypatch, exit_code
):
    gpu_ssh.configure(ssh_environment)
    acknowledge = tmp_path / "log-received"

    class LiveLog(io.StringIO):
        def write(self, text):
            if "stderr-first" in text:
                acknowledge.touch()
            return super().write(text)

    output = LiveLog()
    monkeypatch.setattr(gpu_ssh.sys, "stdout", output)
    script = """
import os, pathlib, sys, time
assert not any(name in os.environ for name in (
    'GH_TOKEN', 'GPU_SSH_PRIVATE_KEY', 'TENCENTCLOUD_SECRET_KEY', 'SSH_AUTH_SOCK'))
print('::error::stdout-first', flush=True)
print('stderr-first', file=sys.stderr, flush=True)
deadline = time.monotonic() + 3
while not pathlib.Path(sys.argv[1]).exists():
    if time.monotonic() >= deadline:
        sys.exit(99)
    time.sleep(0.01)
print('process-finished', flush=True)
sys.exit(int(sys.argv[2]))
"""
    arguments = [sys.executable, "-u", "-c", script, str(acknowledge), str(exit_code)]
    assert gpu_ssh.SSH(ssh_environment).execute(arguments, timeout=10) == exit_code
    text = output.getvalue()
    markers = re.findall(r"::stop-commands::([0-9a-f]{64})", text)
    assert len(markers) == 1
    assert text.index("::stop-commands::") < text.index("::error::stdout-first")
    assert text.index("process-finished") < text.index(f"::{markers[0]}::")
    assert markers[0] not in str(arguments)


def test_large_local_process_output_is_streamed_in_bounded_chunks(
    ssh_environment, monkeypatch
):
    gpu_ssh.configure(ssh_environment)

    class CountingLog:
        size = 0
        largest_write = 0

        def write(self, text):
            self.size += len(text)
            self.largest_write = max(self.largest_write, len(text))
            return len(text)

        def flush(self):
            pass

    output = CountingLog()
    monkeypatch.setattr(gpu_ssh.sys, "stdout", output)
    arguments = [
        sys.executable,
        "-u",
        "-c",
        "import sys; [sys.stdout.write('x' * 8192) for _ in range(512)]",
    ]
    assert gpu_ssh.SSH(ssh_environment).execute(arguments, timeout=15) == 0
    assert output.size >= 8192 * 512
    assert output.largest_write <= 8192


@pytest.mark.parametrize("exit_code", [0, 7])
def test_local_process_receives_binary_file_stdin(ssh_environment, tmp_path, exit_code):
    gpu_ssh.configure(ssh_environment)
    payload = tmp_path / "payload"
    payload.write_bytes(bytes(range(256)) * 4096)
    received = tmp_path / "received"
    arguments = [
        sys.executable,
        "-c",
        "import pathlib, sys; pathlib.Path(sys.argv[1]).write_bytes(sys.stdin.buffer.read()); "
        "sys.exit(int(sys.argv[2]))",
        str(received),
        str(exit_code),
    ]
    with payload.open("rb") as stream:
        assert (
            gpu_ssh.SSH(ssh_environment).execute(arguments, timeout=10, stdin=stream)
            == exit_code
        )
    assert received.read_bytes() == payload.read_bytes()


def test_local_process_timeout_kills_client_and_restores_command_parsing(
    ssh_environment, tmp_path, capsys
):
    gpu_ssh.configure(ssh_environment)
    payload = tmp_path / "payload"
    payload.write_bytes(bytes(range(256)))
    arguments = [
        sys.executable,
        "-u",
        "-c",
        "import sys, time; assert sys.stdin.buffer.read() == bytes(range(256)); "
        "print('before-timeout', flush=True); time.sleep(10)",
    ]
    started = time.monotonic()
    with (
        payload.open("rb") as stream,
        pytest.raises(gpu_ssh.SSHFailure, match="TCP/SSH operation timed out"),
    ):
        gpu_ssh.SSH(ssh_environment).execute(arguments, timeout=1, stdin=stream)
    assert time.monotonic() - started < 5
    text = capsys.readouterr().out
    marker = re.search(r"::stop-commands::([0-9a-f]{64})", text)[1]
    assert text.index("before-timeout") < text.index(f"::{marker}::")
    assert not list(gpu_ssh.local_directory(ssh_environment).glob("log-*"))


@pytest.fixture(name="connection")
def ssh_case(ssh_environment, monkeypatch):
    environment = ssh_environment
    gpu_ssh.configure(environment)
    (Path(environment["RUNNER_TEMP"]) / "fatqat-gpu-source.tar").write_bytes(
        b"opaque source archive\x00\xff\r\n"
    )
    calls = []
    results = []
    clock = [0.0]

    class Process:
        def __init__(self, command, **kwargs):
            calls.append((command, kwargs))
            response = results.pop(0) if results else (0, b"", b"")
            if isinstance(response, subprocess.TimeoutExpired):
                self.returncode = None
                stdout, stderr = response.output, response.stderr
            else:
                self.returncode, stdout, stderr = response
            kwargs["stdout"].write((stdout or b"") + (stderr or b""))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

        def wait(self):
            assert self.returncode is not None
            return self.returncode

    monkeypatch.setattr(gpu_ssh.subprocess, "Popen", Process)
    monkeypatch.setattr(gpu_ssh.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        gpu_ssh.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    return SimpleNamespace(
        env=environment, calls=calls, results=results, ssh=gpu_ssh.SSH(environment)
    )


def test_config_writes_key_and_known_hosts_only_inside_runner_temp(connection, capsys):
    directory = gpu_ssh.local_directory(connection.env)
    assert (directory / "id_ed25519").read_text(encoding="utf-8") == connection.env[
        "GPU_SSH_PRIVATE_KEY"
    ]
    assert (
        (directory / "known_hosts")
        .read_text()
        .startswith("fatqat-hai-gpu ssh-ed25519 ")
    )
    if os.name != "nt":
        assert (directory / "id_ed25519").stat().st_mode & 0o777 == 0o600
        assert directory.stat().st_mode & 0o777 == 0o700
    assert not capsys.readouterr().out


@pytest.mark.parametrize(
    "field,value",
    [
        ("TENCENTCLOUD_SECRET_ID", ""),
        ("GPU_SSH_USER", "ubuntu;id"),
        ("GPU_SSH_PORT", "0"),
        ("GPU_SSH_PORT", "65536"),
        ("GPU_SSH_KNOWN_HOSTS", "fatqat-hai-gpu ssh-rsa AAAA"),
        ("GPU_SSH_KNOWN_HOSTS", "8.8.8.8 ssh-ed25519 AAAA"),
        ("GPU_SSH_KNOWN_HOSTS", "fatqat-hai-gpu ssh-ed25519 AAAA"),
        ("GPU_SSH_KNOWN_HOSTS", "fatqat-hai-gpu ssh-ed25519 !!!"),
    ],
)
def test_bad_configuration_fails_before_connecting(connection, field, value, capsys):
    connection.env[field] = value
    assert gpu_ssh.main(["configure"], connection.env) == 1
    assert not connection.calls
    assert "canary" not in capsys.readouterr().err


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.2",
        "224.0.0.1",
        "host;id",
        "",
        "8.8.8.8\n",
        "2606:4700::1%bad\nready=true",
    ],
)
def test_invalid_target_is_never_used_for_ssh(connection, address):
    connection.env["GPU_PUBLIC_IP"] = address
    assert gpu_ssh.main(["identity"], connection.env) == 1
    assert not connection.calls


def test_system_ssh_strictly_pins_host_and_never_forwards_secrets(connection):
    connection.ssh.require("hostname && whoami && nvidia-smi")
    command, kwargs = connection.calls[-1]
    assert command[0:2] == ["ssh", "-T"]
    for option in (
        "BatchMode=yes",
        "IdentitiesOnly=yes",
        "StrictHostKeyChecking=yes",
        "HostKeyAlias=fatqat-hai-gpu",
        "HostKeyAlgorithms=ssh-ed25519",
        "ForwardAgent=no",
        "ConnectTimeout=10",
        "ServerAliveInterval=15",
        "ServerAliveCountMax=3",
    ):
        assert option in command
    assert any(item.startswith("UserKnownHostsFile=") for item in command)
    assert command[-2:] == ["ubuntu@8.8.8.8", "hostname && whoami && nvidia-smi"]
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert not {
        "GH_TOKEN",
        "TENCENTCLOUD_SECRET_ID",
        "TENCENTCLOUD_SECRET_KEY",
        "GPU_SSH_PRIVATE_KEY",
        "SSH_AUTH_SOCK",
    }.intersection(kwargs["env"])
    assert "canary" not in str(command) + str(kwargs)


@pytest.mark.parametrize(
    "error,diagnostic",
    [
        (b"Permission denied (publickey).", "public-key authentication failed"),
        (b"WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!", "host key mismatch"),
        (b"Host key verification failed.", "host key mismatch"),
        (b"no matching host key type found", "host key mismatch"),
        (b'Load key "id_ed25519": invalid format', "public-key authentication failed"),
    ],
)
def test_authentication_and_host_key_failures_are_not_retried(
    connection, error, diagnostic
):
    connection.results.append((255, b"", error))
    with pytest.raises(gpu_ssh.SSHFailure, match=diagnostic):
        connection.ssh.wait()
    assert len(connection.calls) == 1


@pytest.mark.parametrize(
    "failure",
    [
        (255, b"", b"ssh: connect: Connection refused"),
        subprocess.TimeoutExpired(["ssh"], 20, output=b"partial", stderr=b"timeout"),
    ],
)
def test_connection_wait_is_bounded(connection, monkeypatch, failure):
    now = [0.0]
    monkeypatch.setattr(gpu_ssh.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        gpu_ssh.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    connection.results.extend([failure] * 60)
    with pytest.raises(
        gpu_ssh.SSHFailure, match="readiness timed out after 10 minutes"
    ):
        connection.ssh.wait()
    assert len(connection.calls) == (
        20 if isinstance(failure, subprocess.TimeoutExpired) else 60
    )
    assert now[0] == 600


def test_connection_retries_then_succeeds(connection, monkeypatch):
    monkeypatch.setattr(gpu_ssh.time, "sleep", lambda _: None)
    connection.results.extend([(255, b"", b"Connection refused"), (0, b"", b"")])
    connection.ssh.wait()
    assert len(connection.calls) == 2


@pytest.mark.parametrize(
    "code,diagnostic",
    [
        (1, "Remote command/test failed"),
        (124, "deadline exceeded"),
        (137, "deadline exceeded"),
        (255, "TCP/SSH connection failed"),
    ],
)
def test_remote_failure_propagates_to_controller_exit(
    connection, code, diagnostic, capsys
):
    connection.ssh.upload(connection.env)
    connection.results.append((code, b"pytest output", b""))
    assert gpu_ssh.main(["run", "pr"], connection.env) == 1
    assert diagnostic in capsys.readouterr().err


def test_remote_computation_and_client_have_separate_deadlines(connection, monkeypatch):
    connection.ssh.upload(connection.env)
    require = connection.ssh.require

    def require_with_deadline(command, timeout):
        assert timeout == 1860
        return require(command, timeout)

    monkeypatch.setattr(connection.ssh, "require", require_with_deadline)
    connection.ssh.run("pr")
    command, _ = connection.calls[-1]
    assert "timeout --signal=TERM --kill-after=30s 1800s bash" in command[-1]
    assert command[-1].startswith("env -i HOME=/home/ubuntu PATH=/usr/bin:/bin ")


def test_client_timeout_is_failure_and_logs_are_protected(connection, capsys):
    connection.results.append(
        subprocess.TimeoutExpired(
            ["ssh"], 60, output=b"::error::remote", stderr=b"::warning::remote"
        )
    )
    assert gpu_ssh.main(["identity"], connection.env) == 1
    output = capsys.readouterr()
    assert "TCP/SSH operation timed out" in output.err
    assert output.out.count("::stop-commands::") == 1


def test_remote_output_cannot_write_actions_commands_or_environment(connection, capsys):
    connection.results.append(
        (0, b"::error::fake\n::set-env name=BAD::value", b"::add-mask::fake")
    )
    connection.ssh.require("true")
    connection.ssh.require("true")
    output = capsys.readouterr().out
    markers = re.findall(r"::stop-commands::([0-9a-f]{64})", output)
    assert len(markers) == 2 and markers[0] != markers[1]
    assert (
        output.index("::stop-commands::")
        < output.index("::error::fake")
        < output.index(f"::{markers[0]}::")
    )
    assert all(marker not in str(connection.calls) for marker in markers)
    assert not any(
        path.name in {"GITHUB_ENV", "GITHUB_OUTPUT"}
        for path in Path(connection.env["RUNNER_TEMP"]).rglob("*")
    )


def test_upload_transfers_only_archive_and_trusted_driver(connection, capsys):
    connection.ssh.upload(connection.env)
    uploads = [
        (command, kwargs)
        for command, kwargs in connection.calls
        if kwargs["stdin"] != subprocess.DEVNULL
    ]
    assert len(uploads) == 1
    command, kwargs = uploads[0]
    assert all(command[0] == "ssh" for command, _ in connection.calls)
    assert "StrictHostKeyChecking=yes" in command
    assert command[-1].endswith("/payload.tar.gz.part")
    assert kwargs["stdin"].mode == "rb"
    payload = Path(kwargs["stdin"].name)
    assert payload.name == "payload.tar.gz"
    with tarfile.open(payload, "r:gz") as archive:
        assert archive.getnames() == ["source.tar", "run.sh"]
        assert all(member.isfile() for member in archive.getmembers())
        assert (
            archive.extractfile("source.tar").read()
            == (
                Path(connection.env["RUNNER_TEMP"]) / "fatqat-gpu-source.tar"
            ).read_bytes()
        )
        assert archive.extractfile("run.sh").read() == gpu_ssh.REMOTE_SCRIPT.encode()
    assert all("canary" not in str(command) for command, _ in connection.calls)
    assert "_work" not in str(connection.calls)
    output = capsys.readouterr().out
    assert f"Payload size: {payload.stat().st_size} bytes" in output
    assert "Starting payload upload" in output
    assert "Payload upload elapsed:" in output
    assert "canary" not in output


@pytest.mark.parametrize("corrupt", [False, True])
def test_payload_checksum_controls_extraction(
    ssh_environment, tmp_path, monkeypatch, ssh_tools, capsys, corrupt
):
    gpu_ssh.configure(ssh_environment)
    source = tmp_path / "fatqat-gpu-source.tar"
    source.write_bytes(b"opaque source archive\x00\xff\r\n")
    bash = (
        str(Path(ssh_tools).with_name("bash.exe"))
        if os.name == "nt"
        else shutil.which("bash")
    )
    popen = subprocess.Popen
    workspace = tmp_path / "workspace"

    def local_ssh(arguments, **kwargs):
        assert arguments[0] == "ssh"
        remote = (gpu_ssh.local_directory(ssh_environment) / "remote_dir").read_text()
        if "sha256sum" in arguments[-1] and corrupt:
            (workspace / "payload.tar.gz.part").write_bytes(b"corrupted transfer")
        # Run exactly the generated remote shell command, entirely locally.
        command = arguments[-1].replace(remote, "workspace")
        return popen([bash, "-c", command], cwd=tmp_path, **kwargs)

    monkeypatch.setattr(gpu_ssh.subprocess, "Popen", local_ssh)
    assert gpu_ssh.main(["upload"], ssh_environment) == int(corrupt)
    output = capsys.readouterr()
    if corrupt:
        assert "Payload SHA-256 verification failed" in output.out
        assert "Remote command/test failed" in output.err
        assert not (workspace / "source.tar").exists()
        assert not (workspace / "run.sh").exists()
    else:
        assert "Payload SHA-256 verified" in output.out
        assert (workspace / "source.tar").read_bytes() == source.read_bytes()
        assert (workspace / "run.sh").read_bytes() == gpu_ssh.REMOTE_SCRIPT.encode()
    assert not (workspace / "source").exists()


@pytest.mark.parametrize("failed_upload", [0, 1, "timeout", None])
def test_cleanup_targets_recorded_workspace_after_upload(connection, failed_upload):
    if failed_upload is None:
        connection.ssh.upload(connection.env)
    else:
        failure = (
            subprocess.TimeoutExpired(["ssh"], 180)
            if failed_upload == "timeout"
            else (1, b"", b"upload failed")
        )
        connection.results.extend(
            [(0, b"", b"")] * (2 if failed_upload == 1 else 1) + [failure]
        )
        with pytest.raises(
            gpu_ssh.SSHFailure, match="timed out|Remote command/test failed"
        ):
            connection.ssh.upload(connection.env)
        if failed_upload != 1:
            assert len(connection.calls) == 2  # No checksum/extraction after failure.
    assert not any("rm -rf" in command[-1] for command, _ in connection.calls)
    remote = (connection.ssh.directory / "remote_dir").read_text(encoding="utf-8")
    assert gpu_ssh.main(["cleanup"], connection.env) == 0
    assert connection.calls[-1][0][-1] == (
        f"timeout --signal=TERM --kill-after=5s 30s rm -rf -- {remote}"
    )


def test_cleanup_without_a_recorded_directory_does_not_connect(connection):
    assert gpu_ssh.main(["cleanup"], connection.env) == 0
    assert not connection.calls


@pytest.mark.parametrize(
    "remote",
    [
        "/home/ubuntu",
        "/home/ubuntu/fatqat-gpu.*",
        "/home/ubuntu/fatqat-gpu." + "a" * 32 + "/..",
        "/home/ubuntu/fatqat-gpu." + "a" * 32 + "\n",
    ],
)
def test_cleanup_rejects_paths_outside_the_exact_workspace(connection, remote):
    (connection.ssh.directory / "remote_dir").write_text(remote, encoding="utf-8")
    assert gpu_ssh.main(["cleanup"], connection.env) == 1
    assert not connection.calls


@pytest.mark.parametrize(
    "failure",
    [
        (255, b"", b"Connection refused"),
        subprocess.TimeoutExpired(["ssh"], 45),
    ],
)
def test_remote_cleanup_failure_is_reported_and_bounded(connection, failure, capsys):
    connection.ssh.upload(connection.env)
    connection.results.append(failure)
    started = time.monotonic()
    assert gpu_ssh.main(["cleanup"], connection.env) == 1
    assert time.monotonic() - started <= 45.1
    assert "TCP/SSH" in capsys.readouterr().err


def test_remote_driver_preserves_install_preflight_pytest_order_and_exit_status():
    script = gpu_ssh.REMOTE_SCRIPT
    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert (
        script.index("-m venv")
        < script.index("-m pip install . --group test-full")
        < script.index("import cupy")
        < script.index("-m pytest")
    )
    assert '"cupy-cuda12x==13.6.0"' in script
    assert 'if [[ "$mode" == pr ]]' in script
    assert "export CUDA_PATH=/usr/local/cuda-12.0" in script
    assert 'export LD_LIBRARY_PATH="$CUDA_PATH/lib64"' in script
    assert "--confcutdir=tests/gpu_tests -v -rs" in script
    assert "tee" not in script
    assert "GPU pytest did not execute any tests" in script
    assert "GH_TOKEN" not in script and "TENCENTCLOUD" not in script


@pytest.mark.parametrize(
    "devices,result,success",
    [(1, [2.0, 3.0, 4.0], True), (0, [], False), (1, [9.0], False)],
)
def test_independent_preflight_requires_gpu_and_actual_correct_computation(
    monkeypatch, tmp_path, devices, result, success
):
    class Array:
        def __add__(self, other):
            assert other == 1.0
            return self

        def get(self):
            return self

        def tolist(self):
            return result

    class Device:
        def __init__(self, index):
            assert index == 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    cupy = SimpleNamespace(
        show_config=lambda: None,
        float32=float,
        asarray=lambda values, dtype: Array(),
        cuda=SimpleNamespace(
            Device=Device,
            runtime=SimpleNamespace(getDeviceCount=lambda: devices),
            get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None),
        ),
    )
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    preflight = gpu_ssh.REMOTE_SCRIPT.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    path = tmp_path / "preflight.py"
    path.write_text(preflight, encoding="utf-8")
    if success:
        runpy.run_path(str(path))
    else:
        with pytest.raises(RuntimeError):
            runpy.run_path(str(path))


def test_remote_permission_error_is_not_misreported_as_ssh_authentication(connection):
    connection.results.append((1, b"", b"Permission denied writing test data"))
    with pytest.raises(gpu_ssh.SSHFailure, match="Remote command/test failed"):
        connection.ssh.require("test-driver")


@pytest.mark.parametrize(
    "cases,success",
    [
        ("", False),
        ("<testcase><skipped/></testcase>", False),
        ("<testcase/><testcase><skipped/></testcase>", True),
    ],
)
def test_remote_result_requires_at_least_one_executed_test(
    tmp_path, monkeypatch, cases, success
):
    report = tmp_path / "pytest.xml"
    report.write_text(
        f"<testsuites><testsuite>{cases}</testsuite></testsuites>", encoding="utf-8"
    )
    validation = gpu_ssh.REMOTE_SCRIPT.rsplit("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    driver = tmp_path / "check_result.py"
    driver.write_text(validation, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [str(driver), str(report)])
    if success:
        runpy.run_path(str(driver))
    else:
        with pytest.raises(RuntimeError, match="did not execute any tests"):
            runpy.run_path(str(driver))
