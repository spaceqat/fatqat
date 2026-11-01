"""Use system OpenSSH to run the trusted GPU test driver on the existing HAI."""

import base64
import binascii
import codecs
import hashlib
import ipaddress
import os
from pathlib import Path
import re
import secrets
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time

# Uploaded from the trusted control checkout, never from the source archive.
REMOTE_SCRIPT = r"""#!/usr/bin/env bash
set -euo pipefail
mode="$1"
readonly work="$2"
[[ "$work" =~ ^/home/ubuntu/fatqat-gpu\.[0-9a-f]{32}$ ]]
trap 'rm -rf -- "$work"' EXIT
export CUDA_PATH=/usr/local/cuda-12.0
export PATH="$CUDA_PATH/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64"
export PYTHONUNBUFFERED=1
mkdir "$work/source"
tar -xf "$work/source.tar" -C "$work/source"
cd "$work/source"
PYTHON=/home/ubuntu/fatqat-ci-venv/bin/python
if [[ "$mode" == pr ]]; then
    "$PYTHON" -m venv "$work/venv"
    PYTHON="$work/venv/bin/python"
    "$PYTHON" -m pip install --upgrade "pip>=25.1"
    "$PYTHON" -m pip install . --group test-full "cupy-cuda12x==13.6.0"
elif [[ "$mode" != manual ]]; then
    exit 2
fi
"$PYTHON" --version
nvidia-smi
"$PYTHON" -I - <<'PY'
import cupy as cp

cp.show_config()
device_count = cp.cuda.runtime.getDeviceCount()
if device_count < 1:
    raise RuntimeError("GPU CI requires an available CUDA GPU")
with cp.cuda.Device(0):
    probe = cp.asarray([1.0, 2.0, 3.0], dtype=cp.float32)
    result_gpu = probe + 1.0
    cp.cuda.get_current_stream().synchronize()
    result = result_gpu.get().tolist()
if result != [2.0, 3.0, 4.0]:
    raise RuntimeError(f"CUDA preflight returned an incorrect result: {result}")
print(f"CUDA_PREFLIGHT_PASSED: {device_count} device(s)")
PY
export PYTEST_ADDOPTS="--junitxml=$work/pytest.xml"
"$PYTHON" -m pytest tests/gpu_tests --confcutdir=tests/gpu_tests -v -rs
/home/ubuntu/fatqat-ci-venv/bin/python -I - "$work/pytest.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET

cases = list(ET.parse(sys.argv[1]).iter("testcase"))
if not any(case.find("skipped") is None for case in cases):
    raise RuntimeError("GPU pytest did not execute any tests")
PY
"""


class SSHFailure(RuntimeError):
    """An OpenSSH operation failed, with a classified diagnostic."""


def local_directory(environment):
    return Path(environment["RUNNER_TEMP"]) / "fatqat-gpu-ssh"


def connection_settings(environment):
    user = environment["GPU_SSH_USER"]
    port = environment["GPU_SSH_PORT"]
    if not re.fullmatch(r"[a-z_][a-z0-9_-]*", user):
        raise ValueError("Invalid GPU_SSH_USER")
    if not re.fullmatch(r"[0-9]{1,5}", port) or not 1 <= int(port) <= 65535:
        raise ValueError("Invalid GPU_SSH_PORT")
    return user, port


def process_environment(environment):
    """Give local OpenSSH tools no cloud credentials, tokens or agent socket."""
    return {
        key: environment[key]
        for key in ("PATH", "SYSTEMROOT", "TEMP", "TMP")
        if key in environment
    }


def configure(environment):
    """Check all configuration before startup and write only local SSH files."""
    for name in (
        "TENCENTCLOUD_SECRET_ID",
        "TENCENTCLOUD_SECRET_KEY",
        "HAI_REGION",
        "HAI_INSTANCE_ID",
        "GPU_SSH_PRIVATE_KEY",
        "GPU_SSH_KNOWN_HOSTS",
        "GPU_SSH_USER",
        "GPU_SSH_PORT",
    ):
        if not environment.get(name, "").strip():
            raise ValueError(f"Missing configuration: {name}")
    connection_settings(environment)
    known_hosts = environment["GPU_SSH_KNOWN_HOSTS"].strip().split()
    if len(known_hosts) != 3 or known_hosts[:2] != ["fatqat-hai-gpu", "ssh-ed25519"]:
        raise ValueError(
            "GPU_SSH_KNOWN_HOSTS must contain one fatqat-hai-gpu ed25519 key"
        )
    try:
        key = base64.b64decode(known_hosts[2], validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("GPU_SSH_KNOWN_HOSTS has invalid key encoding") from None
    if len(key) != 51 or not key.startswith(
        b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20"
    ):
        raise ValueError("GPU_SSH_KNOWN_HOSTS has an invalid ed25519 key")
    directory = local_directory(environment)
    directory.mkdir(mode=0o700)
    private_key = (
        environment["GPU_SSH_PRIVATE_KEY"].replace("\r\n", "\n").replace("\r", "\n")
    )
    for filename, content in (
        ("id_ed25519", private_key),
        ("known_hosts", " ".join(known_hosts)),
    ):
        path = directory / filename
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            path.chmod(0o600)
            stream.write(content.rstrip("\n") + "\n")
    try:
        result = subprocess.run(
            ["ssh-keygen", "-y", "-P", "", "-f", str(directory / "id_ed25519")],
            env=process_environment(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise ValueError("GPU_SSH_PRIVATE_KEY validation timed out") from None
    if result.returncode:
        raise ValueError("GPU_SSH_PRIVATE_KEY is invalid or requires a passphrase")


class SSH:
    """A single configured host, without forwarded credentials or user SSH config."""

    def __init__(self, environment):
        user, port = connection_settings(environment)
        address = ipaddress.ip_address(environment["GPU_PUBLIC_IP"])
        if (
            not address.is_global
            or address.is_multicast
            or getattr(address, "scope_id", None)
        ):
            raise ValueError("HAI did not supply a public IP")
        self.host = f"{user}@{address}"
        self.directory = local_directory(environment)
        self.port = port
        self.options = [
            "-F",
            os.devnull,
            "-i",
            str(self.directory / "id_ed25519"),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "HostKeyAlias=fatqat-hai-gpu",
            "-o",
            "HostKeyAlgorithms=ssh-ed25519",
            "-o",
            f"UserKnownHostsFile={self.directory / 'known_hosts'}",
            "-o",
            f"GlobalKnownHostsFile={os.devnull}",
            "-o",
            "ForwardAgent=no",
            "-o",
            "ClearAllForwardings=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ConnectionAttempts=1",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "LogLevel=ERROR",
        ]
        self.environment = process_environment(environment)

    def execute(self, arguments, timeout, stdin=subprocess.DEVNULL):
        """Stream both output channels; retain only an 8 KiB diagnostic tail."""
        marker = secrets.token_hex(32)  # Local only; never sent to the process.
        print(f"::stop-commands::{marker}", flush=True)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        tail = b""
        try:
            # Separate handles avoid sharing the child's file offset. Disk spooling
            # keeps memory bounded, including a long line or a silent/hung client.
            with tempfile.TemporaryDirectory(
                prefix="log-", dir=self.directory
            ) as directory:
                log = Path(directory) / "output"
                with (
                    log.open("wb", buffering=0) as writer,
                    log.open("rb", buffering=0) as reader,
                ):
                    with subprocess.Popen(
                        arguments,
                        env=self.environment,
                        stdin=stdin,
                        stdout=writer,
                        stderr=subprocess.STDOUT,
                    ) as process:
                        deadline = time.monotonic() + timeout
                        try:
                            while True:
                                if time.monotonic() >= deadline:
                                    raise SSHFailure("TCP/SSH operation timed out")
                                finished = process.poll() is not None
                                chunk = reader.read(8192)
                                if chunk:
                                    tail = (tail + chunk)[-8192:]
                                    sys.stdout.write(decoder.decode(chunk))
                                    sys.stdout.flush()
                                elif finished:
                                    break
                                else:
                                    time.sleep(0.05)
                        finally:
                            if process.poll() is None:
                                process.kill()
                            process.wait()
                        result = process.returncode
        finally:
            sys.stdout.write(decoder.decode(b"", final=True))
            print(f"\n::{marker}::", flush=True)
        diagnostic = tail.decode("utf-8", errors="replace").lower()
        if result == 255 and any(
            text in diagnostic
            for text in (
                "host key verification failed",
                "remote host identification has changed",
                "no matching host key type",
            )
        ):
            raise SSHFailure("SSH host key mismatch or untrusted host key")
        if result == 255 and any(
            text in diagnostic
            for text in (
                "permission denied",
                "load key",
                "error in libcrypto",
            )
        ):
            raise SSHFailure("SSH public-key authentication failed")
        return result

    def command(self, command, timeout=60, stdin=subprocess.DEVNULL):
        return self.execute(
            ["ssh", "-T", *self.options, "-p", self.port, self.host, command],
            timeout,
            stdin=stdin,
        )

    def require(self, command, timeout=60, stdin=subprocess.DEVNULL):
        result = self.command(command, timeout, stdin=stdin)
        if result == 255:
            raise SSHFailure("TCP/SSH connection failed")
        if result in {124, 137}:
            raise SSHFailure("Remote GPU test deadline exceeded")
        if result:
            raise SSHFailure(f"Remote command/test failed (exit {result})")

    def wait(self):
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            try:
                result = self.command(
                    "true", timeout=min(20, deadline - time.monotonic())
                )
            except SSHFailure as exc:
                if str(exc) != "TCP/SSH operation timed out":
                    raise
            else:
                if result == 0:
                    return
                if result != 255:
                    raise SSHFailure(f"SSH readiness command failed (exit {result})")
            print("Waiting for TCP/SSH readiness...", flush=True)
            time.sleep(min(10, max(0, deadline - time.monotonic())))
        raise SSHFailure("TCP/SSH readiness timed out after 10 minutes")

    def upload(self, environment):
        remote = f"/home/ubuntu/fatqat-gpu.{secrets.token_hex(16)}"
        # Only this locally generated directory is saved; no remote output is trusted.
        (self.directory / "remote_dir").write_text(remote, encoding="utf-8")
        driver = self.directory / "run.sh"
        driver.write_text(REMOTE_SCRIPT, encoding="utf-8", newline="\n")
        source = Path(environment["RUNNER_TEMP"]) / "fatqat-gpu-source.tar"
        payload = self.directory / "payload.tar.gz"
        with tarfile.open(payload, "w:gz") as archive:
            archive.add(source, arcname="source.tar", recursive=False)
            archive.add(driver, arcname="run.sh", recursive=False)
        with payload.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        print(f"Payload size: {payload.stat().st_size} bytes", flush=True)
        self.require(f"umask 077 && mkdir -- {shlex.quote(remote)}")
        print("Starting payload upload over SSH stdin", flush=True)
        started = time.monotonic()
        try:
            with payload.open("rb") as stream:
                self.require(
                    "umask 077 && timeout --signal=TERM --kill-after=5s 170s "
                    f"cat > {shlex.quote(remote)}/payload.tar.gz.part",
                    timeout=180,
                    stdin=stream,
                )
        finally:
            print(
                f"Payload upload elapsed: {time.monotonic() - started:.1f}s", flush=True
            )
        verify_and_extract = (
            f"set -e; cd -- {shlex.quote(remote)}; "
            f"printf '%s\\n' '{digest}  payload.tar.gz.part' | "
            "sha256sum --check --status || "
            "{ echo 'Payload SHA-256 verification failed' >&2; exit 1; }; "
            "echo 'Payload SHA-256 verified'; "
            "tar -xzf payload.tar.gz.part -- source.tar run.sh"
        )
        self.require(
            "timeout --signal=TERM --kill-after=5s 50s sh -c "
            + shlex.quote(verify_and_extract),
            timeout=60,
        )

    def cleanup(self):
        record = self.directory / "remote_dir"
        if not record.exists():
            return
        remote = record.read_text(encoding="utf-8")
        if not re.fullmatch(r"/home/ubuntu/fatqat-gpu\.[0-9a-f]{32}", remote):
            raise ValueError("Invalid remote working directory")
        self.require(
            f"timeout --signal=TERM --kill-after=5s 30s rm -rf -- {shlex.quote(remote)}",
            timeout=45,
        )

    def run(self, mode):
        if mode not in {"pr", "manual"}:
            raise ValueError("Invalid GPU mode")
        remote = (self.directory / "remote_dir").read_text(encoding="utf-8")
        if not re.fullmatch(r"/home/ubuntu/fatqat-gpu\.[0-9a-f]{32}", remote):
            raise ValueError("Invalid remote working directory")
        self.require(
            "env -i HOME=/home/ubuntu PATH=/usr/bin:/bin "
            "timeout --signal=TERM --kill-after=30s 1800s "
            f"bash {shlex.quote(remote)}/run.sh {mode} {shlex.quote(remote)}",
            timeout=1860,
        )


def main(arguments=None, environment=None):
    arguments = sys.argv[1:] if arguments is None else arguments
    environment = os.environ if environment is None else environment
    try:
        if arguments == ["configure"]:
            configure(environment)
            return 0
        ssh = SSH(environment)
        if arguments == ["wait"]:
            ssh.wait()
        elif arguments == ["identity"]:
            ssh.require("hostname && whoami && nvidia-smi")
        elif arguments == ["upload"]:
            ssh.upload(environment)
        elif arguments == ["cleanup"]:
            ssh.cleanup()
        elif len(arguments) == 2 and arguments[0] == "run":
            ssh.run(arguments[1])
        else:
            raise ValueError(
                "Usage: gpu_ssh.py configure|wait|identity|upload|cleanup|run pr|manual"
            )
    except (SSHFailure, ValueError, OSError) as exc:
        print(f"GPU SSH failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
