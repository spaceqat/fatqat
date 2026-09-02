"""Release checks for packaged internal ZAP resources."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import zipfile

PROJECT_ROOT = Path(__file__).parents[2]
PROFILE_NAMES = ("default", "scale_to_100", "scale_to_500")
_PROFILE_ORIGIN_GUARD_CODE = """
from importlib import resources
import os
from pathlib import Path

import fatqat
import fatqat.compiler.algorithms.zap as zap

install_dir = Path(os.environ["FATQAT_INSTALL_DIR"]).resolve()


def assert_installed_origin(origin, label):
    resolved = Path(origin).resolve()
    assert resolved.is_relative_to(install_dir), (
        f"{label} origin {resolved} is outside isolated target {install_dir}"
    )


assert_installed_origin(fatqat.__file__, "fatqat package")
assert_installed_origin(zap.__file__, "internal ZAP module")

for name in ("default", "scale_to_100", "scale_to_500"):
    resource = resources.files(zap).joinpath("architectures", f"{name}.json")
    with resources.as_file(resource) as resource_path:
        assert_installed_origin(resource_path, f"{name} profile resource")
    assert zap.load_architecture(name)

"""


def _build_wheel(wheel_dir: Path) -> Path:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    return next(wheel_dir.glob("fatqat-*.whl"))


def test_wheel_contains_internal_zap_license_and_profiles(tmp_path):
    wheel_path = _build_wheel(tmp_path)

    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())

    assert "fatqat/compiler/algorithms/zap/LICENSE" in names
    for profile in PROFILE_NAMES:
        assert f"fatqat/compiler/algorithms/zap/architectures/{profile}.json" in names


def test_installed_wheel_loads_all_zap_profiles(tmp_path):
    wheel_dir = tmp_path / "wheel"
    install_dir = tmp_path / "install"
    wheel_dir.mkdir()
    wheel_path = _build_wheel(wheel_dir)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(install_dir),
            str(wheel_path),
        ],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-c", _PROFILE_ORIGIN_GUARD_CODE],
        check=True,
        cwd=tmp_path,
        env=_isolated_target_env(install_dir),
    )


def test_profile_origin_guard_rejects_worktree_source_fallback(tmp_path):
    missing_install_dir = tmp_path / "missing-install"

    result = subprocess.run(
        [sys.executable, "-c", _PROFILE_ORIGIN_GUARD_CODE],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        env=_isolated_target_env(missing_install_dir),
        text=True,
    )

    assert result.returncode != 0, "source fallback escaped the install origin guard"
    assert "fatqat package origin" in result.stderr
    assert "outside isolated target" in result.stderr


def _isolated_target_env(install_dir: Path) -> dict[str, str]:
    return {
        **os.environ,
        "FATQAT_INSTALL_DIR": str(install_dir),
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(install_dir),
    }
