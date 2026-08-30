"""Import-layer checks for the reorganized pulse emulator."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys


def test_root_emulator_import_does_not_eagerly_import_solver_adapters():
    script = """
import sys
import fatqat.emulator
assert 'fatqat.emulator._atom_3level' not in sys.modules
assert 'fatqat.emulator._atom_3level.qutip_adapter' not in sys.modules
assert 'fatqat.emulator.atom_2level.qutip_adapter' not in sys.modules
assert 'fatqat.emulator.superconducting.qutip_adapter' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


def test_removed_atom_package_paths_do_not_resolve():
    script = """
import importlib.util

assert importlib.util.find_spec('fatqat.emulator.' + 'atom') is None
assert importlib.util.find_spec('fatqat.emulator.' + 'atom_3level') is None
try:
    importlib.util.find_spec('fatqat.emulator.' + 'atom' + '.analog')
except ModuleNotFoundError:
    pass
else:
    raise AssertionError('removed nested atom package unexpectedly resolved')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


def test_removed_runtime_and_model_contract_modules_do_not_resolve():
    script = """
import importlib.util
import sys
import fatqat.emulator

removed = (
    'fatqat.emulator._core.' + 'model_' + 'contract',
    'fatqat.emulator.atom_2level.' + 'runtime',
    'fatqat.emulator._atom_3level.' + 'runtime',
)
for name in removed:
    assert name not in sys.modules
    assert importlib.util.find_spec(name) is None
"""
    result = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


def test_core_never_imports_a_family_or_qutip():
    core = Path(__file__).parents[3] / "src" / "fatqat" / "emulator" / "_core"
    forbidden = ("atom", "superconducting", "qutip")

    for path in core.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(
                token in name.split(".") for name in names for token in forbidden
            ), path


def test_shared_model_document_has_no_family_registry_or_solver_surface():
    path = (
        Path(__file__).parents[3]
        / "src"
        / "fatqat"
        / "emulator"
        / "_core"
        / "model_document.py"
    )
    source = path.read_text(encoding="utf-8")

    assert "_SC_" not in source
    assert "_DIGITAL" + "_" not in source
    assert "_ANALOG" + "_" not in source
    assert "qutip" not in source.lower()


def test_shared_pulse_values_have_no_emulator_or_solver_dependency():
    path = Path(__file__).parents[3] / "src" / "fatqat" / "_pulse_values.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden = ("emulator", "atom", "superconducting", "engine", "noise", "qutip")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        assert not any(
            token in name.split(".") for name in names for token in forbidden
        ), path
