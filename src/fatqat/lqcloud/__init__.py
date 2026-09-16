"""Optional QEC17 integration for the LQCloud circuit API.

This namespace is importable without the third-party ``lqcloud`` package.
Constructing a backend or converting a program loads that dependency lazily.
"""

from __future__ import annotations

from .backend import LQCloudQEC17Backend
from .converter import program_to_qec17_circuit

__all__ = ["LQCloudQEC17Backend", "program_to_qec17_circuit"]
