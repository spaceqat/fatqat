"""OpenQASM source boundary for compiler pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..errors import ValidationError


@dataclass(frozen=True, slots=True)
class QasmSource:
    """OpenQASM text before semantic parsing."""

    IR_ID: ClassVar[str] = "qasm.text.v1"

    text: str
    filename: str | None = None


def verify_qasm_source(program: object) -> None:
    if type(program) is not QasmSource:
        raise ValidationError("expected QasmSource")
    if not isinstance(program.text, str) or not program.text.strip():
        raise ValidationError("QASM source must be a non-empty string")
    if program.filename is not None and not isinstance(program.filename, str):
        raise ValidationError("QASM filename must be a string or None")
