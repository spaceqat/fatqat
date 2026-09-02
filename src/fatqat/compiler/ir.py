"""Registration of immutable top-level compiler IR types and validators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar, Protocol


class IRProgram(Protocol):
    """Structural contract shared by every public top-level compiler IR."""

    IR_ID: ClassVar[str]


Verifier = Callable[[object], None]


@dataclass(frozen=True, slots=True)
class IRDefinition:
    """A top-level IR type paired with its self-contained validator."""

    program_type: type
    verifier: Verifier

    @property
    def ir_id(self) -> str:
        return self.program_type.IR_ID


class IRRegistry:
    """Exact-type lookup for IR identities and validators; not a route graph."""

    def __init__(self) -> None:
        self._by_id: dict[str, IRDefinition] = {}
        self._by_type: dict[type, IRDefinition] = {}

    def register(self, program_type: type, verifier: Verifier) -> None:
        if program_type in self._by_type:
            raise ValueError(
                f"program type already registered: {program_type.__name__}"
            )
        ir_id = getattr(program_type, "IR_ID", None)
        if not isinstance(ir_id, str) or not ir_id:
            raise ValueError("program type must declare a non-empty IR_ID")
        if ir_id in self._by_id:
            raise ValueError(f"duplicate IR ID: {ir_id}")
        definition = IRDefinition(program_type, verifier)
        self._by_id[ir_id] = definition
        self._by_type[program_type] = definition

    def for_id(self, ir_id: str) -> IRDefinition:
        return self._by_id[ir_id]

    def for_type(self, program_type: type) -> IRDefinition:
        return self._by_type[program_type]

    def for_program(self, program: object) -> IRDefinition:
        return self.for_type(type(program))
