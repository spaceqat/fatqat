"""Errors raised by FatQat's lightweight compiler framework."""

from __future__ import annotations


class CompilerError(Exception):
    """Base class for compiler-specific failures."""


class ValidationError(CompilerError, ValueError):
    """A program does not satisfy its current IR contract."""


class PipelineNotFoundError(CompilerError, LookupError):
    """The requested explicit pipeline has not been registered."""


class EmitNotFoundError(CompilerError, LookupError):
    """The requested emit ID is not a boundary in the selected pipeline."""


class UnsupportedFeatureError(CompilerError, ValueError):
    """The input uses semantics intentionally unsupported by this compiler version."""


class PassError(CompilerError):
    """A compiler pass raised an exception or returned the wrong type.

    If the pass raised, ``__cause__`` holds the original exception. A wrong
    return type is reported directly and has no underlying exception.

    Attributes:
        pass_name: Name of the failed pass.
        source_ir: IR ID the pass accepts.
        target_ir: IR ID the pass is expected to produce.
    """

    def __init__(self, pass_name: str, source_ir: str, target_ir: str, detail: str):
        self.pass_name = pass_name
        self.source_ir = source_ir
        self.target_ir = target_ir
        super().__init__(
            f"pass {pass_name!r} failed ({source_ir} -> {target_ir}): {detail}"
        )
