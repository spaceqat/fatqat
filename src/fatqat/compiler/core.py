"""Typed passes and explicit pipelines for the FatQat compiler."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Generic, Mapping, Protocol, TypeVar

from .errors import (
    EmitNotFoundError,
    PassError,
    PipelineNotFoundError,
    ValidationError,
)
from .ir import IRRegistry, Verifier

SourceT = TypeVar("SourceT")
TargetT = TypeVar("TargetT")


@dataclass(frozen=True, slots=True)
class CompileContext:
    """Per-compilation target and options passed through the explicit pipeline."""

    target: object | None = None
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


class TranslationPass(Protocol[SourceT, TargetT]):
    """A typed transformation between two registered top-level IRs."""

    name: str
    source_type: type[SourceT]
    target_type: type[TargetT]

    def run(self, source: SourceT, context: CompileContext) -> TargetT: ...


@dataclass(frozen=True, slots=True)
class Pipeline:
    """An explicitly ordered pass sequence selected by name."""

    name: str
    passes: tuple[TranslationPass, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("pipeline name must be a non-empty string")
        if not self.passes:
            raise ValueError("pipeline must contain at least one pass")
        for left, right in zip(self.passes, self.passes[1:]):
            if left.target_type is not right.source_type:
                raise ValueError(
                    "incompatible passes: "
                    f"{left.name} produces {left.target_type.__name__}, "
                    f"but {right.name} consumes {right.source_type.__name__}"
                )

    @property
    def source_type(self) -> type:
        return self.passes[0].source_type

    @property
    def target_type(self) -> type:
        return self.passes[-1].target_type

    @property
    def boundary_types(self) -> tuple[type, ...]:
        return (self.source_type,) + tuple(item.target_type for item in self.passes)


@dataclass(frozen=True, slots=True)
class CompilationResult(Generic[TargetT]):
    """The output program plus the pass names actually executed."""

    output: TargetT
    route: tuple[str, ...]


class Compiler:
    """Registry and executor for a small set of explicit pipelines."""

    def __init__(self) -> None:
        self.ir = IRRegistry()
        self._pipelines: dict[str, Pipeline] = {}

    def register_ir(self, program_type: type, verifier: Verifier) -> None:
        self.ir.register(program_type, verifier)

    def register_pipeline(self, pipeline: Pipeline) -> None:
        if pipeline.name in self._pipelines:
            raise ValueError(f"pipeline already registered: {pipeline.name}")
        for program_type in pipeline.boundary_types:
            try:
                self.ir.for_type(program_type)
            except KeyError as exc:
                raise ValueError(
                    f"pipeline {pipeline.name!r} uses unregistered IR type "
                    f"{program_type.__name__}"
                ) from exc
        self._pipelines[pipeline.name] = pipeline

    def compile(
        self,
        source: object,
        *,
        pipeline: str,
        emit: str | None = None,
        context: CompileContext | None = None,
    ) -> CompilationResult:
        try:
            selected = self._pipelines[pipeline]
        except KeyError as exc:
            raise PipelineNotFoundError(f"unknown pipeline: {pipeline}") from exc

        if type(source) is not selected.source_type:
            raise ValidationError(
                f"pipeline {pipeline!r} expects {selected.source_type.__name__}, "
                f"got {type(source).__name__}"
            )

        boundaries = {item.IR_ID for item in selected.boundary_types}
        target_ir = selected.target_type.IR_ID if emit is None else emit
        if target_ir not in boundaries:
            raise EmitNotFoundError(
                f"emit {target_ir!r} is not a boundary of pipeline {pipeline!r}"
            )

        current = source
        route: list[str] = []
        self._verify(current)
        if type(current).IR_ID == target_ir:
            return CompilationResult(current, ())

        compile_context = context or CompileContext()
        for translation in selected.passes:
            try:
                output = translation.run(current, compile_context)
            except Exception as exc:
                raise PassError(
                    translation.name,
                    translation.source_type.IR_ID,
                    translation.target_type.IR_ID,
                    str(exc),
                ) from exc
            if type(output) is not translation.target_type:
                raise PassError(
                    translation.name,
                    translation.source_type.IR_ID,
                    translation.target_type.IR_ID,
                    f"returned {type(output).__name__}, expected "
                    f"{translation.target_type.__name__}",
                )
            self._verify(output)
            current = output
            route.append(translation.name)
            if type(current).IR_ID == target_ir:
                return CompilationResult(current, tuple(route))

        raise AssertionError("validated emit boundary was not reached")

    def _verify(self, program: object) -> None:
        try:
            definition = self.ir.for_program(program)
        except KeyError as exc:
            raise ValidationError(
                f"unregistered IR type: {type(program).__name__}"
            ) from exc
        definition.verifier(program)
