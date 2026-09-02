from dataclasses import dataclass
from typing import ClassVar

import pytest

from fatqat.compiler import (
    CompilationResult,
    Compiler,
    EmitNotFoundError,
    IRRegistry,
    PassError,
    Pipeline,
    PipelineNotFoundError,
    ValidationError,
)


@dataclass(frozen=True)
class Source:
    IR_ID: ClassVar[str] = "test.source.v1"
    value: int


@dataclass(frozen=True)
class Middle:
    IR_ID: ClassVar[str] = "test.middle.v1"
    value: int


@dataclass(frozen=True)
class Target:
    IR_ID: ClassVar[str] = "test.target.v1"
    value: int


@dataclass(frozen=True)
class DuplicateId:
    IR_ID: ClassVar[str] = "test.source.v1"
    value: int


class SourceToMiddle:
    name = "source-to-middle"
    source_type = Source
    target_type = Middle

    def run(self, source, context):
        return Middle(source.value + 1)


class MiddleToTarget:
    name = "middle-to-target"
    source_type = Middle
    target_type = Target

    def run(self, source, context):
        return Target(source.value * 2)


class SourceToTarget:
    name = "source-to-target"
    source_type = Source
    target_type = Target

    def run(self, source, context):
        return Target(source.value)


class BrokenPass:
    name = "broken"
    source_type = Source
    target_type = Middle

    def run(self, source, context):
        raise RuntimeError("boom")


def _positive(program):
    if program.value < 0:
        raise ValidationError("value must be non-negative")


def _compiler():
    compiler = Compiler()
    compiler.register_ir(Source, _positive)
    compiler.register_ir(Middle, _positive)
    compiler.register_ir(Target, _positive)
    compiler.register_pipeline(Pipeline("test", (SourceToMiddle(), MiddleToTarget())))
    return compiler


def test_ir_registry_rejects_duplicate_id_and_type():
    registry = IRRegistry()
    registry.register(Source, _positive)

    with pytest.raises(ValueError, match="duplicate IR ID"):
        registry.register(DuplicateId, _positive)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(Source, _positive)


def test_pipeline_rejects_incompatible_adjacent_passes():
    with pytest.raises(ValueError, match="incompatible passes"):
        Pipeline("broken", (SourceToTarget(), MiddleToTarget()))


def test_pipeline_runs_in_order_and_validates_boundaries():
    result = _compiler().compile(Source(2), pipeline="test")

    assert result == CompilationResult(
        Target(6), ("source-to-middle", "middle-to-target")
    )


def test_pipeline_validates_each_ir_boundary_once():
    validation_counts = {Source: 0, Middle: 0, Target: 0}

    def count(program):
        validation_counts[type(program)] += 1

    compiler = Compiler()
    compiler.register_ir(Source, count)
    compiler.register_ir(Middle, count)
    compiler.register_ir(Target, count)
    compiler.register_pipeline(
        Pipeline("counted", (SourceToMiddle(), MiddleToTarget()))
    )

    compiler.compile(Source(2), pipeline="counted")

    assert validation_counts == {Source: 1, Middle: 1, Target: 1}


def test_emit_stops_at_registered_pipeline_boundary():
    result = _compiler().compile(Source(2), pipeline="test", emit=Middle.IR_ID)

    assert result == CompilationResult(Middle(3), ("source-to-middle",))


def test_emit_can_return_the_input_without_running_a_pass():
    source = Source(2)
    result = _compiler().compile(source, pipeline="test", emit=Source.IR_ID)

    assert result == CompilationResult(source, ())


def test_invalid_emit_and_unknown_pipeline_fail_explicitly():
    compiler = _compiler()

    with pytest.raises(EmitNotFoundError, match="test.unknown.v1"):
        compiler.compile(Source(1), pipeline="test", emit="test.unknown.v1")
    with pytest.raises(PipelineNotFoundError, match="missing"):
        compiler.compile(Source(1), pipeline="missing")


def test_pass_exceptions_keep_the_original_cause():
    compiler = Compiler()
    compiler.register_ir(Source, _positive)
    compiler.register_ir(Middle, _positive)
    compiler.register_pipeline(Pipeline("broken", (BrokenPass(),)))

    with pytest.raises(PassError, match="broken") as caught:
        compiler.compile(Source(1), pipeline="broken")

    assert isinstance(caught.value.__cause__, RuntimeError)


def test_output_validator_rejects_invalid_pass_result():
    compiler = _compiler()

    with pytest.raises(ValidationError, match="non-negative"):
        compiler.compile(Source(-1), pipeline="test")
