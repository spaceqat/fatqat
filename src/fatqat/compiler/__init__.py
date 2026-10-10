"""FatQat's lightweight, Python-native compiler framework."""

from .._sc_target import SCTarget
from .core import (
    CompilationResult,
    CompileContext,
    Compiler,
    ExecutableCompilationResult,
    Pipeline,
    TranslationPass,
)
from .errors import (
    CompilerError,
    EmitNotFoundError,
    PassError,
    PipelineNotFoundError,
    UnsupportedFeatureError,
    ValidationError,
)
from .ir import IRDefinition, IRProgram, IRRegistry
from .pipelines import (
    LOGICAL_NA_PIPELINE,
    LOGICAL_SC_PIPELINE,
    NA_PIPELINE,
    SC_PIPELINE,
    compile_to_na,
    compile_to_sc,
    compile_qasm_to_na,
    compile_qasm_to_sc,
    create_na_pipeline,
    create_sc_pipeline,
)
from .simulator_bridge import to_na_simulator_program, to_sc_simulator_program
from .visualization import create_na_animation, save_na_animation

__all__ = [
    "SCTarget",
    "CompilationResult",
    "ExecutableCompilationResult",
    "CompileContext",
    "Compiler",
    "Pipeline",
    "TranslationPass",
    "CompilerError",
    "EmitNotFoundError",
    "PassError",
    "PipelineNotFoundError",
    "UnsupportedFeatureError",
    "ValidationError",
    "IRDefinition",
    "IRProgram",
    "IRRegistry",
    "NA_PIPELINE",
    "SC_PIPELINE",
    "LOGICAL_NA_PIPELINE",
    "LOGICAL_SC_PIPELINE",
    "compile_to_na",
    "compile_to_sc",
    "compile_qasm_to_na",
    "compile_qasm_to_sc",
    "create_na_pipeline",
    "create_sc_pipeline",
    "to_sc_simulator_program",
    "to_na_simulator_program",
    "create_na_animation",
    "save_na_animation",
]
