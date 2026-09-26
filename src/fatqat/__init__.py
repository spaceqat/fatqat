"""Build quantum programs and run them on FATQAT backends.

Core program, register, parameter, estimator, job, and result types are
available from this namespace. Gates live in :mod:`fatqat.operations`,
gate-level backends in :mod:`fatqat.simulator`, pulse models in
:mod:`fatqat.emulator`, and noise declarations in :mod:`fatqat.noise`.
"""

from . import emulator
from . import errors
from . import multisite
from . import noise
from . import operations
from . import simulator
from . import visualization
from .estimator import Estimator
from .execution import ExecutableProgram
from .job import Job
from .logical_program import LogicalProgram
from .noise import NoiseModel
from .observable import Observable
from .operations import Measurement
from .parameters import Parameter, ParameterVector
from .program import Program
from .registers import (
    ClassicalRegister,
    GridRegister,
    QuantumRegister,
    Register,
    RegisterRef,
    RegisterView,
)
from .resource_layout import DeviceOperand, ResourceLayout
from .result import Result

__version__ = "0.1.0a1"


def __getattr__(name: str):
    if name == "compiler":
        from importlib import import_module

        module = import_module(".compiler", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"compiler"})


__all__ = [
    "operations",
    "simulator",
    "emulator",
    "visualization",
    "errors",
    "multisite",
    "noise",
    "Estimator",
    "ExecutableProgram",
    "NoiseModel",
    "Observable",
    "Program",
    "Measurement",
    "Parameter",
    "ParameterVector",
    "Register",
    "QuantumRegister",
    "ClassicalRegister",
    "RegisterRef",
    "GridRegister",
    "RegisterView",
    "ResourceLayout",
    "DeviceOperand",
    "Job",
    "LogicalProgram",
    "compiler",
    "Result",
]
