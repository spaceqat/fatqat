"""Build quantum programs and run them on FATQAT backends.

Core program, register, parameter, estimator, job, and result types are
available from this namespace. Gates live in :mod:`fatqat.operations`,
gate-level backends in :mod:`fatqat.simulator`, pulse models in
:mod:`fatqat.emulator`, and noise declarations in :mod:`fatqat.noise`.
"""

from . import emulator
from . import errors
from . import noise
from . import operations
from . import simulator
from .estimator import Estimator
from .job import Job
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

__version__ = "0.0.1"

__all__ = [
    "operations",
    "simulator",
    "emulator",
    "errors",
    "noise",
    "Estimator",
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
    "Result",
]
