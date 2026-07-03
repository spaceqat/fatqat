"""Public package surface for qnsim."""

from . import backends
from . import operations as ops
from .backends import StateVectorBackend
from .errors import (
    BackendValidationError,
    MatrixImplementationError,
    NoMeasurementWarning,
    QnsimError,
    ResultFieldUnavailableError,
    UnsupportedOperationError,
)
from .job import Job
from .program import AppliedOperation, Measurement, Program
from .registers import (
    ClassicalRegister,
    QuantumRegister,
    Register,
    RegisterRef,
)
from .result import Result

__version__ = "0.0.1"

__all__ = [
    "ops",
    "backends",
    "Program",
    "AppliedOperation",
    "Measurement",
    "Register",
    "QuantumRegister",
    "ClassicalRegister",
    "RegisterRef",
    "StateVectorBackend",
    "Job",
    "Result",
    "QnsimError",
    "BackendValidationError",
    "UnsupportedOperationError",
    "MatrixImplementationError",
    "ResultFieldUnavailableError",
    "NoMeasurementWarning",
]
