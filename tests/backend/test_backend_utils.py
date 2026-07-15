from fatqat.backends.backend_utils import _PlanFacts
from fatqat.backends.engine_contract import (
    _DensityMatrixResultRequest,
    _StateVectorResultRequest,
)
from fatqat.backends.simulator_backend import _resolve_result_request
from fatqat.result import _DensityMatrixResultConfig, _StateVectorResultConfig


def test_resolve_result_request_defaults_statevector_for_nonstochastic_program():
    request = _resolve_result_request(
        _StateVectorResultConfig(counts=None, statevector=None),
        _PlanFacts(has_measurement=False, has_reset=False),
        _StateVectorResultRequest,
        "statevector",
        reset_is_stochastic=True,
    )

    assert request.counts is False
    assert request.statevector is True


def test_resolve_result_request_reset_suppresses_statevector_default():
    request = _resolve_result_request(
        _StateVectorResultConfig(counts=None, statevector=None),
        _PlanFacts(has_measurement=False, has_reset=True),
        _StateVectorResultRequest,
        "statevector",
        reset_is_stochastic=True,
    )

    assert request.statevector is False


def test_resolve_result_request_reset_keeps_density_matrix_default():
    request = _resolve_result_request(
        _DensityMatrixResultConfig(counts=None, density_matrix=None),
        _PlanFacts(has_measurement=False, has_reset=True),
        _DensityMatrixResultRequest,
        "density_matrix",
        reset_is_stochastic=False,
    )

    assert request.counts is False
    assert request.density_matrix is True
