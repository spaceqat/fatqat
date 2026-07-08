from fatqat.backends.backend_utils import _PlanFacts, _resolve_result_request
from fatqat.result import _ResultConfig


def test_resolve_result_request_defaults_statevector_for_nonstochastic_program():
    request = _resolve_result_request(
        _ResultConfig(counts=None, statevector=None),
        _PlanFacts(has_measurement=False, has_reset=False),
    )

    assert request.counts is False
    assert request.statevector is True
