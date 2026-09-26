import gzip
import json

import pytest
import requests


@pytest.fixture
def cloud(monkeypatch):
    """Replace HTTP only; use the real SDK for discovery, payloads and jobs."""
    monkeypatch.setenv("LQCLOUD_API_KEY", "test-only-not-a-credential")
    monkeypatch.setenv("LQCLOUD_URL", "https://cloud.logicalqubit.com")
    state = {
        "submitted": [],
        "status": "completed",
        "result": {"counts": {"0001": 10}, "success": True},
        "configs": [
            {
                "name": name,
                "qubits": 10,
                "native_gates": None,
                "topology": {"coupling_map": [[i, i + 1] for i in range(9)]},
                "supported_measurement_types": modes,
            }
            for name, modes in [
                ("AGate-100", ["measure2"]),
                ("QZ01-surface_code", ["measure", "measure2"]),
                ("MQ02", ["measure2"]),
            ]
        ],
    }

    def request(_session, method, url, **kwargs):
        if method.upper() == "GET" and url.endswith("/qpus"):
            data = state["configs"]
        elif method.upper() == "POST" and url.endswith("/tasks/async"):
            body = kwargs.get("json")
            if body is None:
                raw = kwargs["data"]
                if kwargs.get("headers", {}).get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                body = json.loads(raw)
            state["submitted"].append(body)
            data = {"task_id": "test-job", "status": "queued"}
        elif url.endswith("/cancel"):
            state["status"] = "cancelled"
            data = {"success": True, "status": "cancelled"}
        elif method.upper() == "GET" and url.endswith("/test-job"):
            data = {
                "task_id": "test-job",
                "status": state["status"],
                "result": state["result"],
                "error": "test failure",
            }
        else:
            raise AssertionError(f"Unexpected HTTP request: {method} {url}")
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(data).encode()
        response.headers["Content-Type"] = "application/json"
        return response

    monkeypatch.setattr(requests.Session, "request", request)
    return state
