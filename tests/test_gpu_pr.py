"""Offline PR revision, approval gating and GPU status tests."""

import copy
import importlib.util
import io
import json
from pathlib import Path
import re
from urllib.error import HTTPError, URLError

import pytest

SPEC = importlib.util.spec_from_file_location(
    "gpu_pr", Path(__file__).resolve().parents[1] / ".github/scripts/gpu_pr.py"
)
gpu_pr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gpu_pr)
SHA, TRUSTED, BASE, MERGE = (character * 40 for character in "abcd")


@pytest.fixture(name="case")
def controller_case(tmp_path, monkeypatch):
    pr = {
        "number": 42,
        "state": "open",
        "base": {"repo": {"full_name": "upstream/fatqat"}, "sha": BASE},
        "head": {"sha": SHA, "repo": {"full_name": "contributor/fatqat"}},
        "mergeable": True,
        "merge_commit_sha": MERGE,
    }
    merge = {"sha": MERGE, "parents": [{"sha": BASE}, {"sha": SHA}]}
    event = {
        "action": "opened",
        "number": 42,
        "repository": {"default_branch": "main"},
        "pull_request": copy.deepcopy(pr),
    }
    env = {
        "GITHUB_REPOSITORY": "upstream/fatqat",
        "GITHUB_EVENT_NAME": "pull_request_target",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_SHA": TRUSTED,
        "GITHUB_RUN_ID": "99",
        "GITHUB_EVENT_PATH": str(tmp_path / "event.json"),
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        "GH_TOKEN": "job-token",
    }
    calls, waits = [], []
    protection = {
        "protection_rules": [
            {
                "type": "required_reviewers",
                "reviewers": [{"type": "User", "reviewer": {"id": 1}}],
            }
        ]
    }

    def request(_api, path, payload=None):
        calls.append((path, payload))
        if payload is not None:
            return {}
        if path == "/environments/gpu-ci":
            return copy.deepcopy(protection)
        if path == f"/git/commits/{pr['merge_commit_sha']}":
            return copy.deepcopy(merge)
        if path == "/pulls/42":
            return copy.deepcopy(pr)
        raise AssertionError(path)

    monkeypatch.setattr(gpu_pr.GitHub, "request", request)
    monkeypatch.setattr(gpu_pr, "sleep", waits.append)

    def invoke(action="select"):
        Path(env["GITHUB_EVENT_PATH"]).write_text(json.dumps(event), encoding="utf-8")
        return gpu_pr.main([action], env)

    return {
        "pr": pr,
        "merge": merge,
        "event": event,
        "env": env,
        "calls": calls,
        "waits": waits,
        "protection": protection,
        "invoke": invoke,
    }


def outputs(case):
    path = Path(case["env"]["GITHUB_OUTPUT"])
    return (
        dict(
            line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines()
        )
        if path.exists()
        else {}
    )


@pytest.mark.parametrize("action", ["opened", "synchronize", "reopened"])
def test_selects_current_pr_from_event_head_not_control_sha(case, action):
    case["event"]["action"] = action
    assert case["invoke"]() == 0
    assert outputs(case) == {
        "mode": "pr",
        "pr_number": "42",
        "head_sha": SHA,
        "base_sha": BASE,
        "merge_sha": MERGE,
        "ready": "true",
    }


@pytest.mark.parametrize("title", ["docs: update README", "fix: core simulator"])
def test_all_prs_prepare_approval_without_cpu_or_file_queries(case, title):
    # PR events do not contain a file list; preparation must not request one.
    case["event"]["pull_request"]["title"] = title
    assert case["invoke"]("prepare") == 0
    assert outputs(case)["mode"] == "pr"
    assert outputs(case)["ready"] == "true"
    assert case["calls"] == [
        ("/pulls/42", None),
        (f"/git/commits/{MERGE}", None),
        ("/environments/gpu-ci", None),
    ]
    summary = Path(case["env"]["GITHUB_STEP_SUMMARY"]).read_text(encoding="utf-8")
    assert "PR #42" in summary and "gpu-ci" in summary
    assert all(sha in summary for sha in (SHA, BASE, MERGE))


@pytest.mark.parametrize(
    "path", ["/pulls/42", f"/git/commits/{MERGE}", "/environments/gpu-ci"]
)
def test_api_failure_blocks_approval(case, monkeypatch, path):
    request = gpu_pr.GitHub.request

    def fail(api, endpoint, payload=None):
        if endpoint == path:
            raise gpu_pr.APIError("Unavailable")
        return request(api, endpoint, payload)

    monkeypatch.setattr(gpu_pr.GitHub, "request", fail)
    assert case["invoke"]("prepare") == 1
    assert outputs(case) == {}


@pytest.mark.parametrize(
    "rules",
    [[], [{"type": "required_reviewers", "reviewers": []}], [{"type": "wait_timer"}]],
)
def test_missing_required_reviewers_cannot_request_approval(case, rules, capsys):
    case["protection"]["protection_rules"] = rules
    assert case["invoke"]("prepare") == 1
    assert outputs(case) == {}
    assert "Required reviewers" in capsys.readouterr().err


@pytest.mark.parametrize("changed", [None, "head", "base", "merge", "closed"])
def test_post_approval_recheck_keeps_the_prepared_revision(case, changed):
    assert case["invoke"]("prepare") == 0
    prepared = outputs(case)
    Path(case["env"]["GITHUB_OUTPUT"]).unlink()
    case["env"].update(
        EXPECTED_PR=prepared["pr_number"],
        EXPECTED_HEAD_SHA=prepared["head_sha"],
        EXPECTED_BASE_SHA=prepared["base_sha"],
        EXPECTED_MERGE_SHA=prepared["merge_sha"],
    )
    if changed in {"head", "base"}:
        case["pr"][changed]["sha"] = "e" * 40
        case["merge"]["parents"][changed == "head"]["sha"] = "e" * 40
    elif changed == "merge":
        case["pr"]["merge_commit_sha"] = case["merge"]["sha"] = "e" * 40
    elif changed == "closed":
        case["pr"]["state"] = "closed"
    assert case["invoke"]() == 0
    assert (outputs(case).get("ready") == "true") == (changed is None)


def test_environment_protection_is_rechecked_before_power(case):
    case["env"].update(
        EXPECTED_PR="42",
        EXPECTED_HEAD_SHA=SHA,
        EXPECTED_BASE_SHA=BASE,
        EXPECTED_MERGE_SHA=MERGE,
    )
    case["protection"]["protection_rules"] = []
    assert case["invoke"]() == 1
    assert outputs(case) == {}


def test_unsupported_pr_activity_is_skipped(case):
    case["event"]["action"] = "edited"
    assert case["invoke"]() == 0
    assert outputs(case) == {}
    assert not case["calls"]


@pytest.mark.parametrize("change", ["number", "base_repository"])
def test_wrong_pr_identity_is_never_selected(case, change):
    if change == "number":
        case["pr"]["number"] = 43
    else:
        case["pr"]["base"]["repo"]["full_name"] = "other/fatqat"
    assert case["invoke"]() == 0
    assert outputs(case) == {}


@pytest.mark.parametrize(
    "field,value",
    [
        ("EXPECTED_PR", "43"),
        ("EXPECTED_HEAD_SHA", "e" * 40),
        ("EXPECTED_BASE_SHA", "e" * 40),
        ("EXPECTED_MERGE_SHA", "e" * 40),
    ],
)
def test_recheck_after_gpu_start_rejects_changed_selection(case, field, value):
    case["env"].update(
        EXPECTED_PR="42",
        EXPECTED_HEAD_SHA=SHA,
        EXPECTED_BASE_SHA=BASE,
        EXPECTED_MERGE_SHA=MERGE,
    )
    expected = case["env"][field]
    case["env"][field] = value
    assert case["invoke"]() == 0
    assert outputs(case) == {}
    case["env"][field] = expected
    assert case["invoke"]() == 0
    assert outputs(case)["ready"] == "true"


def test_conflicting_merge_cannot_start_gpu(case):
    case["pr"]["mergeable"] = False
    assert case["invoke"]("prepare") == 0
    assert outputs(case) == {}
    assert case["calls"] == [("/pulls/42", None)]
    assert not case["waits"]


def test_unknown_mergeability_retries_then_prepares_approval(case, monkeypatch):
    case["pr"]["mergeable"] = None

    def resolve(seconds):
        case["waits"].append(seconds)
        case["pr"]["mergeable"] = True

    monkeypatch.setattr(gpu_pr, "sleep", resolve)
    assert case["invoke"]("prepare") == 0
    assert outputs(case)["ready"] == "true"
    assert outputs(case)["merge_sha"] == MERGE
    assert case["waits"] == [2]
    assert case["calls"] == [
        ("/pulls/42", None),
        ("/pulls/42", None),
        (f"/git/commits/{MERGE}", None),
        ("/environments/gpu-ci", None),
    ]


def test_persistently_unknown_mergeability_fails_with_rerun_hint(case, capsys):
    case["pr"]["mergeable"] = None
    assert case["invoke"]("prepare") == 1
    assert outputs(case) == {}
    assert case["calls"] == [("/pulls/42", None)] * 5
    assert case["waits"] == [2] * 4
    assert "unknown after 5 queries; rerun the GPU workflow" in capsys.readouterr().err


@pytest.mark.parametrize("changed", ["closed", "head"])
def test_mergeability_retry_rechecks_open_state_and_event_head(
    case, monkeypatch, changed
):
    case["pr"]["mergeable"] = None

    def change_pr(seconds):
        case["waits"].append(seconds)
        if changed == "closed":
            case["pr"]["state"] = "closed"
        else:
            case["pr"]["head"]["sha"] = "e" * 40

    monkeypatch.setattr(gpu_pr, "sleep", change_pr)
    assert case["invoke"]("prepare") == 0
    assert outputs(case) == {}
    assert case["calls"] == [("/pulls/42", None)] * 2
    assert case["waits"] == [2]


@pytest.mark.parametrize("field", ["base", "merge_commit_sha"])
@pytest.mark.parametrize("value", [None, "short", MERGE + ";echo unsafe"])
def test_missing_or_invalid_merge_revision_cannot_start_gpu(case, field, value):
    if field == "base":
        case["pr"][field]["sha"] = value
    else:
        case["pr"][field] = value
    assert case["invoke"]() == 0
    assert outputs(case) == {}


@pytest.mark.parametrize(
    "parents",
    [[], [BASE], [SHA, BASE], [BASE, TRUSTED], [TRUSTED, SHA], [BASE, SHA, TRUSTED]],
)
def test_merge_must_have_exact_selected_base_and_head_parents(case, parents):
    case["merge"]["parents"] = [{"sha": sha} for sha in parents]
    assert case["invoke"]() == 0
    assert outputs(case) == {}


def test_merge_api_must_return_the_requested_commit(case):
    case["merge"]["sha"] = TRUSTED
    assert case["invoke"]() == 0
    assert outputs(case) == {}


def test_selection_log_records_fixed_revision(case, capsys):
    assert case["invoke"]() == 0
    message = capsys.readouterr().out
    assert "PR #42" in message
    assert all(sha in message for sha in (SHA, BASE, MERGE))


def test_stale_recheck_skips_tests_and_reports_error_on_original_head(case):
    case["env"].update(
        EXPECTED_PR="42",
        EXPECTED_HEAD_SHA=SHA,
        EXPECTED_BASE_SHA=BASE,
        EXPECTED_MERGE_SHA=MERGE,
    )
    case["pr"]["head"]["sha"] = "e" * 40
    assert case["invoke"]() == 0
    assert outputs(case) == {}
    case["env"].update(
        PR_HEAD_SHA=SHA,
        START_RESULT="success",
        GPU_JOB_RESULT="success",
        TEST_RESULT="skipped",
        STOP_RESULT="success",
    )
    assert case["invoke"]("report") == 0
    path, payload = case["calls"][-1]
    assert path == f"/statuses/{SHA}"
    assert payload["state"] == "error"


@pytest.mark.parametrize("action", ["prepare", "select"])
def test_manual_default_branch_needs_no_pr_or_api_queries(case, action):
    case["env"]["GITHUB_EVENT_NAME"] = "workflow_dispatch"
    assert case["invoke"](action) == 0
    assert outputs(case) == {
        "mode": "manual",
        "ready": "true",
    }
    assert not case["calls"]


@pytest.mark.parametrize(
    "field,value", [("GITHUB_REF", "refs/heads/feature"), ("GITHUB_EVENT_NAME", "push")]
)
def test_non_default_branch_and_other_events_are_skipped(case, field, value):
    case["env"][field] = value
    assert case["invoke"]() == 0
    assert not case["calls"]
    assert outputs(case) == {}


def test_pending_links_to_gpu_run_on_exact_pr_commit(case):
    case["env"]["PR_HEAD_SHA"] = SHA
    assert case["invoke"]("pending") == 0
    path, payload = case["calls"][0]
    assert path == f"/statuses/{SHA}"
    assert payload["state"] == "pending"
    assert payload["target_url"] == "https://github.com/upstream/fatqat/actions/runs/99"


@pytest.mark.parametrize(
    "start,gpu_job,test,stop,state",
    [
        ("success", "success", "success", "success", "success"),
        ("success", "failure", "failure", "success", "failure"),
        ("success", "success", "failure", "success", "failure"),
        ("failure", "skipped", "skipped", "success", "error"),
        ("success", "success", "skipped", "success", "error"),
        ("success", "failure", "skipped", "success", "error"),
        ("success", "failure", "success", "success", "error"),
        ("success", "cancelled", "success", "success", "error"),
        ("success", "cancelled", "failure", "success", "error"),
        ("success", "cancelled", "cancelled", "success", "error"),
        ("success", "success", "success", "failure", "error"),
        ("success", "failure", "failure", "failure", "error"),
        ("success", "success", None, "success", "error"),
    ],
)
def test_status_requires_job_and_test_success(case, start, gpu_job, test, stop, state):
    case["env"].update(
        PR_HEAD_SHA=SHA,
        START_RESULT=start,
        GPU_JOB_RESULT=gpu_job,
        TEST_RESULT=test,
        STOP_RESULT=stop,
    )
    assert case["invoke"]("report") == 0
    path, payload = case["calls"][0]
    assert path == f"/statuses/{SHA}"
    assert payload["state"] == state


def workflow_job(name):
    text = (
        Path(__file__).resolve().parents[1] / ".github/workflows/gpu.yml"
    ).read_text(encoding="utf-8")
    return re.search(rf"^  {name}:\n(.*?)(?=^  [\w-]+:|\Z)", text, re.M | re.S)[1]


def test_workflow_cleanup_covers_start_and_gpu_failures_only_after_attempt():
    start, stop = workflow_job("start-gpu"), workflow_job("stop-gpu")
    # Check the actual job condition: no success() or GPU result can suppress cleanup
    # after start failure, install/preflight/test failure, or GPU job timeout.
    assert "needs: [start-gpu, gpu-tests]" in stop
    condition = re.search(r"^    if: (.+)$", stop, re.M)[1]
    assert condition == "${{ always() && needs.start-gpu.outputs.attempted == 'true' }}"
    # Ineligible selections emit no ready output, so neither start nor stop can run.
    power = start.split("      - name: Start instance and wait for runner\n", 1)[1]
    assert "if: steps.select.outputs.ready == 'true'" in power
    assert (
        'echo "attempted=true" >> "$GITHUB_OUTPUT"\n          python -u .github/scripts/hai_power.py start'
        in power
    )
    assert "environment:" not in stop


def test_workflow_requests_environment_approval_for_every_pr():
    prepare, approve = workflow_job("prepare"), workflow_job("approve")
    text = (
        Path(__file__).resolve().parents[1] / ".github/workflows/gpu.yml"
    ).read_text(encoding="utf-8")
    triggers = text.split("on:\n", 1)[1].split("\npermissions:", 1)[0]
    assert (
        triggers.strip()
        == "workflow_dispatch:\n  pull_request_target:\n    types: [opened, synchronize, reopened]"
    )
    assert "gpu_pr.py prepare" in prepare
    assert "secrets." not in prepare
    assert "pip install" not in prepare
    assert "environment:" not in prepare
    assert re.search(r"^    if: (.+)$", approve, re.M)[1] == (
        "needs.prepare.outputs.mode == 'pr'"
    )
    assert "needs: prepare" in approve
    assert "environment:\n      name: gpu-ci" in approve
    assert "permissions: {}" in approve
    assert "hai_power.py" not in approve
    assert "secrets." not in approve


def test_workflow_cannot_start_pr_without_successful_gpu_approval():
    start = workflow_job("start-gpu")
    condition = re.search(r"^    if: >-\n(.*?)(?=^    \w)", start, re.M | re.S)[1]
    assert " ".join(condition.split()) == (
        "always() && !cancelled() && needs.prepare.result == 'success' && "
        "needs.prepare.outputs.ready == 'true' && "
        "(needs.prepare.outputs.mode == 'manual' || "
        "(needs.prepare.outputs.mode == 'pr' && "
        "needs.approve.result == 'success'))"
    )
    assert "needs: [prepare, approve]" in start
    for field in ("head_sha", "base_sha", "merge_sha"):
        assert (
            f"EXPECTED_{field.upper()}: ${{{{ needs.prepare.outputs.{field} }}}}"
            in start
        )
    assert "EXPECTED_PR: ${{ needs.prepare.outputs.pr_number }}" in start
    assert start.index("gpu_pr.py select") < start.index("hai_power.py start")
    report = workflow_job("report")
    assert re.search(r"^    if: (.+)$", report, re.M)[1] == (
        "${{ always() && needs.prepare.outputs.mode == 'pr' }}"
    )


@pytest.mark.parametrize(
    "start_result,ready,cancelled,expected",
    [
        pytest.param("success", "true", False, True, id="manual-approve-skipped"),
        ("failure", "true", False, False),
        ("skipped", "true", False, False),
        ("cancelled", "true", False, False),
        ("success", "false", False, False),
        ("success", "", False, False),
        ("success", "true", True, False),
    ],
)
def test_gpu_job_explicitly_requires_successful_ready_start(
    start_result, ready, cancelled, expected
):
    gpu = workflow_job("gpu-tests")
    condition = re.search(r"^    if: (.+)$", gpu, re.M)[1]
    assert condition == (
        "${{ !cancelled() && needs.start-gpu.result == 'success' && "
        "needs.start-gpu.outputs.ready == 'true' }}"
    )
    # A skipped manual approval must not block a successful start's GPU job.
    # PR approval remains enforced by the start-gpu gate checked above.
    assert "needs: start-gpu" in gpu
    # Evaluate only these predicates locally, not GitHub's scheduler.
    predicates = {
        "!cancelled()": not cancelled,
        "needs.start-gpu.result == 'success'": start_result == "success",
        "needs.start-gpu.outputs.ready == 'true'": ready == "true",
    }
    assert all(predicates[part] for part in condition[4:-3].split(" && ")) == expected


def test_workflow_reports_job_result_separately_from_pytest_outcome():
    gpu, report = workflow_job("gpu-tests"), workflow_job("report")
    assert "test_result: ${{ steps.tests.outcome }}" in gpu
    for key, expression in {
        "PR_HEAD_SHA": "needs.prepare.outputs.head_sha",
        "START_RESULT": "needs.start-gpu.result",
        "GPU_JOB_RESULT": "needs.gpu-tests.result",
        "TEST_RESULT": "needs.gpu-tests.outputs.test_result",
        "STOP_RESULT": "needs.stop-gpu.result",
    }.items():
        assert f"{key}: ${{{{ {expression} }}}}" in report


def test_unsafe_checkout_is_only_for_pinned_pr_merge_and_verification_precedes_install():
    gpu = workflow_job("gpu-tests")
    trusted, pr_checkout = gpu.split(
        "      - name: Check out selected PR merge commit\n"
    )
    assert "allow-unsafe-pr-checkout" not in trusted
    assert "ref: ${{ needs.start-gpu.outputs.merge_sha }}" in pr_checkout
    assert "fetch-depth: 2" in pr_checkout
    assert "allow-unsafe-pr-checkout: true" in pr_checkout
    assert "persist-credentials: false" in pr_checkout
    verification = pr_checkout.split("          export PATH=", 1)[0]
    assert 'test "$(git rev-parse HEAD)" = "$MERGE_SHA"' in verification
    assert (
        'test "$(git show -s --format=%P HEAD)" = "$BASE_SHA $PR_HEAD_SHA"'
        in verification
    )
    assert "pip install" not in verification
    assert "secrets." not in gpu
    assert ": write" not in gpu
    for name in ("prepare", "start-gpu", "stop-gpu", "report"):
        control = workflow_job(name)
        assert "runs-on: ubuntu-latest" in control
        assert "ref: ${{ github.sha }}" in control
        assert "allow-unsafe-pr-checkout" not in control


@pytest.mark.parametrize("value", ["a" * 7, SHA + ";echo unsafe", SHA + "\n"])
def test_malformed_sha_never_becomes_output(case, value):
    case["event"]["pull_request"]["head"]["sha"] = value
    assert case["invoke"]() == 0
    assert outputs(case) == {}


@pytest.mark.parametrize(
    "error",
    [
        URLError("secret"),
        TimeoutError("secret"),
        HTTPError("url", 403, "secret", {}, None),
    ],
)
def test_transport_failure_redacts_details_and_uses_job_token(monkeypatch, error):
    def fail(_opener, request, timeout):
        assert request.get_header("Authorization") == "Bearer job-token"
        assert timeout == 20
        raise error

    monkeypatch.setattr(
        gpu_pr, "build_opener", lambda *_args: type("Opener", (), {"open": fail})()
    )
    api = gpu_pr.GitHub(
        {"GITHUB_REPOSITORY": "upstream/fatqat", "GH_TOKEN": "job-token"}
    )
    with pytest.raises(gpu_pr.APIError) as caught:
        api.request("/pulls/42")
    assert "secret" not in str(caught.value)


def test_transport_reads_metadata_and_posts_status_with_job_token(monkeypatch):
    class Response(io.BytesIO):
        pass

    requests = []

    def respond(_opener, request, timeout):
        requests.append(request)
        return Response(b'{"number": 42}' if request.data is None else b"{}")

    monkeypatch.setattr(
        gpu_pr, "build_opener", lambda *_args: type("Opener", (), {"open": respond})()
    )
    api = gpu_pr.GitHub(
        {
            "GITHUB_REPOSITORY": "upstream/fatqat",
            "GH_TOKEN": "job-token",
            "GITHUB_RUN_ID": "99",
        }
    )
    assert api.request("/pulls/42") == {"number": 42}
    api.status(SHA, "success")
    assert requests[-1].get_method() == "POST"
    assert json.loads(requests[-1].data)["state"] == "success"
    assert (
        gpu_pr.NoRedirect().redirect_request(
            None, None, 302, "", {}, "https://other.example"
        )
        is None
    )
