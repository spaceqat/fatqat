"""Offline selection and status tests for the workflow_run GPU controller."""

import copy
import importlib.util
import io
import json
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

import pytest

SPEC = importlib.util.spec_from_file_location(
    "gpu_pr", Path(__file__).resolve().parents[1] / ".github/scripts/gpu_pr.py"
)
gpu_pr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gpu_pr)
SHA, TRUSTED, BASE, MERGE = (character * 40 for character in "abcd")


@pytest.fixture(name="case")
def controller_case(tmp_path, monkeypatch):
    run = {
        "id": 10,
        "run_attempt": 1,
        "workflow_id": 20,
        "event": "pull_request",
        "status": "completed",
        "conclusion": "success",
        "head_sha": SHA,
        "path": ".github/workflows/tests.yml",
        "repository": {"full_name": "upstream/fatqat"},
        "head_repository": {"id": 30, "owner": {"login": "contributor"}},
        "head_branch": "feature",
        "pull_requests": [{"number": 42}],
    }
    pr = {
        "number": 42,
        "state": "open",
        "base": {"repo": {"full_name": "upstream/fatqat"}, "sha": BASE},
        "head": {"sha": SHA, "ref": "feature", "repo": {"id": 30}},
        "mergeable": True,
        "merge_commit_sha": MERGE,
        "changed_files": 1,
    }
    merge = {"sha": MERGE, "parents": [{"sha": BASE}, {"sha": SHA}]}
    event = {
        "action": "completed",
        "repository": {"default_branch": "main"},
        "workflow_run": copy.deepcopy(run),
    }
    env = {
        "GITHUB_REPOSITORY": "upstream/fatqat",
        "GITHUB_EVENT_NAME": "workflow_run",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_SHA": TRUSTED,
        "GITHUB_RUN_ID": "99",
        "GITHUB_EVENT_PATH": str(tmp_path / "event.json"),
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        "GH_TOKEN": "job-token",
    }
    calls, pages = [], [[pr]]
    file_pages = [[{"filename": "tests/gpu_tests/test_cupy.py", "status": "modified"}]]
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
        if path == "/actions/runs/10":
            return copy.deepcopy(run)
        if path == "/actions/workflows/tests.yml":
            return {"id": 20}
        if path == "/environments/gpu-ci":
            return copy.deepcopy(protection)
        if path == f"/git/commits/{pr['merge_commit_sha']}":
            return copy.deepcopy(merge)
        if path.startswith("/pulls?"):
            query = parse_qs(urlsplit(path).query)
            assert query["state"] == ["open"]
            assert query["head"] == [f"contributor:{run['head_branch']}"]
            return copy.deepcopy(pages[int(query["page"][0]) - 1])
        if path.startswith("/pulls/42/files?"):
            query = parse_qs(urlsplit(path).query)
            assert query["per_page"] == ["100"]
            return copy.deepcopy(file_pages[int(query["page"][0]) - 1])
        if path.startswith("/pulls/"):
            result = copy.deepcopy(pr)
            result["number"] = int(path.rsplit("/", 1)[1])
            return result
        raise AssertionError(path)

    monkeypatch.setattr(gpu_pr.GitHub, "request", request)

    def invoke(action="select"):
        Path(env["GITHUB_EVENT_PATH"]).write_text(json.dumps(event), encoding="utf-8")
        return gpu_pr.main([action], env)

    return {
        "run": run,
        "pr": pr,
        "merge": merge,
        "event": event,
        "env": env,
        "calls": calls,
        "pages": pages,
        "file_pages": file_pages,
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


@pytest.mark.parametrize("fork", [False, True])
def test_resolves_current_pr_even_when_run_omits_pr_list(case, fork):
    if fork:
        case["run"]["pull_requests"] = []
        case["event"]["workflow_run"]["pull_requests"] = []
    assert case["invoke"]() == 0
    assert outputs(case) == {
        "mode": "pr",
        "pr_number": "42",
        "head_sha": SHA,
        "base_sha": BASE,
        "merge_sha": MERGE,
        "ready": "true",
    }


def test_fork_lookup_paginates_and_url_encodes_branch(case):
    case["run"]["pull_requests"] = []
    case["run"]["head_branch"] = "feature/x&other=1"
    case["pr"]["head"]["ref"] = "feature/x&other=1"
    case["pages"][:] = [[{"number": 42}] * 100, []]
    assert case["invoke"]() == 0
    assert outputs(case)["ready"] == "true"
    assert sum(path.startswith("/pulls?") for path, _ in case["calls"]) == 2


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "timed_out"])
def test_failed_cpu_tests_do_not_inspect_paths_or_request_gpu(case, conclusion):
    case["run"]["conclusion"] = conclusion
    case["event"]["workflow_run"]["conclusion"] = conclusion
    assert case["invoke"]("prepare") == 0
    assert outputs(case) == {}
    assert not any(
        "/files?" in path or "/environments/" in path for path, _ in case["calls"]
    )


@pytest.mark.parametrize(
    "path,required",
    [
        ("tests/gpu_tests/test_cupy.py", True),
        ("tests/gpu_tests/data/input.npy", True),
        ("tests/test_gpu_pr.py", True),
        (".github/workflows/gpu.yml", True),
        (".github/scripts/gpu_pr.py", True),
        (".github/scripts/hai_power.py", True),
        ("pyproject.toml", True),
        ("tests/gpu_tests/README.md", False),
        ("tests/gpu_tests/notes.rst", False),
        ("docs/mkdocs/en/gpu.md", False),
        ("README.md", False),
        ("src/fatqat/simulator/simulator.py", False),
        ("tests/simulator/test_execution.py", False),
        (".github/workflows/tests.yml", False),
        (".github/workflows/lint.yml", False),
    ],
)
@pytest.mark.parametrize("status", ["added", "modified", "removed"])
def test_gpu_selection_uses_changed_paths_not_repository_contents(
    case, path, required, status
):
    case["file_pages"][:] = [[{"filename": path, "status": status}]]
    assert case["invoke"]("prepare") == 0
    assert outputs(case)["needs_gpu"] == str(required).lower()
    assert any(path == "/environments/gpu-ci" for path, _ in case["calls"]) == required
    assert all(payload is None for _, payload in case["calls"])
    summary = Path(case["env"]["GITHUB_STEP_SUMMARY"]).read_text(encoding="utf-8")
    assert "PR #42" in summary
    assert all(sha in summary for sha in (SHA, BASE, MERGE))


@pytest.mark.parametrize(
    "old,new,required",
    [
        ("tests/gpu_tests/test_old.py", "tests/cpu/test_old.py", True),
        ("tests/cpu/test_old.py", "tests/gpu_tests/test_new.py", True),
        ("docs/old.md", "docs/new.md", False),
        ("tests/gpu_tests/README.md", "docs/gpu.md", False),
    ],
)
def test_renames_check_both_old_and_new_paths(case, old, new, required):
    case["file_pages"][:] = [
        [{"filename": new, "previous_filename": old, "status": "renamed"}]
    ]
    assert case["invoke"]("prepare") == 0
    assert outputs(case)["needs_gpu"] == str(required).lower()


def test_file_pagination_includes_gpu_change_on_later_page(case):
    case["pr"]["changed_files"] = 101
    case["file_pages"].insert(
        0, [{"filename": f"docs/{i}.md", "status": "modified"} for i in range(100)]
    )
    assert case["invoke"]("prepare") == 0
    assert outputs(case)["needs_gpu"] == "true"
    assert sum("/files?" in path for path, _ in case["calls"]) == 2


@pytest.mark.parametrize(
    "path", ["/pulls/42/files?per_page=100&page=1", "/environments/gpu-ci"]
)
def test_file_or_environment_api_failure_never_means_no_gpu_needed(
    case, monkeypatch, path
):
    request = gpu_pr.GitHub.request

    def fail(api, endpoint, payload=None):
        if endpoint == path:
            raise gpu_pr.APIError("Unavailable")
        return request(api, endpoint, payload)

    monkeypatch.setattr(gpu_pr.GitHub, "request", fail)
    assert case["invoke"]("prepare") == 1
    assert outputs(case) == {}


@pytest.mark.parametrize("count", [2, 3001, None])
def test_incomplete_or_uninspectable_file_lists_fail_closed(case, count):
    case["pr"]["changed_files"] = count
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


def test_version_change_during_file_query_cannot_produce_selection(case, monkeypatch):
    request = gpu_pr.GitHub.request

    def change_head(api, path, payload=None):
        result = request(api, path, payload)
        if "/files?" in path:
            case["pr"]["head"]["sha"] = "e" * 40
        return result

    monkeypatch.setattr(gpu_pr.GitHub, "request", change_head)
    assert case["invoke"]("prepare") == 0
    assert outputs(case) == {}


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


@pytest.mark.parametrize(
    "changes",
    [
        {"conclusion": "failure"},
        {"status": "in_progress"},
        {"event": "push"},
        {"event": "pull_request_target"},
        {"workflow_id": 21},
        {"run_attempt": 2},
        {"repository": {"full_name": "contributor/fatqat"}},
        {"head_sha": "c" * 40},
    ],
)
def test_unsuccessful_wrong_or_superseded_source_run_cannot_start_gpu(case, changes):
    case["run"].update(changes)
    assert case["invoke"]() == 0
    assert outputs(case) == {}


@pytest.mark.parametrize(
    "field,value", [("action", "requested"), ("conclusion", "failure")]
)
def test_source_event_must_be_successful_completion(case, field, value):
    target = case["event"] if field == "action" else case["event"]["workflow_run"]
    target[field] = value
    assert case["invoke"]() == 0
    assert outputs(case) == {}


@pytest.mark.parametrize(
    "change", ["closed", "sha", "fork", "branch", "base", "deleted_fork"]
)
@pytest.mark.parametrize("empty_list", [False, True])
def test_closed_stale_or_wrong_pr_is_never_selected(case, change, empty_list):
    if empty_list:
        case["run"]["pull_requests"] = []
    pr = case["pr"]
    if change == "closed":
        pr["state"] = "closed"
    elif change == "sha":
        pr["head"]["sha"] = "c" * 40
    elif change == "fork":
        pr["head"]["repo"]["id"] = 31
    elif change == "branch":
        pr["head"]["ref"] = "different"
    elif change == "base":
        pr["base"]["repo"]["full_name"] = "other/fatqat"
    else:
        pr["head"]["repo"] = None
    assert case["invoke"]() == 0
    assert outputs(case) == {}


@pytest.mark.parametrize("candidates", [[], [{"number": 42}, {"number": 43}]])
def test_missing_or_ambiguous_fork_pr_is_skipped(case, candidates):
    case["run"]["pull_requests"] = []
    case["pages"][:] = [candidates]
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


@pytest.mark.parametrize("mergeable", [False, None])
def test_conflicting_or_unconfirmed_merge_cannot_start_gpu(case, mergeable):
    case["pr"]["mergeable"] = mergeable
    assert case["invoke"]() == 0
    assert outputs(case) == {}


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


def test_selection_log_distinguishes_current_merge_from_cpu_tested_revision(
    case, capsys
):
    assert case["invoke"]() == 0
    message = capsys.readouterr().out
    assert all(sha in message for sha in (SHA, BASE, MERGE))
    assert (
        "not verified as the merge commit tested by the preceding CPU Tests" in message
    )


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


def test_non_default_branch_and_other_events_are_skipped(case):
    case["env"]["GITHUB_REF"] = "refs/heads/feature"
    assert case["invoke"]() == 0
    assert not case["calls"]
    assert outputs(case) == {}


def test_api_failure_fails_closed(case, monkeypatch):
    def unavailable(*_args):
        raise gpu_pr.APIError("Unavailable")

    monkeypatch.setattr(gpu_pr.GitHub, "request", unavailable)
    assert case["invoke"]() == 1
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


def test_workflow_requests_environment_approval_only_for_gpu_relevant_prs():
    prepare, approve = workflow_job("prepare"), workflow_job("approve")
    assert "github.event.workflow_run.conclusion == 'success'" in prepare
    assert "gpu_pr.py prepare" in prepare
    assert "secrets." not in prepare
    assert "pip install" not in prepare
    assert "environment:" not in prepare
    assert re.search(r"^    if: (.+)$", approve, re.M)[1] == (
        "needs.prepare.outputs.mode == 'pr' && needs.prepare.outputs.needs_gpu == 'true'"
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
        "(needs.prepare.outputs.mode == 'pr' && needs.prepare.outputs.needs_gpu == 'true' && "
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
    assert "needs.prepare.outputs.needs_gpu == 'true'" in report


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
    case["run"]["head_sha"] = value
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
        api.request("/actions/runs/10")
    assert "secret" not in str(caught.value)


def test_transport_accepts_paginated_list_and_posts_status_with_job_token(monkeypatch):
    class Response(io.BytesIO):
        pass

    requests = []

    def respond(_opener, request, timeout):
        requests.append(request)
        return Response(b"[]" if request.data is None else b"{}")

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
    assert api.request("/pulls?state=open") == []
    api.status(SHA, "success")
    assert requests[-1].get_method() == "POST"
    assert json.loads(requests[-1].data)["state"] == "success"
    assert (
        gpu_pr.NoRedirect().redirect_request(
            None, None, 302, "", {}, "https://other.example"
        )
        is None
    )
