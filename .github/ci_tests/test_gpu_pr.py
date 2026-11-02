"""Offline PR pinning, source transfer boundaries and workflow tests."""

import base64
import copy
import importlib.util
import io
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "gpu_pr", ROOT / ".github/scripts/gpu_pr.py"
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
        "GITHUB_EVENT_PATH": str(tmp_path / "event.json"),
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        "RUNNER_TEMP": str(tmp_path),
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
    manual_protection = {"protection_rules": []}

    def request(_api, path):
        calls.append(path)
        if path == "/environments/gpu-ci":
            return copy.deepcopy(protection)
        if path == "/environments/gpu-ci-manual":
            return copy.deepcopy(manual_protection)
        if path == f"/git/commits/{pr['merge_commit_sha']}":
            return copy.deepcopy(merge)
        if path == "/pulls/42":
            return copy.deepcopy(pr)
        raise AssertionError(path)

    monkeypatch.setattr(gpu_pr.GitHub, "request", request)
    monkeypatch.setattr(gpu_pr, "sleep", waits.append)

    def invoke(action="prepare"):
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
        "manual_protection": manual_protection,
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
    }


def test_merge_base_comes_from_tested_commit_not_pr_api(case):
    case["pr"]["base"]["sha"] = TRUSTED
    assert case["invoke"]() == 0
    assert outputs(case)["base_sha"] == BASE
    assert case["invoke"]("recheck") == 0


def test_unexpected_prepare_error_preserves_traceback(case, monkeypatch, capsys):
    def broken(_api, _event):
        raise TypeError("unexpected metadata shape")

    monkeypatch.setattr(gpu_pr, "select_pr", broken)
    assert case["invoke"]() == 1
    error = capsys.readouterr().err
    assert "Traceback" in error
    assert "TypeError: unexpected metadata shape" in error
    assert "job-token" not in error


def test_unexpected_exception_redacts_credentials_in_its_message(
    case, monkeypatch, capsys
):
    def broken(_api, _event):
        raise RuntimeError(f"unexpected error with {case['env']['GH_TOKEN']}")

    monkeypatch.setattr(gpu_pr, "select_pr", broken)
    assert case["invoke"]() == 1
    error = capsys.readouterr().err
    assert "Traceback" in error and "<redacted>" in error
    assert "job-token" not in error


@pytest.mark.parametrize("title", ["docs: update README", "fix: core simulator"])
def test_all_prs_selected_without_cpu_file_or_runner_queries(case, title, capsys):
    case["event"]["pull_request"]["title"] = title
    assert case["invoke"]() == 0
    assert case["calls"] == [
        "/pulls/42",
        f"/git/commits/{MERGE}",
        "/environments/gpu-ci",
    ]
    summary = Path(case["env"]["GITHUB_STEP_SUMMARY"]).read_text(encoding="utf-8")
    assert "after gpu-ci approval" in summary
    assert all(sha in summary for sha in (SHA, BASE, MERGE))
    log = capsys.readouterr().out
    assert all(sha in log for sha in (SHA, BASE, MERGE))


@pytest.mark.parametrize(
    "path", ["/pulls/42", f"/git/commits/{MERGE}", "/environments/gpu-ci"]
)
def test_api_failure_blocks_start(case, monkeypatch, path):
    request = gpu_pr.GitHub.request

    def fail(api, endpoint):
        if endpoint == path:
            raise gpu_pr.APIError("HTTP 403")
        return request(api, endpoint)

    monkeypatch.setattr(gpu_pr.GitHub, "request", fail)
    assert case["invoke"]() == 1
    assert outputs(case) == {}


@pytest.mark.parametrize(
    "rules",
    [
        None,
        {},
        [None],
        [{}],
        [],
        [{"type": "required_reviewers", "reviewers": []}],
        [{"type": "required_reviewers", "reviewers": {"id": 1}}],
        [{"type": "required_reviewers", "reviewers": [None]}],
        [{"type": "wait_timer"}],
    ],
)
def test_invalid_environment_protection_fails_explicitly(case, rules, capsys):
    case["protection"]["protection_rules"] = rules
    assert case["invoke"]() == 1
    assert outputs(case) == {}
    assert "Environment" in capsys.readouterr().err


def test_missing_environment_fields_fail_explicitly(case, capsys):
    case["protection"].clear()
    assert case["invoke"]() == 1
    assert "protection_rules" in capsys.readouterr().err


@pytest.mark.parametrize(
    "changed", [None, "head", "base", "merge", "closed", "protection"]
)
def test_recheck_keeps_pinned_revision_and_approval_rules(case, changed):
    assert case["invoke"]() == 0
    if changed in {"head", "base"}:
        case["merge"]["parents"][changed == "head"]["sha"] = "e" * 40
        case["pr"][changed]["sha"] = "e" * 40
    elif changed == "merge":
        case["pr"]["merge_commit_sha"] = case["merge"]["sha"] = "e" * 40
    elif changed == "closed":
        case["pr"]["state"] = "closed"
    elif changed == "protection":
        case["protection"]["protection_rules"] = []
    assert case["invoke"]("recheck") == (0 if changed is None else 1)


@pytest.mark.parametrize(
    "change",
    [
        "number",
        "event_number",
        "base_repository",
        "closed",
        "new_head",
        "conflict",
        "activity",
    ],
)
def test_wrong_or_outdated_pr_never_selected(case, change):
    target, key, value = {
        "number": (case["pr"], "number", 43),
        "event_number": (case["event"], "number", 43),
        "base_repository": (case["pr"]["base"]["repo"], "full_name", "other/fatqat"),
        "closed": (case["pr"], "state", "closed"),
        "new_head": (case["pr"]["head"], "sha", TRUSTED),
        "activity": (case["event"], "action", "edited"),
        "conflict": (case["pr"], "mergeable", False),
    }[change]
    target[key] = value
    assert case["invoke"]() == 1
    assert outputs(case) == {}


@pytest.mark.parametrize("resolve", [True, False, "head", "closed"])
def test_unknown_mergeability_has_bounded_retry_and_rechecks_identity(
    case, monkeypatch, resolve
):
    case["pr"]["mergeable"] = None

    def wait(seconds):
        case["waits"].append(seconds)
        if resolve is True:
            case["pr"]["mergeable"] = True
        elif resolve == "head":
            case["pr"]["head"]["sha"] = TRUSTED
        elif resolve == "closed":
            case["pr"]["state"] = "closed"

    monkeypatch.setattr(gpu_pr, "sleep", wait)
    assert case["invoke"]() == (0 if resolve is True else 1)
    assert case["waits"] == [2] * (4 if resolve is False else 1)


@pytest.mark.parametrize("value", [None, "a" * 7, SHA + ";echo unsafe", SHA + "\n"])
@pytest.mark.parametrize("field", ["head", "merge"])
def test_malformed_sha_never_becomes_output(case, field, value):
    if field == "head":
        case["event"]["pull_request"]["head"]["sha"] = value
    else:
        case["pr"]["merge_commit_sha"] = value
    assert case["invoke"]() == 1
    assert outputs(case) == {}


@pytest.mark.parametrize(
    "parents",
    [[], [BASE], [SHA, BASE], [BASE, TRUSTED], [BASE, SHA, TRUSTED], ["bad", SHA]],
)
def test_merge_requires_exactly_two_valid_parents_with_event_head_second(case, parents):
    case["merge"]["parents"] = [{"sha": sha} for sha in parents]
    assert case["invoke"]() == 1
    assert outputs(case) == {}


def test_merge_api_must_return_requested_commit(case):
    case["merge"]["sha"] = TRUSTED
    assert case["invoke"]() == 1
    assert outputs(case) == {}


def test_manual_pins_default_branch_event_commit_and_uses_manual_environment(case):
    case["env"]["GITHUB_EVENT_NAME"] = "workflow_dispatch"
    assert case["invoke"]() == 0
    assert outputs(case) == {"mode": "manual", "merge_sha": TRUSTED}
    assert case["calls"] == ["/environments/gpu-ci-manual"]
    case["manual_protection"].update(case["protection"])
    assert case["invoke"]("recheck") == 1


@pytest.mark.parametrize(
    "field,value", [("GITHUB_REF", "refs/heads/feature"), ("GITHUB_EVENT_NAME", "push")]
)
def test_non_default_branch_and_other_events_fail(case, field, value):
    case["env"][field] = value
    assert case["invoke"]() == 1
    assert not case["calls"]


@pytest.mark.parametrize(
    "change",
    [None, "revision", "parents", "api_head", "git_failure", "git_timeout", "manual"],
)
def test_archive_fetches_only_pinned_source_without_executing_it(
    case, monkeypatch, change
):
    if change == "manual":
        case["env"]["GITHUB_EVENT_NAME"] = "workflow_dispatch"
    assert case["invoke"]() == 0
    commands = []

    def git(command, **kwargs):
        commands.append((command, dict(kwargs["env"])))
        assert kwargs["capture_output"] and kwargs["timeout"] == 300
        operation = command[3]
        if operation == "fetch":
            if change == "git_failure":
                raise subprocess.CalledProcessError(1, command, stderr="secret")
            if change == "git_timeout":
                raise subprocess.TimeoutExpired(command, 300)
            if change == "api_head":
                case["pr"]["head"]["sha"] = TRUSTED
        output = ""
        if operation == "rev-parse":
            output = TRUSTED if change in {"revision", "manual"} else MERGE
        elif operation == "show":
            output = f"{BASE} {TRUSTED if change == 'parents' else SHA}"
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(gpu_pr.subprocess, "run", git)
    assert case["invoke"]("archive") == (0 if change in {None, "manual"} else 1)
    fetch, fetch_env = next(item for item in commands if item[0][3] == "fetch")
    assert fetch[-1] == (TRUSTED if change == "manual" else "refs/pull/42/merge")
    assert "--bare" in commands[0][0] and "--template=" in commands[0][0]
    assert all(
        not {"checkout", "pip", "pytest"}.intersection(command)
        for command, _ in commands
    )
    assert all("job-token" not in str(command) for command, _ in commands)
    assert all("GH_TOKEN" not in env for _, env in commands)
    assert (
        base64.b64decode(fetch_env["GIT_CONFIG_VALUE_0"].split()[-1])
        == b"x-access-token:job-token"
    )
    assert all(
        "GIT_CONFIG_VALUE_0" not in env
        for command, env in commands
        if command[3] != "fetch"
    )


def workflow():
    return (ROOT / ".github/workflows/gpu.yml").read_text(encoding="utf-8-sig")


def test_workflow_has_one_hosted_job_and_environment_selected_before_all_steps():
    text = workflow()
    assert re.findall(r"^  ([\w-]+):$", text.split("jobs:\n")[1], re.M) == ["gpu"]
    assert "runs-on: ubuntu-latest" in text
    assert (
        "name: ${{ github.event_name == 'pull_request_target' && 'gpu-ci' || 'gpu-ci-manual' }}"
        in text
    )
    display_name = (
        "${{ github.event_name == 'pull_request_target' && "
        "format('GPU PR #{0}', github.event.number) || 'GPU manual' }}"
    )
    assert text.startswith(f"name: GPU Tests\nrun-name: >-\n  {display_name}\n")
    assert f"    name: >-\n      {display_name}\n" in text
    assert (
        "github.ref == format('refs/heads/{0}', github.event.repository.default_branch)"
        in text
    )
    assert "types: [opened, synchronize, reopened]" in text
    assert "workflow_dispatch:" in text
    assert "group: fatqat-hai-gpu\n  cancel-in-progress: false\n  queue: max" in text
    for forbidden in (
        "self-hosted",
        "GH_RUNNER_READ_TOKEN",
        "HAI_RUNNER_NAME",
        "needs:",
        "paths:",
        "workflow_run:",
        "continue-on-error",
        "|| true",
        "statuses:",
        "STEP_RESULTS",
        "RUN_CANCELLED",
        "gpu_pr.py pending",
        "gpu_pr.py report",
        "allow-unsafe-pr-checkout",
    ):
        assert forbidden not in text
    hai = (ROOT / ".github/scripts/hai_power.py").read_text()
    assert "/actions/runners" not in hai and "GH_RUNNER_READ_TOKEN" not in hai


def test_workflow_orders_shutdown_and_cleanup_after_failure():
    text = workflow()
    step_ids = re.findall(r"^        id: (\w+)$", text, re.M)
    assert (
        step_ids
        == (
            "control configure select sdk power ssh identity source upload current tests stop cleanup"
        ).split()
    )
    assert (
        'echo "attempted=true" >> "$GITHUB_OUTPUT"\n          python -u .github/scripts/hai_power.py start'
        in text
    )
    assert "if: ${{ always() && steps.power.outputs.attempted == 'true' }}" in text
    remote_cleanup = text[
        text.index("      - name: Clean remote workspace") : text.index(
            "      - name: Stop HAI"
        )
    ]
    assert (
        "if: ${{ always() && (steps.upload.outcome == 'failure' || steps.current.outcome == 'failure') }}"
        in remote_cleanup
    )
    assert "gpu_ssh.py cleanup" in remote_cleanup
    assert text.index("id: tests") < text.index("      - name: Clean remote workspace")
    stop = text[text.index("      - name: Stop HAI") :]
    assert "hai_power.py stop" in stop
    assert stop.index("hai_power.py stop") < stop.index("id: cleanup")
    assert (
        "if: always()" in stop
        and 'rm -f -- "$RUNNER_TEMP/fatqat-gpu-ssh/id_ed25519"' in stop
    )
    budgets = [
        int(value)
        for value in re.findall(r"^        timeout-minutes: (\d+)$", text, re.M)
    ]
    assert sum(budgets) < int(re.search(r"^    timeout-minutes: (\d+)$", text, re.M)[1])


def test_workflow_only_executes_trusted_control_locally():
    text = workflow()
    assert text.count("uses: actions/checkout") == 1
    assert (
        "ref: ${{ github.sha }}\n          path: control\n          persist-credentials: false"
        in text
    )
    assert "working-directory: control" in text
    assert "run: python -u .github/scripts/gpu_pr.py recheck" in text
    assert "pytest tests/" not in text and "pip install . " not in text


@pytest.mark.parametrize(
    "error",
    [
        URLError("secret"),
        TimeoutError("secret"),
        HTTPError("url", 403, "secret", {}, None),
    ],
)
def test_transport_errors_log_method_path_without_credentials(
    monkeypatch, error, capsys
):
    def fail(_opener, request, timeout):
        assert request.get_header("Authorization") == "Bearer job-token"
        assert timeout == 20
        raise error

    monkeypatch.setattr(
        gpu_pr, "build_opener", lambda *_: type("Opener", (), {"open": fail})()
    )
    api = gpu_pr.GitHub(
        {"GITHUB_REPOSITORY": "upstream/fatqat", "GH_TOKEN": "job-token"}
    )
    with pytest.raises(gpu_pr.APIError) as caught:
        api.request("/pulls/42")
    output = capsys.readouterr().out + str(caught.value)
    assert "GET /repos/upstream/fatqat/pulls/42" in output
    assert "secret" not in output and "job-token" not in output


def test_transport_only_reads_metadata_with_pinned_api_and_rejects_redirects(
    monkeypatch,
):
    requests = []

    def respond(_opener, request, timeout):
        requests.append(request)
        return io.BytesIO(b'{"number": 42}')

    monkeypatch.setattr(
        gpu_pr, "build_opener", lambda *_: type("Opener", (), {"open": respond})()
    )
    api = gpu_pr.GitHub(
        {
            "GITHUB_REPOSITORY": "upstream/fatqat",
            "GH_TOKEN": "job-token",
        }
    )
    assert api.request("/pulls/42") == {"number": 42}
    assert requests[-1].get_method() == "GET"
    assert requests[-1].data is None
    assert requests[-1].get_header("X-github-api-version") == "2022-11-28"
    assert (
        gpu_pr.NoRedirect().redirect_request(
            None, None, 302, "", {}, "https://other.example"
        )
        is None
    )
