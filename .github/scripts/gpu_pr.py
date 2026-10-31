"""Pin a PR event revision for GPU approval and report its GPU result."""

import json
import os
from pathlib import Path
import re
import sys
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener


class Ineligible(ValueError):
    """The PR event no longer identifies an eligible revision."""


class APIError(RuntimeError):
    """GitHub could not confirm or record the run."""


class NoRedirect(HTTPRedirectHandler):
    """Do not forward the job token to redirects."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def full_sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise Ineligible("Expected a full commit SHA")
    return value.lower()


def number(value):
    if not re.fullmatch(r"[1-9][0-9]*", str(value)):
        raise Ineligible("Expected a positive GitHub ID")
    return str(value)


class GitHub:
    """Use only the job-scoped GITHUB_TOKEN, never the runner-read PAT."""

    def __init__(self, environment):
        self.environment = environment
        self.repository = environment["GITHUB_REPOSITORY"]
        if not re.fullmatch(r"[A-Za-z0-9-]+/[A-Za-z0-9_.-]+", self.repository):
            raise Ineligible("Invalid repository")
        self.opener = build_opener(NoRedirect())

    def request(self, path, payload=None):
        request = Request(
            f"https://api.github.com/repos/{self.repository}{path}",
            data=None if payload is None else json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.environment['GH_TOKEN']}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2026-03-10",
            },
        )
        try:
            with self.opener.open(request, timeout=20) as response:
                data = json.load(response)
        except HTTPError as exc:
            raise APIError(f"GitHub returned HTTP {exc.code}") from exc
        except (URLError, OSError, ValueError) as exc:
            raise APIError("GitHub request failed") from exc
        if not isinstance(data, dict):
            raise APIError("Unexpected GitHub response")
        return data

    def status(self, sha, state):
        run_id = number(self.environment["GITHUB_RUN_ID"])
        self.request(
            f"/statuses/{full_sha(sha)}",
            {
                "state": state,
                "context": "GPU tests",
                "description": f"GPU tests: {state}; see workflow for test and shutdown results",
                "target_url": f"https://github.com/{self.repository}/actions/runs/{run_id}",
            },
        )


def select_pr(api, event):
    """Match the current PR to this event's head, then pin its merge commit."""
    source = event["pull_request"]
    pr_number = number(event["number"])
    head_sha = full_sha(source["head"]["sha"])
    if (
        event.get("action") not in {"opened", "synchronize", "reopened"}
        or number(source["number"]) != pr_number
    ):
        raise Ineligible("Not a supported PR event")
    for attempt in range(5):
        pr = api.request(f"/pulls/{pr_number}")
        if (
            pr.get("state") != "open"
            or number(pr["number"]) != pr_number
            or pr["base"]["repo"]["full_name"].lower() != api.repository.lower()
            or full_sha(pr["head"]["sha"]) != head_sha
        ):
            raise Ineligible("PR is closed or no longer matches the event revision")
        if pr.get("mergeable") is not None:
            break
        if attempt < 4:
            sleep(2)
    else:
        raise APIError(
            "PR mergeability is still unknown after 5 queries; rerun the GPU workflow"
        )
    if pr.get("mergeable") is not True:
        raise Ineligible("PR has merge conflicts")
    base_sha = full_sha(pr["base"]["sha"])
    merge_sha = full_sha(pr["merge_commit_sha"])
    commit = api.request(f"/git/commits/{merge_sha}")
    if commit.get("sha") != merge_sha or [
        parent["sha"] for parent in commit["parents"]
    ] != [base_sha, head_sha]:
        raise Ineligible("Merge commit does not match the current PR head and base")
    return {
        "pr_number": number(pr["number"]),
        "head_sha": head_sha,
        "base_sha": base_sha,
        "merge_sha": merge_sha,
    }


def check_gpu_environment(api):
    """Fail closed if gpu-ci has no Required reviewers or cannot be queried."""
    environment = api.request("/environments/gpu-ci")
    if not any(
        rule.get("type") == "required_reviewers" and rule.get("reviewers")
        for rule in environment["protection_rules"]
    ):
        raise APIError("Configure Required reviewers on the gpu-ci Environment")


def select(api, event, prepare=False):
    environment = api.environment
    default_branch = event["repository"]["default_branch"]
    if environment["GITHUB_REF"] != f"refs/heads/{default_branch}":
        raise Ineligible("GPU control must run from the default branch")
    if environment["GITHUB_EVENT_NAME"] == "workflow_dispatch":
        selected = {"mode": "manual"}
    elif environment["GITHUB_EVENT_NAME"] == "pull_request_target":
        selected = {"mode": "pr", **select_pr(api, event)}
    else:
        raise Ineligible("Unsupported GPU trigger")
    if environment.get("EXPECTED_PR") and (
        selected.get("pr_number") != environment["EXPECTED_PR"]
        or any(
            selected.get(field) != environment.get(f"EXPECTED_{field.upper()}")
            for field in ("head_sha", "base_sha", "merge_sha")
        )
    ):
        raise Ineligible(
            "Selected PR head, base or merge changed while waiting for GPU"
        )
    if selected["mode"] == "pr":
        check_gpu_environment(api)
        if prepare:
            with Path(environment["GITHUB_STEP_SUMMARY"]).open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(
                    f"## GPU selection: PR #{selected['pr_number']}\n\n"
                    f"- Head: `{selected['head_sha']}`\n"
                    f"- Base: `{selected['base_sha']}`\n"
                    f"- Merge to test: `{selected['merge_sha']}`\n\n"
                    "Review this fixed revision in the gpu-ci Environment.\n"
                )
        print(
            f"Selected PR #{selected['pr_number']}: head={selected['head_sha']}, "
            f"base={selected['base_sha']}, merge={selected['merge_sha']}."
        )
    with Path(environment["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
        for key, value in {**selected, "ready": "true"}.items():
            stream.write(f"{key}={value}\n")


def report(api):
    start, gpu_job, test, stop = (
        api.environment.get(f"{part}_RESULT")
        for part in ("START", "GPU_JOB", "TEST", "STOP")
    )
    if start == gpu_job == test == stop == "success":
        state = "success"
    elif (
        start == stop == "success"
        and test == "failure"
        and gpu_job in {"success", "failure"}
    ):
        state = "failure"
    else:
        state = "error"
    api.status(api.environment["PR_HEAD_SHA"], state)


def main(arguments=None, environment=None):
    arguments = sys.argv[1:] if arguments is None else arguments
    environment = os.environ if environment is None else environment
    try:
        api = GitHub(environment)
        if arguments in (["prepare"], ["select"]):
            event = json.loads(
                Path(environment["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8")
            )
            select(api, event, prepare=arguments == ["prepare"])
        elif arguments == ["pending"]:
            api.status(environment["PR_HEAD_SHA"], "pending")
        elif arguments == ["report"]:
            report(api)
        else:
            raise ValueError("Usage: gpu_pr.py prepare|select|pending|report")
    except Ineligible as exc:
        print(f"GPU request skipped: {exc}")
        return 0 if arguments in (["prepare"], ["select"]) else 1
    except APIError as exc:
        print(f"GPU control failed: {exc}", file=sys.stderr)
        return 1
    except (OSError, KeyError, TypeError, ValueError, AttributeError):
        print(
            "GPU control failed; check GitHub availability and event metadata",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
