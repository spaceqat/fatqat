"""Select a current PR after successful CPU CI and report its GPU result."""

import json
import os
from pathlib import Path
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

# Keep this list narrow: the current GPU suite exercises CuPy, not CPU engines.
# Extend it when GPU implementations or their build/dependency files are added.
GPU_PATHS = (
    "tests/gpu_tests/",
    "tests/test_gpu_pr.py",
    ".github/workflows/gpu.yml",
    ".github/scripts/gpu_pr.py",
    ".github/scripts/hai_power.py",
    "pyproject.toml",
)


class Ineligible(ValueError):
    """The completed run no longer identifies one eligible PR revision."""


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
        if not isinstance(data, (dict, list)):
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
    """Re-fetch run and PR metadata; never trust a PR number alone."""
    source = event["workflow_run"]
    run = api.request(f"/actions/runs/{number(source['id'])}")
    workflow = api.request("/actions/workflows/tests.yml")
    if (
        event.get("action") != "completed"
        or source.get("event") != "pull_request"
        or source.get("conclusion") != "success"
        or run.get("id") != source["id"]
        or run.get("run_attempt") != source["run_attempt"]
        or run.get("workflow_id") != workflow["id"]
        or run.get("event") != "pull_request"
        or run.get("status") != "completed"
        or run.get("conclusion") != "success"
        or run["repository"]["full_name"].lower() != api.repository.lower()
        or full_sha(run["head_sha"]) != full_sha(source["head_sha"])
    ):
        raise Ineligible("Not the completed successful PR attempt of Tests")

    head_sha = full_sha(run["head_sha"])
    fork_id = number(run["head_repository"]["id"])
    candidates = run.get("pull_requests") or []
    if not candidates:
        # Fork runs can have an empty pull_requests array. Query open PRs by
        # fork owner/branch, then check the repository ID and SHA as well.
        head = f"{run['head_repository']['owner']['login']}:{run['head_branch']}"
        page = 1
        while True:
            query = urlencode(
                {"state": "open", "head": head, "per_page": 100, "page": page}
            )
            batch = api.request(f"/pulls?{query}")
            if not isinstance(batch, list):
                raise APIError("Unexpected PR list")
            candidates.extend(batch)
            if len(batch) < 100:
                break
            page += 1

    matches = []
    for pr_number in {number(pr["number"]) for pr in candidates}:
        pr = api.request(f"/pulls/{pr_number}")
        head = pr.get("head") or {}
        if (
            pr.get("state") == "open"
            and number(pr["number"]) == pr_number
            and pr["base"]["repo"]["full_name"].lower() == api.repository.lower()
            and head.get("sha") == head_sha
            and str((head.get("repo") or {}).get("id")) == fork_id
            and head.get("ref") == run["head_branch"]
        ):
            matches.append(pr)
    if len(matches) != 1:
        raise Ineligible("No unique open PR at the successful Tests revision")
    pr = matches[0]
    if pr.get("mergeable") is not True:
        raise Ineligible("PR has conflicts or mergeability is not yet known")
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


def needs_gpu(api, pr_number):
    """Inspect the complete PR file list, including both sides of renames."""
    pr = api.request(f"/pulls/{pr_number}")
    count = pr["changed_files"]
    if type(count) is not int or not 0 <= count <= 3000:
        raise APIError("Cannot inspect the complete PR file list (GitHub limit: 3000)")
    files = []
    for page in range(1, (count + 99) // 100 + 1):
        batch = api.request(f"/pulls/{pr_number}/files?per_page=100&page={page}")
        if not isinstance(batch, list):
            raise APIError("Unexpected PR file list")
        files.extend(batch)
    if len(files) != count or len({item["filename"] for item in files}) != count:
        raise APIError("Incomplete or changing PR file list")
    required = False
    for item in files:
        paths = [item["filename"]]
        if item["status"] == "renamed":
            paths.append(item["previous_filename"])
        for path in paths:
            if not isinstance(path, str) or not path:
                raise APIError("Invalid changed-file path")
            if not path.lower().endswith((".md", ".rst")) and any(
                path.startswith(rule) if rule.endswith("/") else path == rule
                for rule in GPU_PATHS
            ):
                required = True
    return required


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
    elif environment["GITHUB_EVENT_NAME"] == "workflow_run":
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
        if prepare:
            required = needs_gpu(api, selected["pr_number"])
            # The file API describes a moving PR. Confirm the pinned version again
            # after pagination, including when deciding that GPU is unnecessary.
            if select_pr(api, event) != {
                k: v for k, v in selected.items() if k != "mode"
            }:
                raise Ineligible("PR version changed while inspecting changed files")
            selected["needs_gpu"] = str(required).lower()
            if required:
                check_gpu_environment(api)
            with Path(environment["GITHUB_STEP_SUMMARY"]).open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(
                    f"## GPU selection: PR #{selected['pr_number']}\n\n"
                    f"GPU required: **{'yes; review gpu-ci' if required else 'no'}**\n\n"
                    f"- Head: `{selected['head_sha']}`\n"
                    f"- Base: `{selected['base_sha']}`\n"
                    f"- Merge to test: `{selected['merge_sha']}`\n\n"
                    "This current merge is not verified as the merge tested by CPU Tests.\n"
                )
        elif environment.get("EXPECTED_PR"):
            check_gpu_environment(api)
        print(
            f"Selected PR #{selected['pr_number']}: head={selected['head_sha']}, "
            f"base={selected['base_sha']}, merge={selected['merge_sha']}. "
            "This is the current merge commit, not verified as the merge commit "
            "tested by the preceding CPU Tests run."
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
