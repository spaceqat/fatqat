"""Pin approved event revisions, archive source without executing it, and report."""

import base64
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from time import sleep
import traceback
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
    """Use only the job-scoped GITHUB_TOKEN."""

    def __init__(self, environment):
        self.environment = environment
        self.repository = environment["GITHUB_REPOSITORY"]
        if not re.fullmatch(r"[A-Za-z0-9-]+/[A-Za-z0-9_.-]+", self.repository):
            raise Ineligible("Invalid repository")
        self.opener = build_opener(NoRedirect())

    def request(self, path, payload=None):
        method = "GET" if payload is None else "POST"
        location = f"{method} /repos/{self.repository}{path}"
        print(f"GitHub API: {location}", flush=True)
        request = Request(
            f"https://api.github.com/repos/{self.repository}{path}",
            data=None if payload is None else json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.environment['GH_TOKEN']}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                # PR revision selection relies on merge_commit_sha.
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with self.opener.open(request, timeout=20) as response:
                data = json.load(response)
        except HTTPError as exc:
            raise APIError(
                f"{location}: HTTP {exc.code}; check job permissions"
            ) from None
        except (URLError, OSError, ValueError) as exc:
            raise APIError(
                f"{location}: request failed ({type(exc).__name__})"
            ) from None
        if not isinstance(data, dict):
            raise APIError(f"{location}: expected a JSON object")
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
    merge_sha = full_sha(pr["merge_commit_sha"])
    commit = api.request(f"/git/commits/{merge_sha}")
    parents = [full_sha(parent["sha"]) for parent in commit["parents"]]
    if commit.get("sha") != merge_sha or len(parents) != 2 or parents[1] != head_sha:
        raise Ineligible(
            "PR merge must have two parents and the event head as parent 2"
        )
    # GitHub's PR base.sha can advance ahead of its synthesized merge revision.
    base_sha = parents[0]
    return {
        "pr_number": number(pr["number"]),
        "head_sha": head_sha,
        "base_sha": base_sha,
        "merge_sha": merge_sha,
    }


def check_gpu_environment(api, mode):
    """Validate the Environment used by this already-approved job."""
    name = "gpu-ci" if mode == "pr" else "gpu-ci-manual"
    try:
        environment = api.request(f"/environments/{name}")
    except APIError as exc:
        raise APIError(
            f"Cannot verify {name} Environment; require Actions: read access. {exc}"
        ) from None
    rules = environment.get("protection_rules")
    if not isinstance(rules, list) or any(
        not isinstance(rule, dict) or not isinstance(rule.get("type"), str)
        for rule in rules
    ):
        raise APIError(f"{name} Environment: invalid or missing protection_rules array")
    reviewer_rules = [rule for rule in rules if rule["type"] == "required_reviewers"]
    if mode == "manual":
        if reviewer_rules:
            raise APIError(
                "Remove Required reviewers from the gpu-ci-manual Environment"
            )
        return
    for rule in reviewer_rules:
        reviewers = rule.get("reviewers")
        if not isinstance(reviewers, list) or any(
            not isinstance(item, dict)
            or item.get("type") not in {"User", "Team"}
            or not isinstance(item.get("reviewer"), dict)
            or not isinstance(item["reviewer"].get("id"), int)
            for item in reviewers
        ):
            raise APIError(
                "gpu-ci Environment: invalid required_reviewers.reviewers array"
            )
        if reviewers:
            return
    raise APIError("Configure Required reviewers on the gpu-ci Environment")


def select(api, event, prepare=False):
    environment = api.environment
    default_branch = event["repository"]["default_branch"]
    if environment["GITHUB_REF"] != f"refs/heads/{default_branch}":
        raise Ineligible("GPU control must run from the default branch")
    if environment["GITHUB_EVENT_NAME"] == "workflow_dispatch":
        selected = {"mode": "manual", "merge_sha": full_sha(environment["GITHUB_SHA"])}
    elif environment["GITHUB_EVENT_NAME"] == "pull_request_target":
        selected = {"mode": "pr", **select_pr(api, event)}
    else:
        raise Ineligible("Unsupported GPU trigger")
    check_gpu_environment(api, selected["mode"])
    if selected["mode"] == "pr":
        if prepare:
            with Path(environment["GITHUB_STEP_SUMMARY"]).open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(
                    f"## GPU selection: PR #{selected['pr_number']}\n\n"
                    f"- Head: `{selected['head_sha']}`\n"
                    f"- Base: `{selected['base_sha']}`\n"
                    f"- Merge to test: `{selected['merge_sha']}`\n\n"
                    "Selected after gpu-ci approval of this event's head.\n"
                )
        print(
            f"Selected PR #{selected['pr_number']}: head={selected['head_sha']}, "
            f"base={selected['base_sha']}, merge={selected['merge_sha']}."
        )
    return selected


def selection_path(environment):
    return Path(environment["RUNNER_TEMP"]) / "fatqat-gpu-revision.json"


def read_event(environment):
    return json.loads(
        Path(environment["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8")
    )


def recheck(api):
    pinned = json.loads(selection_path(api.environment).read_text(encoding="utf-8"))
    if select(api, read_event(api.environment)) != pinned:
        raise Ineligible(
            "Selected PR head, base or merge changed while waiting for GPU"
        )
    return pinned


def archive(api):
    """Fetch a PR merge ref into a bare repository; never check out PR files here."""
    environment = api.environment
    selected = json.loads(selection_path(environment).read_text(encoding="utf-8"))
    directory = tempfile.mkdtemp(
        prefix="fatqat-gpu-git-", dir=environment["RUNNER_TEMP"]
    )
    git_env = {
        key: environment[key]
        for key in ("PATH", "SYSTEMROOT", "TEMP", "TMP")
        if key in environment
    }
    git_env.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_TERMINAL_PROMPT="0",
    )

    def git(*args):
        try:
            return subprocess.run(
                ["git", "-C", directory, *args],
                env=git_env,
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            ).stdout.strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            # Git diagnostics are not needed to expose credentials or remote content.
            raise APIError(f"Source git {args[0]} failed or timed out") from None

    git("init", "--bare", "--template=")
    auth = base64.b64encode(
        f"x-access-token:{environment['GH_TOKEN']}".encode()
    ).decode()
    git_env.update(
        GIT_CONFIG_COUNT="1",
        GIT_CONFIG_KEY_0="http.https://github.com/.extraheader",
        GIT_CONFIG_VALUE_0=f"AUTHORIZATION: basic {auth}",
    )
    revision = full_sha(selected["merge_sha"])
    ref = (
        f"refs/pull/{number(selected['pr_number'])}/merge"
        if selected["mode"] == "pr"
        else revision
    )
    git(
        "fetch",
        "--no-tags",
        "--depth=2",
        f"https://github.com/{api.repository}.git",
        ref,
    )
    del git_env["GIT_CONFIG_VALUE_0"]
    git_env["GIT_CONFIG_COUNT"] = "0"
    if git("rev-parse", "FETCH_HEAD") != revision:
        raise Ineligible("Fetched PR merge ref no longer matches the selected commit")
    if selected["mode"] == "pr" and git(
        "show", "-s", "--format=%P", revision
    ).split() != [
        selected["base_sha"],
        selected["head_sha"],
    ]:
        raise Ineligible("Fetched merge parents do not match the pinned base and head")
    target = Path(environment["RUNNER_TEMP"]) / "fatqat-gpu-source.tar"
    git("archive", "--format=tar", f"--output={target}", revision)
    recheck(api)


def report(api):
    steps = json.loads(api.environment["STEP_RESULTS"])
    required = (
        "configure",
        "select",
        "sdk",
        "pending",
        "power",
        "ssh",
        "identity",
        "source",
        "upload",
        "current",
        "tests",
        "stop",
        "cleanup",
    )
    outcomes = {name: steps.get(name, {}).get("outcome") for name in required}
    if api.environment.get("RUN_CANCELLED") == "true":
        state = "error"
    elif all(value == "success" for value in outcomes.values()):
        state = "success"
    elif outcomes["tests"] == "failure" and all(
        value == "success" for name, value in outcomes.items() if name != "tests"
    ):
        state = "failure"
    else:
        state = "error"
    api.status(read_event(api.environment)["pull_request"]["head"]["sha"], state)


def main(arguments=None, environment=None):
    arguments = sys.argv[1:] if arguments is None else arguments
    environment = os.environ if environment is None else environment
    try:
        api = GitHub(environment)
        if arguments == ["prepare"]:
            selected = select(api, read_event(environment), prepare=True)
            selection_path(environment).write_text(
                json.dumps(selected), encoding="utf-8"
            )
            with Path(environment["GITHUB_OUTPUT"]).open(
                "a", encoding="utf-8"
            ) as stream:
                for key, value in selected.items():
                    stream.write(f"{key}={value}\n")
        elif arguments == ["recheck"]:
            recheck(api)
        elif arguments == ["archive"]:
            archive(api)
        elif arguments == ["pending"]:
            api.status(
                read_event(environment)["pull_request"]["head"]["sha"], "pending"
            )
        elif arguments == ["report"]:
            report(api)
        else:
            raise ValueError("Usage: gpu_pr.py prepare|recheck|archive|pending|report")
    except Ineligible as exc:
        print(f"GPU request rejected: {exc}", file=sys.stderr)
        return 1
    except APIError as exc:
        print(f"GPU control failed: {exc}", file=sys.stderr)
        return 1
    except Exception:
        # No locals, request headers, event dumps, or environment dumps.
        diagnostic = traceback.format_exc()
        for name in (
            "GH_TOKEN",
            "GPU_SSH_PRIVATE_KEY",
            "TENCENTCLOUD_SECRET_ID",
            "TENCENTCLOUD_SECRET_KEY",
        ):
            if environment.get(name):
                diagnostic = diagnostic.replace(environment[name], "<redacted>")
        print(diagnostic, file=sys.stderr, end="")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
