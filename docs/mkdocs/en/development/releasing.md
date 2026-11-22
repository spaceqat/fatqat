# Making a release

Keep `main` on a development version and prepare releases on a release branch. Use two pull requests: one to prepare the release, and one to update `main` afterward.

Before starting, choose the package version, the release branch, and the commit to include. The Git tag is the package version prefixed with `v`. In the instructions below, `<version>` and `<tag>` refer to those values.

## Prepare the release

### Choose the release branch

| Release | Branch and scope |
| --- | --- |
| Patch, such as `0.1.1` | Use the existing `v0.1.x` branch and bring in the intended fixes. Do not merge unrelated development from `main`. |
| New minor or major series, such as `0.2.0` or `1.0.0` | Create `v0.2.x` or `v1.0.x` from the chosen `main` commit. If the series already has a branch from its prereleases, use that branch. |
| Alpha, beta, or release candidate | Use the branch for the intended final release. The first prerelease may need a new branch; later prereleases use the same one. |

For a new series, create the release branch in `spaceqat/fatqat` before opening the release PR. Check that the repository's branch rules and publishing environment allow the new branch and its tags. Keep the older series branch for maintenance releases.

Create a working branch from the selected release branch. Add any remaining release fixes there. During a prerelease cycle, merge further changes from `main` only while they still belong to that series; once `main` has moved on, backport individual fixes instead.

### Set the version and release notes

- Set `__version__` in `src/fatqat/__init__.py` to the package version being released, without a development suffix.
- Add a dated entry to `CHANGELOG.md`. For a patch or prerelease, compare against the preceding release in that series. For a new stable minor or major release, summarize the changes since the previous stable series, including work introduced during prereleases. Use tagged contents rather than recent commit messages to distinguish new features from updates.
- Describe user-facing changes and migration steps. Credit external contributors beside their contributions, with links to their PRs.

Open a PR targeting the release branch. Follow the [contribution checks](https://github.com/spaceqat/fatqat/blob/main/CONTRIBUTING.md#code-changes-and-validation) and [documentation validation](https://github.com/spaceqat/fatqat/blob/main/docs/mkdocs/README.md#validate-before-submission), then review and merge the PR. Confirm that tests and lint pass on the merged commit.

## Check the package build

In `spaceqat/fatqat`, open [Build and optionally publish to PyPI](https://github.com/spaceqat/fatqat/actions/workflows/build.yaml).

1. Click **Run workflow** and select the release branch.
2. Leave `confirm_ref` blank and start the run.
3. Check that the build succeeds. The `distributions` artifact contains the wheel and source archive; the workflow checks both with Twine.

Install the wheel in a fresh environment and check the version and a small simulation. A successful package build does not check runtime dependencies or replace the test suite. Resolve failed release checks before publishing.

## Create the GitHub release

Open [New release](https://github.com/spaceqat/fatqat/releases/new). Enter `<tag>`, set the target to the release branch, and use the changelog entry as the release notes. Mark alpha, beta, and release-candidate versions as **pre-releases**; leave that option unchecked for a stable release.

Save a draft for review. Saving a draft does not create a new tag; GitHub creates it when the release is published. Before clicking **Publish release**, confirm that the target still points to the commit you validated and that the release date is correct. If the tag already exists, check its commit instead.

## Publish to PyPI

Start a new run of the same build workflow:

1. Select the published **release tag** in the branch/tag selector.
2. Enter that exact tag name in `confirm_ref` and run the workflow.
3. When the `publish` job waits for approval, an authorized reviewer opens **Review deployments**, selects `pypi`, and clicks **Approve and deploy**. The environment may also impose a wait timer.

Selecting a branch but entering a tag name fails confirmation. The workflow also checks that the tag matches the package version with a `v` prefix and rejects development and local versions.

Check that the `publish` job succeeds and that [PyPI](https://pypi.org/project/fatqat/#history) lists the new version with both a wheel and a source archive. Verify an installation from PyPI in a fresh environment, replacing `<version>` below with the published package version:

```sh
python -m pip install "fatqat==<version>"
python -c "import fatqat; print(fatqat.__version__)"
```

Run a small simulation as well. If publication fails, inspect the failed job before starting another run; check PyPI first in case files were already uploaded.

## Update main

Open the second PR against `main`. Bring back the release notes and any fixes made only on the release branch, preserving newer changes on `main`.

- After a prerelease, use the next planned prerelease followed by `.dev0` if `main` is still working toward the same final release.
- After a stable minor or major release, set `main` to the next planned development version. For example, use `0.3.0.dev0` after `0.2.0` if the next development series is `0.3`.
- After a maintenance release, keep `main`'s version if it already targets a later series. Publishing `0.1.1` should not move `main` back from `0.2.0.dev0`.

Keep the published tag unchanged. Review and merge the follow-up PR after its checks pass.
