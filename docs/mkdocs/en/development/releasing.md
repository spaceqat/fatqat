# Making a release

Use this guide when you are preparing a FatQat release. It walks you through choosing which changes to ship, checking the package that users will install, publishing it, and getting `main` ready for further development.

A release has three names you will use along the way. For example, the **release branch** `v0.2.x` can carry several releases in the same series. The **Git tag** `v0.2.0` identifies the exact commit used for one of them. The **package version** `0.2.0` is what a user gives to pip. For a prerelease such as `0.2.0rc1`, the branch stays the same and the tag becomes `v0.2.0rc1`.

You will open two pull requests. The first prepares the package on its release branch while development continues on `main`. After publication, the second brings the release notes and any release fixes back to `main`.

## Choose what to release

Start with the intended package version. It determines which branch to use and which changes belong in the release.

| What are you releasing? | Where to start |
| --- | --- |
| A patch such as `0.1.1` | Start from `v0.1.x` and backport the fixes intended for users of that series. |
| A new minor or major version such as `0.2.0` or `1.0.0` | Use the series branch if one already exists from a prerelease. Otherwise, create `v0.2.x` or `v1.0.x` from the commit on `main` that contains the intended changes. |
| An alpha, beta, or release candidate | Use the branch for the intended final release. Create it for the first prerelease if necessary, then reuse it throughout the series. |

Create any new series branch in `spaceqat/fatqat` so it can be the base of a PR. Check the repository's branch rules and the PyPI environment's allowed branches and tags for the new series. Keep earlier series branches available for maintenance.

Create your working branch from the selected release branch. During a prerelease cycle you can bring in further changes from `main` while they belong to the same series. Once development on `main` moves ahead, backport the individual fixes needed for the release.

## Prepare the release PR

Set `__version__` in `src/fatqat/__init__.py` to the version you want users to install. Drop the development suffix and retain any prerelease identifier, for example `0.2.0rc1`.

Add a dated entry to `CHANGELOG.md`. Choose the comparison point from the reader's perspective. Someone moving from one alpha or patch to the next needs the changes since the preceding tag in that series. Someone upgrading from one stable series to the next also needs the features introduced during the intervening prereleases. Compare the tagged contents when deciding whether a feature is new or an existing feature was updated. Include migration steps where an API changed, and credit external contributors alongside their changes with links to the relevant PRs.

Open the PR against the release branch. Run the [contribution checks](https://github.com/spaceqat/fatqat/blob/main/CONTRIBUTING.md#code-changes-and-validation) and [documentation validation](https://github.com/spaceqat/fatqat/blob/main/docs/mkdocs/README.md#validate-before-submission). After review and merge, confirm tests and lint pass and note the merged commit SHA. You will use it to check the release tag.

## Try the package before publishing

The release workflow produces two files, a wheel for installation and a source archive. Build them from the merged release branch first so you can try a clean installation before publishing.

1. Open [Build and optionally publish to PyPI](https://github.com/spaceqat/fatqat/actions/workflows/build.yaml) in `spaceqat/fatqat`.
2. Choose **Run workflow**, select the release branch, and leave `confirm_ref` blank. This runs the build and Twine checks only.
3. When the run succeeds, download and unpack the **distributions** artifact. Check that it contains the wheel and source archive for the intended version.

Install the downloaded wheel in a fresh Python environment, from a directory outside the source checkout. Replace the example path below with the path to the wheel.

```sh
python -m pip install "path/to/downloaded.whl"
python -c "import fatqat; print(fatqat.__version__)"
```

Confirm that the printed version matches the release PR, then run a small simulation from the quickstart. This exercises the installed package and its dependencies. Resolve any problems before tagging the release.

## Tag the tested commit on GitHub

Open [New release](https://github.com/spaceqat/fatqat/releases/new). Enter the tag (the package version prefixed with `v`), select the release branch as the target, and copy the changelog entry into the release notes. For an alpha, beta, or release candidate, select **Set as a pre-release**. Leave that option unchecked for a stable version.

Save the draft so the notes can be reviewed. A new tag entered in the draft is created when you click **Publish release**. Before publishing, check the release date and confirm the target still points to the commit you tested. If you selected an existing tag, check the commit that tag already identifies.

## Publish the tagged version to PyPI

Run the build workflow again, this time using the published tag. The workflow rebuilds the distributions from that tag and submits them to the PyPI publishing environment for approval.

1. In the workflow's branch/tag selector, select the **tag**. Enter the same tag, including the initial `v`, in `confirm_ref`. For example, for `0.2.0rc1`, both fields should read `v0.2.0rc1`.
2. Start the run. The workflow checks that the confirmation matches the selected ref and that the tag matches the version in the built package. Development versions and versions with a local suffix are rejected.
3. When the **publish** job requests approval, an authorized reviewer opens **Review deployments**, selects `pypi`, and clicks **Approve and deploy**. Allow for any wait timer configured on the environment.

After the job succeeds, find the version in the [PyPI release history](https://pypi.org/project/fatqat/#history) and check that both files are listed. In a fresh environment, install the published version, substituting its package version for `<version>`. Repeat the version check and the small simulation you ran with the downloaded wheel.

```sh
python -m pip install "fatqat==<version>"
python -c "import fatqat; print(fatqat.__version__)"
```

If the workflow fails, inspect the failed job and check which files, if any, reached PyPI before retrying.

## Bring the release back to main

Open the second PR against `main`. Include the published notes and any fixes made only on the release branch. Preserve newer changes already on `main`. The development version should tell someone installing from `main` what release series they are testing.

- If the next target is another prerelease of the same version, use its planned identifier followed by `.dev0`, for example `0.2.0rc2.dev0` after `0.2.0rc1`.
- After a stable minor or major release, use the next planned development version, for example `0.3.0.dev0` if development moves to `0.3`.
- After a maintenance release, keep the current development version if `main` already targets a later series.

Review and merge this PR after its checks pass. The release tag stays on the commit used to build the published package.
