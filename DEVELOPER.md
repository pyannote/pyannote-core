# Development and releases

## Local development

Install [uv](https://docs.astral.sh/uv/), then run:

```sh
uv sync --extra test
uv run pytest tests
```

Python 3.11 is the minimum supported version and the default for local builds.
CI tests Python 3.11–3.15 plus a `3.x` lane that automatically follows the
latest stable Python 3 release available to `actions/setup-python`. Package
metadata uses `requires-python = ">=3.11"`, with no upper bound. Future releases
are allowed; compatibility is checked by that latest-stable CI lane as they
become available. Prerelease interpreters are not included.

To build the wheel and source archive:

```sh
uv build
```

## How releases work

The workflow follows this pattern: publish a GitHub Release, build
once, run tests and a version guard, then publish the checked artifacts.
The destination here is public PyPI.

The entry point is [ci.yml](.github/workflows/ci.yml), named **CI**.
It runs on pull requests and published GitHub Releases, including prereleases. Branch
pushes, tag pushes, and draft releases do not trigger publication.

1. Build one wheel and one source distribution with `uv build` via
   [build-wheels.yml](.github/workflows/build-wheels.yml).
2. Install that wheel with its test extras and run the tests on all configured
   Python versions via [test.yml](.github/workflows/test.yml).
3. On releases, [version-guard.yml](.github/workflows/version-guard.yml)
   verifies that the wheel version matches the release tag and
   compares both artifacts with PyPI's recorded SHA-256 hashes. The check is
   implemented inline in the workflow, following `on-premise`. An existing
   filename with different contents fails the release; identical files are
   allowed for retries. Network errors other than HTTP 404 fail the check.
4. After tests and the version guard succeed, attach those distributions to
   the existing GitHub Release, then upload them to PyPI using trusted
   publishing. Identical files already on PyPI are skipped.

[lint.yml](.github/workflows/lint.yml) runs Ruff checks and
format checks on PRs in parallel with the build. Findings are advisory
(`continue-on-error`) until the existing codebase is clean.

The final `ci-complete` job reports failures from any required job. Pull
requests build and test but skip release-only jobs. Build artifacts are kept
for three days on PRs and 90 days on releases.

## Versioning

The Git tag is the single source of truth for the package version.
`pyproject.toml` declares a dynamic version, and `hatch-vcs` derives it from
Git metadata. There is no version field to bump before a release.

A clean checkout tagged `6.1.0` or `v6.1.0` builds version `6.1.0`; a tag such
as `6.1.0rc1` builds a release candidate. Untagged commits get development
versions derived from the preceding tag and commit history. Uncommitted
changes can also affect the computed version, so release from a clean tree.

The build and documentation workflows fetch full Git history and tags.
For local builds, use a Git clone with tags available (`git fetch --tags`;
use `git fetch --unshallow` first if the clone is shallow). Published source
distributions retain their version metadata and can build without Git.
The version guard still verifies that the built wheel matches the GitHub
Release tag before publication and rejects conflicting existing PyPI files.

## Making a release

Use `main` as the single long-lived branch. Merge changes into `main`, then
create the release tag directly on GitHub when you are ready to publish.
Check [one-time repository configuration](#one-time-repository-configuration)
before the first release.

### 1. Merge into main

Open a pull request targeting `main`. Include any changelog updates in
`CHANGELOG.md`, wait for `ci-complete` and the Documentation build
to pass, and merge the PR. No version bump or dedicated release branch is
needed: the tag determines the package version.

Make sure all changes intended for the release are on `main`. The release
workflow will build and test the tagged commit again before uploading to PyPI.

### 2. Create the tag and publish on GitHub

1. Open the repository's **Releases** page and click **Draft a new release**.
2. Under **Choose a tag**, enter a new version such as `6.1.0` and select
   **Create new tag**. Set **Target** to `main`.
3. Enter a release title and notes, or use **Generate release notes**.
4. For a release candidate, use a tag such as `6.1.0rc1` and select
   **This is a pre-release**.
5. Click **Publish release**. GitHub creates the tag, and CI starts the build,
   tests, version guard, and upload to **public PyPI**.

Choose an unused version. Tags may have an optional `v` prefix. Creating a
standalone tag or saving a draft does not publish to PyPI; publishing the
GitHub Release does. Published prereleases also upload to PyPI.

### 3. Verify publication

Monitor the **CI** run in GitHub Actions. Approve the `pypi` environment
deployment if repository settings require it. After the run succeeds, check
the version on [PyPI](https://pypi.org/project/pyannote-core/) and the wheel
and source archive attached to the GitHub Release. Attachments alone do not
confirm PyPI publication.

For failures, follow [Failed releases and retries](#failed-releases-and-retries).
Continue development on `main`; no merge back is needed. Start a fresh
`Next version` changelog section when recording subsequent changes.

## Documentation

Documentation remains in [doc.yml](.github/workflows/doc.yml). Pull requests
build the documentation without publishing it. Pushes to `main` build it
and deploy the resulting artifact to the existing `gh-pages` branch. Sphinx
warnings remain visible but are not treated as errors. Deployment alone gets
write permission.

To reproduce the documentation build locally:

```sh
uv run --extra doc sphinx-build -b html doc/source doc/build/html
```

## One-time repository configuration

Set `main` as the default branch on GitHub and apply the desired branch
protection rules to it.

The PyPI project must authorize this
[trusted publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/):

- Owner: `pyannote`
- Repository: `pyannote-core`
- Workflow filename: `ci.yml`
- GitHub environment: `pypi`

Before the first release, update the PyPI trusted publisher workflow filename
from `release.yml` to `ci.yml`. The repository and `pypi` environment are
unchanged. This configuration change is required for publication to succeed.
No PyPI API token is required. The publishing job requests `id-token: write` for OIDC.
Unlike `on-premise`, the upload job stays in this top-level workflow because
[PyPI trusted publishing does not support reusable publishing workflows](https://docs.pypi.org/trusted-publishers/troubleshooting/).
GitHub Release attachments now contain the wheel and source archive.
Release runs are not automatically cancelled when another run starts, to
avoid interrupting publication.

Check the existing configuration in PyPI and GitHub; it is not stored in this
repository. Keep any desired deployment approval rules on `pypi`.

Update branch protection to require `ci-complete` if it currently requires
checks from the old standalone test workflow. Also require the Documentation
build check if documentation validation should block merges. The release
workflow must be present in the tagged commit. To trigger a release through automation, use a
maintainer credential or GitHub App that can trigger subsequent workflows;
releases created with a workflow's default `GITHUB_TOKEN` do not start this
release workflow.

## Failed releases and retries

Inspect the failed job before retrying. Re-run failed jobs in the original
Actions run to reuse the existing build artifacts. Publishing uses
[`uv publish --check-url`](https://docs.astral.sh/uv/guides/publish/) so a
partial upload can be resumed when existing files have identical contents.
If attachment or upload fails, rerun the failed publishing job; it reuses
the original distribution artifact.

If the version guard detects different contents for an existing filename,
make a new version and a new release. Do not move a published tag or attempt
to replace PyPI files. Rebuilding can produce different bytes if build tools
change; prefer retrying the original run over rebuilding a published version.
A failed run does not undo files that were already uploaded to PyPI.
