---
name: robolab-github-pr
description: >
  Review, fix up, and merge external GitHub pull requests on NVlabs/RoboLab,
  or self-check a contribution before submitting one. Use when asked to
  review a PR against the contributing rules, verify DCO sign-off and SPDX
  headers, push fixes to a contributor's fork, or merge a fork PR.
license: CC-BY-NC-4.0
compatibility: >
  Requires a clone of NVlabs/RoboLab with a GitHub remote; fork pushes
  require maintainer access to the base repository.
metadata:
  author: nvidia
  version: "1.0.0"
---

# RoboLab GitHub PRs

External PRs target `main` on `github.com/NVlabs/RoboLab`. The ground rules
live in [`CONTRIBUTING.md`](../../CONTRIBUTING.md); this skill makes the
review and merge mechanics concrete.

## Review checklist

- **DCO** — every commit carries a `Signed-off-by` line. No sign-off, no
  merge (see the DCO section of `CONTRIBUTING.md`).
- **SPDX headers** — every *new* Python file starts with the project's
  standard header, regardless of author:

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0
  ```

  The DCO sign-off carries the license grant; the header is uniform project
  boilerplate — contributors should not substitute their own copyright line.
- **Assets** — each object/task folder is self-contained, carries its
  inherited license, and is added to `THIRD_PARTY_NOTICES.md`. All object
  references must be relative paths.
- **Scope** — new policy backends need a published paper/project and prior
  maintainer contact via an Issue; core-framework or benchmark-altering
  changes need a prior Issue discussion.

## Pushing fixes to a contributor's fork (LFS gotcha)

Forks share LFS storage with the parent repo, so merging fork PRs that carry
LFS files (assets, robot USDs, images) works normally and does not leave
broken pointers. But a maintainer **cannot upload new LFS objects to a
fork**, even with maintainer-edit access — the git-lfs pre-push hook hits a
forbidden upload API and blocks *any* push to the fork, even a text-only
commit. Workaround:

1. Seed the object via the base repo, where you do have LFS access (shared
   storage makes it visible to the fork's endpoint too):
   `git push <base-remote> HEAD:refs/heads/<tmp-branch>`
2. Push to the fork skipping the pre-push hook (the object already exists,
   so nothing needs uploading): `git push --no-verify <fork-url> <sha>:<branch>`
3. Delete the scratch branch after merging; this does not remove the LFS
   objects.

Also: `git lfs pull` defaults to the `origin` remote. If your clone's
`origin` is not the GitHub repo, a 404 from it says nothing about GitHub —
always name the remote explicitly.

## Merging

Merge fork PRs with a **merge commit — never squash**. Squashing erases
per-commit authorship and the contributors' DCO sign-offs.
