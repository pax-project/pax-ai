# Code quality

How `pax-ai` enforces formatting, linting, tests, and a dependency-vulnerability
audit — and how that enforcement reaches you at three different points
(editing, committing, and CI) without being defined three different times.

## The shape of it

Every check's logic lives in exactly one file, under [`just/`](../../just/).
Everything else — [`justfile`](../../justfile), the git hooks, the CI
workflow, and the Claude Code hooks — is a **facade**: it calls into those
files through `just` and contains no check logic of its own. Change a check,
and all three entry points pick it up automatically.

```
just/{fmt,clippy,test,audit}.just   single-responsibility checks
        │
        ▼
     justfile                       composes them into ci / pre-commit / pre-push / fix
        │
        ├── .githooks/{pre-commit,pre-push}   git-time facade
        ├── .github/workflows/ci.yml          push/PR-time facade
        └── .claude/settings.json             edit-time / turn-end facade (Claude Code)
```

All the tools this depends on (`just`, `cargo-audit`, `jq`) are installed by
the Nix flake's devShell, not assumed to be on your machine — see
[`flake.nix`](../../flake.nix).

## Index

**Checks** — what each quality dimension actually runs:

- [`checks/fmt.md`](checks/fmt.md) — formatting (`rustfmt`)
- [`checks/clippy.md`](checks/clippy.md) — linting (`clippy`)
- [`checks/test.md`](checks/test.md) — correctness (the test suite)
- [`checks/audit.md`](checks/audit.md) — security (`cargo-audit`)

**Facades** — where those checks get called from:

- [`hooks/pre-commit.md`](hooks/pre-commit.md) — the git `pre-commit` hook
- [`hooks/pre-push.md`](hooks/pre-push.md) — the git `pre-push` hook
- [`ci/workflow.md`](ci/workflow.md) — the GitHub Actions workflow
- [`claude/post-tool-use.md`](claude/post-tool-use.md) — Claude Code's
  edit-time auto-fix hook
- [`claude/stop.md`](claude/stop.md) — Claude Code's turn-end gate

## Running it yourself

```bash
just --list      # see every recipe
just ci           # the full gate: fmt + clippy + test + audit
just pre-commit   # the fast subset: fmt + clippy + test
just fix           # auto-fix: clippy --fix, then fmt
```

(`direnv allow` gets you `just` on `PATH`; otherwise prefix with
`nix develop --command`.)
