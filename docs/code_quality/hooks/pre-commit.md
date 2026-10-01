# git `pre-commit` hook

**File:** [`.githooks/pre-commit`](../../../.githooks/pre-commit)

```bash
exec nix develop --command just pre-commit
```

That's the whole file — a facade with no check logic of its own. It runs
`just pre-commit` (fmt + clippy + test; see [`checks/`](../checks)) inside
the flake's devShell, so the right toolchain is guaranteed even if your
interactive shell's `direnv`/`PATH` state is stale.

## Wiring

Git doesn't look in `.githooks/` by default; something has to set
`core.hooksPath`. That's done by the flake devShell's `shellHook`
(see [`flake.nix`](../../../flake.nix)), so it's wired automatically the
moment you `direnv allow` or `nix develop` into this repo — no separate
install step.

## Why not run the full `ci` gate here

`pre-commit` deliberately excludes [`audit`](../checks/audit.md) (network
access, doesn't belong in a tight commit-time loop) — see
[`pre-push`](pre-push.md) for where that happens instead.
