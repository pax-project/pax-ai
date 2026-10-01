# CI workflow

**File:** [`.github/workflows/ci.yml`](../../../.github/workflows/ci.yml)

Triggers on every push to `main` and every pull request. The entire job is:

```yaml
- uses: actions/checkout@v4
- uses: DeterminateSystems/nix-installer-action@main
- uses: DeterminateSystems/magic-nix-cache-action@main
- run: nix develop --command just ci
```

Install Nix, cache it, enter the devShell, run `just ci`. No check logic
lives here — that's the whole point of the facade pattern described in the
[index](../README.md): this file can't drift from what runs locally, because
it calls the exact same `just ci` a developer would run by hand, or that
[`pre-push`](../hooks/pre-push.md) already ran before the push ever reached
GitHub.

`just ci` is fmt + clippy + test + audit — see [`checks/`](../checks) for
each one. It is expected to currently fail on `audit`; see
[`checks/audit.md`](../checks/audit.md).
