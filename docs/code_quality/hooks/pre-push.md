# git `pre-push` hook

**File:** [`.githooks/pre-push`](../../../.githooks/pre-push)

```bash
exec nix develop --command just pre-push
```

Also a pure facade. `pre-push` is `ci` under another name (see
[`justfile`](../../../justfile)): fmt + clippy + test + **audit** — the one
check [`pre-commit`](pre-commit.md) skips. Push is the right place for it:
it's the last local checkpoint before `ci` runs it again anyway, and it's
infrequent enough that `audit`'s network calls don't add friction to the
inner edit/commit loop.

Wiring (`core.hooksPath`) is shared with `pre-commit` — see
[that doc](pre-commit.md#wiring).

## Known failure mode

Right now this hook (and `ci`) will fail on `just audit::check` — see
[`checks/audit.md`](../checks/audit.md) for why that's intentional rather
than a bug in the hook itself.
