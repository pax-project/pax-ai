# Claude Code `Stop` hook

**File:** [`.claude/settings.json`](../../../.claude/settings.json) → `hooks.Stop`

Fires when Claude Code is about to end a turn. Runs
[`just pre-commit`](../hooks/pre-commit.md) (fmt + clippy + test — see
[`checks/`](../checks)); on failure it exits `2` with the check output as
the reason, which blocks the turn from ending and feeds the failure back so
Claude fixes it before stopping.

```bash
nix develop --command bash -c '
  out=$(just pre-commit 2>&1); code=$?
  if [ "$code" -ne 0 ]; then
    echo "$out" >&2
    exit 2
  fi
'
```

Deliberately calls `pre-commit`, not the full `ci`: `audit` needs network
access and is a known, intentional failure right now (see
[`checks/audit.md`](../checks/audit.md)) — making `Stop` depend on it would
block *every* turn for reasons unrelated to what Claude just wrote.

## Why blocking, not advisory

This was a deliberate choice over a non-blocking/informational version: a
failure here means Claude's own edit broke formatting, linting, or a test,
and the point is for Claude to fix it itself before handing the turn back —
not to surface a warning the user then has to act on.

## Reload caveat

See [`post-tool-use.md`](post-tool-use.md#reload-caveat) — same watcher
limitation applies here.
