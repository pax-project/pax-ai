# Claude Code `PostToolUse` hook

**File:** [`.claude/settings.json`](../../../.claude/settings.json) → `hooks.PostToolUse`

Fires after every `Edit` or `Write` tool call. If the touched file is a
`.rs` file, it runs `just fix` (clippy's auto-fixable lints, then `rustfmt`
— see [`checks/clippy.md`](../checks/clippy.md) and
[`checks/fmt.md`](../checks/fmt.md)); otherwise it's a silent no-op.

```bash
nix develop --command bash -c '
  jq -r ".tool_input.file_path // empty" \
    | { read -r f; case "$f" in *.rs) just fix ;; esac; }
' >/dev/null 2>&1 || true
```

- Runs inside `nix develop` for the same reason every other facade does:
  the right toolchain, regardless of ambient shell state.
- `jq` parses the hook-event JSON (`tool_input.file_path`) that Claude Code
  passes on stdin. It's a flake devShell dependency for exactly this reason
  — see [`flake.nix`](../../../flake.nix).
- Deliberately silent and non-blocking (`>/dev/null 2>&1 || true`): this is
  an auto-fix, not a gate. The actual gate is the
  [`Stop` hook](stop.md).

## Why this exists alongside the `Stop` hook

Auto-fixing at edit time means most of what the `Stop` gate checks is
already clean by the time it runs — so `Stop` blocks less often, and when it
does block, it's on something this hook genuinely can't auto-fix (a lint
needing judgment, a broken test).

## Reload caveat

Claude Code's settings watcher only watches `.claude/` if it existed when
the session started. If you create or edit `.claude/settings.json` mid
session, run `/hooks` once (or restart) to pick it up — editing the file
alone won't make a running session notice.
