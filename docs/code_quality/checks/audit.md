# audit — security

**File:** [`just/audit.just`](../../../just/audit.just)

Checks the full dependency tree against the RustSec advisory database.

```
just audit::check   # cargo audit
```

Used only by `ci` (via [`pre-push`](../hooks/pre-push.md) and the
[CI workflow](../ci/workflow.md)) — deliberately **not** by
[`pre-commit`](../hooks/pre-commit.md) or Claude Code's
[`Stop` hook](../claude/stop.md). It needs network access (to fetch the
advisory DB and refresh the crates.io index), which doesn't belong in the
tight commit/edit-time loop.

## Known state: currently failing, on purpose

As of this writing, `cargo audit` reports 12 RUSTSEC advisories. Every one of
them is in `pax-core`'s own transitive dependency tree (old `hyper`/`tokio`
0.1-era crates, `h2`, `quick-xml`, pulled in via `crossref`) — nothing
`pax-ai` can fix without bumping the `pax-core` pin, which per `CLAUDE.md` is
"a deliberate, explicit step, not automatic."

This check is still a **hard blocker** in `ci`/`pre-push`, not ignored or
downgraded to a warning: that's an intentional choice, so the red build
stays a visible, continuous nudge to bump the pin, rather than being quietly
suppressed. Don't add an `--ignore RUSTSEC-...` list here to make it pass —
that was considered and rejected.

## The `[no-cd]` bug

This check is the reason every module in `just/` carries `[no-cd]`. `just`
changes directory into a module file's own location (`just/`) before running
its recipe by default. `cargo fmt`/`clippy`/`test` don't notice, because
cargo searches upward for `Cargo.toml` on its own — but `cargo audit` doesn't
do that upward search for `Cargo.lock`, so without `[no-cd]` it failed with
`Couldn't load Cargo.lock` every time, regardless of where `just` was
actually invoked from.
