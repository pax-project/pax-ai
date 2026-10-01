# fmt — formatting

**File:** [`just/fmt.just`](../../../just/fmt.just)

Runs `rustfmt` across the whole crate.

```
just fmt::check   # cargo fmt --all -- --check   (fails if anything is unformatted)
just fmt::fix     # cargo fmt --all               (reformats in place)
```

Used by: [`pre-commit`](../hooks/pre-commit.md), `ci`
(via [`pre-push`](../hooks/pre-push.md) and the
[CI workflow](../ci/workflow.md)), and Claude Code's
[`PostToolUse` hook](../claude/post-tool-use.md) (as part of `just fix`).

## `[no-cd]`

`just` changes directory into a module file's own location (`just/`) before
running its recipe by default. `cargo fmt` doesn't care — it searches upward
for `Cargo.toml` on its own — but the attribute is here anyway so this
module's behavior doesn't *accidentally* depend on that upward search; see
[`checks/audit.md`](audit.md) for the check where this distinction actually
bit us.
