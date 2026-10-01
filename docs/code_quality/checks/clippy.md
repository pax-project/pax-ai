# clippy — linting

**File:** [`just/clippy.just`](../../../just/clippy.just)

Runs `clippy` across all targets and features, with warnings as errors.

```
just clippy::check   # cargo clippy --all-targets --all-features -- -D warnings
just clippy::fix     # cargo clippy --fix --allow-dirty --allow-staged --all-targets --all-features
```

`check` is used by [`pre-commit`](../hooks/pre-commit.md) and `ci` (via
[`pre-push`](../hooks/pre-push.md) and the [CI workflow](../ci/workflow.md)).

`fix` only auto-corrects clippy's *machine-applicable* lints — anything
requiring judgment still has to fail `check` and be fixed by hand. It's
called from the root `fix` recipe (`fix: clippy::fix fmt::fix`), used by
Claude Code's [`PostToolUse` hook](../claude/post-tool-use.md): clippy fixes
first, then `fmt::fix` reformats whatever the fix pass left behind.

`--allow-dirty --allow-staged` is required here specifically because this
runs mid-edit, against a working tree that's expected to have uncommitted
changes — `cargo clippy --fix` refuses to touch a dirty tree otherwise.
