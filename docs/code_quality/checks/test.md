# test — correctness

**File:** [`just/test.just`](../../../just/test.just)

Runs the crate's test suite (unit + doc tests), all features enabled.

```
just test::run   # cargo test --all-features
```

Used by [`pre-commit`](../hooks/pre-commit.md) (and so also by Claude Code's
[`Stop` hook](../claude/stop.md), which calls `pre-commit`) and by `ci` (via
[`pre-push`](../hooks/pre-push.md) and the [CI workflow](../ci/workflow.md)).

It's in the fast `pre-commit` subset, not just the full `ci` gate, because
this crate's suite runs in effectively zero time — there's no latency
trade-off to leaving it out of the tighter loop.
