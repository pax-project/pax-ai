# Facade: the one calling interface quality tools, git hooks, and CI all go
# through. Each check's own rules live in just/<check>.just (single
# responsibility per file); this file only composes them.

mod fmt 'just/fmt.just'
mod clippy 'just/clippy.just'
mod test 'just/test.just'
mod audit 'just/audit.just'

default:
    @just --list

# Full quality gate — the one thing CI and `pre-push` both run.
ci: fmt::check clippy::check test::run audit::check

# Fast subset for the pre-commit hook (and Claude Code's Stop hook):
# everything short of the network-dependent audit check.
pre-commit: fmt::check clippy::check test::run

# Same gate as CI, run locally before a push.
pre-push: ci

# Auto-fix entry point for Claude Code's PostToolUse hook: clippy's
# machine-applicable fixes first, then reformat the result.
fix: clippy::fix fmt::fix
