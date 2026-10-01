# Formatting — single responsibility: rustfmt.
#
# [no-cd]: `just` otherwise cds into this module file's own directory
# before running the recipe; keep every module running from wherever
# `just` was actually invoked, so behavior doesn't depend on cargo's own
# upward manifest search.

[no-cd]
check:
    cargo fmt --all -- --check

[no-cd]
fix:
    cargo fmt --all
