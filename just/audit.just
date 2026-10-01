# Security — single responsibility: known-vulnerability audit of the dependency tree.
#
# [no-cd]: see just/fmt.just.

[no-cd]
check:
    cargo audit
