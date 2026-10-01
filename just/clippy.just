# Linting — single responsibility: clippy.
#
# [no-cd]: see just/fmt.just.

[no-cd]
check:
    cargo clippy --all-targets --all-features -- -D warnings

# Auto-fix clippy's machine-applicable lints. --allow-dirty/--allow-staged:
# this runs mid-edit, against a working tree that's expected to be dirty.
[no-cd]
fix:
    cargo clippy --fix --allow-dirty --allow-staged --all-targets --all-features
