# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`pax-ai` is the shared AI substrate for the `pax` project: summaries, semantic search, RAG, and an MCP server, all built on top of [`pax-core`](https://github.com/pax-project/pax-core) — never modifying it. [`lazy-pax`](https://github.com/pax-project/lazy-pax) is expected to depend on `pax-ai` for these features instead of implementing them internally, so every AI feature has one shared implementation instead of being duplicated between the TUI and any MCP client.

Full architecture rationale (why this is a separate crate instead of a `pax-core` feature flag) lives in the org's [`docs/README.md`](https://github.com/pax-project/.github/blob/main/docs/README.md#architecture-pax-ai) — read it before making structural decisions here, don't re-derive it locally.

## Non-negotiables (carried over from `pax-core`)

- Writes only ever go through `pax-core` functions — never hand-write `papers.nix`.
- Any persisted state (embeddings, cached summaries) is a rebuildable cache derived purely from `papers.nix` via `pax-core`'s read functions — never a second source of truth.
- Any LLM/API call runs off the UI/event thread (the same `spawn_blocking` pattern `pax-core`/`lazy-pax` use for provider search and `nix` calls) — callers must never block on a network round-trip.
- Ideas that don't need an LLM/embedding step aren't `pax-ai`'s concern — they belong in `pax-core` or `lazy-pax` directly.

## Planned architecture: ports and adapters for the LLM capability

The LLM/embedding capability is a port (trait), not a hardcoded client — cost, privacy, and self-hosting are per-deployment decisions (see "Cost/privacy" in the org docs), so nothing above the port should know which backend is in use.

- **Port:** a trait covering the operations `pax-ai` actually needs (e.g. summarize, embed) — shaped by real call sites, not spec'd upfront.
- **First adapter:** Anthropic (hosted). Sends abstracts/paper text to a third-party API — this is an explicit, deliberate choice, not a default; a local-model adapter (e.g. via Ollama) is the expected second adapter once the port's shape has settled against a real client.
- Adapter choice is a runtime/config concern, not a compile-time feature flag split, so a deployment can switch backends without a rebuild.

None of this is implemented yet — this repo is currently a bootstrap scaffold (`Cargo.toml`, `flake.nix`/`.envrc`, empty `src/lib.rs`). Update this section as the port/adapter split actually lands in code.

## Environment

Nix + direnv for the dev shell (`flake.nix`, `.envrc`), same pattern as `pax-core`/`lazy-pax`: `.envrc` does `use flake`, so `direnv allow` sets up cargo/rustc/rustfmt/clippy/rust-analyzer plus the OpenSSL env vars (`PKG_CONFIG_PATH`, `OPENSSL_DIR`) needed for native TLS dependencies.

## Commands

```bash
cargo build   # builds the pax-ai library
cargo test    # unit tests
cargo fmt     # format
cargo clippy  # lint
```

## Dependency on `pax-core`

crates.io dependency on the published `pax-core-papers` package, aliased back to `pax-core` (its lib name is still `pax_core`), `default-features = false` (the `cli` feature — and its `clap`/`dotenvy`/`tokio` deps — stays off), same as `lazy-pax`. Moving to a later `pax-core` release is a deliberate, explicit step, not automatic.
