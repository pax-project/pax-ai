# pax-ai

**pax-ai** is the shared AI substrate for the [`pax`](https://github.com/pax-project/pax-core) research-paper toolkit: summaries, semantic search, and RAG over a `pax-core` research library, plus an MCP server exposing the same capabilities to external clients (editors, agents, etc.).

It's a client of `pax-core`, the same way [`lazy-pax`](https://github.com/pax-project/lazy-pax) is: no shelling out, no reimplemented library logic, no shadow state. Any persisted state `pax-ai` needs (embeddings, cached summaries) is a rebuildable cache derived purely from `research/papers.nix` via `pax-core`'s read functions — never a second source of truth, and `pax-ai` never writes `papers.nix` directly.

`lazy-pax` is expected to depend on `pax-ai` for AI-heavy features instead of implementing them internally, giving every AI feature (TUI or MCP client) one shared implementation.

## Status

Early bootstrap — repo scaffold only, no functionality yet. See [`CLAUDE.md`](CLAUDE.md) for the planned architecture and the [org's `pax-ai` docs](https://github.com/pax-project/.github/blob/main/docs/README.md#architecture-pax-ai) for the full design rationale.

## Development

Nix devshell (`direnv allow`, or `nix develop` — same flake `pax-core`/`lazy-pax` use):

```Bash
cargo build
cargo test
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
