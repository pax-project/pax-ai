# pax-ai

**pax-ai** is the shared AI substrate for the [`pax`](https://github.com/pax-project/pax-core) research-paper toolkit: summaries, semantic search, and RAG over a `pax-core` research library, plus an MCP server exposing the same capabilities to external clients (editors, agents, etc.).

It's a client of `pax-core`, the same way [`lazy-pax`](https://github.com/pax-project/lazy-pax) is: no shelling out, no reimplemented library logic, no shadow state. Any persisted state `pax-ai` needs (embeddings, cached summaries) is a rebuildable cache derived purely from `research/papers.nix` via `pax-core`'s read functions — never a second source of truth, and `pax-ai` never writes `papers.nix` directly.

`lazy-pax` is expected to depend on `pax-ai` for AI-heavy features instead of implementing them internally, giving every AI feature (TUI or MCP client) one shared implementation.

## Status

Early: the LLM port with Anthropic and Ollama adapters, cached summaries, and a first MCP server (below). See [`CLAUDE.md`](CLAUDE.md) for the planned architecture and the [org's `pax-ai` docs](https://github.com/pax-project/.github/blob/main/docs/README.md#architecture-pax-ai) for the full design rationale.

## MCP server

`pax-mcp [ROOT]` serves the research library under `ROOT` (default: the current directory) over MCP on stdin/stdout. Tools so far: `search`, `show`, `add`, `list`. Results are JSON, and every write goes through `pax-core`, so the library stays usable from the `pax` CLI and lazy-pax.

```Bash
cargo build --release --bin pax-mcp
claude mcp add pax -- /path/to/pax-mcp /path/to/library
```

`SEMANTIC_SCHOLAR_API_KEY` and `ARXIV_CONTACT` are read from the environment, as for `pax`. Library consumers that don't need the server can depend on `pax-ai` with `default-features = false` to leave out the `mcp` feature and its dependencies.

Idea backlog lives in [Discussions → Ideas](https://github.com/pax-project/pax-ai/discussions/categories/ideas) (not Issues — see the org docs above for why), not in a `research.md`/`ideas.md` file.

## Development

Nix devshell (`direnv allow`, or `nix develop` — same flake `pax-core`/`lazy-pax` use):

```Bash
cargo build
cargo test
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
