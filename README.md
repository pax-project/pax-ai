# pax-ai

**pax-ai** is the shared AI substrate for the [`pax`](https://github.com/pax-project/pax-core) research-paper toolkit: summaries, semantic search, and RAG over a `pax-core` research library, plus an MCP server exposing the same capabilities to external clients (editors, agents, etc.).

It's a client of `pax-core`, the same way [`lazy-pax`](https://github.com/pax-project/lazy-pax) is: no shelling out, no reimplemented library logic, no shadow state. Any persisted state `pax-ai` needs (embeddings, cached summaries) is a rebuildable cache derived purely from `research/papers.nix` via `pax-core`'s read functions — never a second source of truth, and `pax-ai` never writes `papers.nix` directly.

`lazy-pax` is expected to depend on `pax-ai` for AI-heavy features instead of implementing them internally, giving every AI feature (TUI or MCP client) one shared implementation.

## Status

Early: the LLM port with Anthropic and Ollama adapters, cached summaries, and a first MCP server (below). See [`CLAUDE.md`](CLAUDE.md) for the planned architecture and the [org's `pax-ai` docs](https://github.com/pax-project/.github/blob/main/docs/README.md#architecture-pax-ai) for the full design rationale.

## MCP server

`pax-mcp [ROOT]` serves the research library under `ROOT` (default: the current directory) over MCP on stdin/stdout. Results are JSON, and every write goes through `pax-core`, so the library stays usable from the `pax` CLI and lazy-pax. Tools:

- `search`: by `query` or `author`, with `limit` (default 50 per provider) or `all`, and `year` (`2020`, `2015-2020`, `2015-`, `-2020`), `type`, and `language` filters. Returns each provider's `retrieved`/`total` counts and how it applied each filter, the merge counts, and one list merged across providers: each paper with the providers and references it came from, flagged `in_library`, `excluded`, and `language_unknown`.
- `refs`, `cited_by`: a seed paper's references or citing papers, from OpenAlex and Semantic Scholar. The seed is a provider reference or a library citation key; paging, filters, and result shape are `search`'s.
- `show`: one paper, by provider reference or citation key.
- `add`: declare a search result in `research/papers.nix`. A paper already in the library is refused.
- `exclude`: record a paper in `research/excluded.nix` with a required `reason` and a `stage` (`screening` or `full-text`), and an optional `title`.
- `list`: declared papers, filtered by author, year, and tag.
- `stats`: the review's PRISMA 2020 flow-diagram counts (records identified per provider, duplicates removed, records screened, exclusions per stage with full-text reasons, reports not retrieved, papers included), computed by pax-core from the library's files.

`search`, `refs`, and `cited_by` are recorded in `research/search-log.jsonl`, the same log the `pax` CLI writes, and each result's `log` says whether it was recorded and, if not, why.

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
