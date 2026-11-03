# pax-ai

**pax-ai** is the shared AI substrate for the [`pax`](https://github.com/pax-project/pax-core) research-paper toolkit: summaries, semantic search, and RAG over a `pax-core` research library, plus an MCP server exposing the same capabilities to external clients (editors, agents, etc.).

It's a client of `pax-core`, the same way [`lazy-pax`](https://github.com/pax-project/lazy-pax) is: no shelling out, no reimplemented library logic, no shadow state. Any persisted state `pax-ai` needs (embeddings, cached summaries) is a rebuildable cache derived purely from `research/papers.nix` via `pax-core`'s read functions — never a second source of truth, and `pax-ai` never writes `papers.nix` directly.

`lazy-pax` is expected to depend on `pax-ai` for AI-heavy features instead of implementing them internally, giving every AI feature (TUI or MCP client) one shared implementation.

## Status

Early: the LLM port with Anthropic and Ollama adapters, cached summaries, and a first MCP server (below). See [`CLAUDE.md`](CLAUDE.md) for the planned architecture and the [org's `pax-ai` docs](https://github.com/pax-project/.github/blob/main/docs/README.md#architecture-pax-ai) for the full design rationale.

## MCP server

`pax-mcp [ROOT]` serves the research library under `ROOT` (default: the current directory) over MCP on stdin/stdout. Results are JSON in the shapes `pax --json` prints (pax-core's `output` types), and every write goes through `pax-core`, so the library stays usable from the `pax` CLI and lazy-pax. Tools:

- `search`: by `query` or `author`, with `limit` (default 50 per provider) or `all`, and `year` (`2020`, `2015-2020`, `2015-`, `-2020`), `type`, and `language` filters, and a `label` (e.g. `R2-backward`) recorded on the search's log lines. `with_abstracts: true` adds each merged paper's `abstract` (the first among its records, in provider order; null if none has one) for title-and-abstract screening, at the cost of a response several times larger; without it, no `abstract` key is sent. Returns each provider's `retrieved`/`total` counts and how it applied each filter, the merge counts, and one list merged across providers: each paper with the providers and references it came from, flagged `in_library`, `excluded`, and `language_unknown`.
- `refs`, `cited_by`: the references or citing papers of one seed (`reference`) or several (`references`), from OpenAlex and Semantic Scholar. A seed is a provider reference, a library citation key or alias, or an excluded paper's alias; paging, filters, `label`, and `with_abstracts` are `search`'s. Each seed is reported on its own, with its per-provider counts and its `incomplete` lists and why (`withheld`, `failed`, `unresolved`, `truncated`), or its error if it's invalid, and `complete` says whether every seed's lists are whole. The results are one list merged across every seed and provider, each paper with the `seeds`, providers, and references it came from. When a references list is missing (`withheld`, `failed`, `unresolved`), use the paper's own reference list instead.
- `related`: the papers the providers recommend as related to one or more seeds (OpenAlex's related works, Semantic Scholar's recommendations): not citation edges, so they widen a snowballing start set. Input and output are `refs`'s. Recommendations have no total, so a related list is never exhaustive: `incomplete` names only missing lists (`failed`, `unresolved`), and a seed's `complete` means no list is missing.
- `show`: one paper, by exactly one of `reference` (a provider reference, or a library citation key or alias) or `doi`. A reference returns `pax show --json`'s shape: a declared paper (`declared`), an excluded paper's exclusion record for its alias (`excluded`, in `exclude`'s shape), or a provider's record with its `pdf_url` and `abstract` (`candidate`). A DOI is looked up at every provider that resolves DOIs, and their records come back merged and flagged like `search` results, in `records`.
- `add`: declare a search result in `research/papers.nix`. A paper already in the library is refused, and so is an excluded one: the refusal names its exclusion (stage, reason, key) and says to call `unexclude` first.
- `exclude`: record a paper in `research/excluded.nix` with a required `reason` and a `stage` (`screening` or `full-text`), and an optional `title`, `year`, protocol `criterion` (e.g. `X3`), screening `round` (e.g. `R1`), and `alias` (e.g. `P18`). Returns the recorded entry, year, criterion, round, and alias included. Re-excluding a paper replaces its entry, except that an alias not given again is kept; `alias: ""` clears it. Give `year` with `title` for a reference no provider resolves, so `stats` can match the exclusion to its logged record.
- `unexclude`: reverse a recorded screening decision, removing the paper's entry from `research/excluded.nix` by its exclusion key, alias, DOI, or provider reference. Returns the removed entry, in `exclude`'s shape, with the alias it freed.
- `edit`: add or remove a declared paper's tags, set its notes, and set its `alias` (`""` clears it); returns the updated paper.
- `fetch`: download a declared paper's PDF through Nix, recording its hash, and return the store path and hash. With `file` (a path, absolute or relative to the library root), declare a PDF the user already has instead, such as a paywalled paper downloaded through a university login, as `pax fetch --file` does: it's added to the Nix store and its hash recorded, and the result has the same shape. A file that doesn't match a hash already recorded is a `hash-mismatch` unless `force` is set, which replaces the hash; a missing or unreadable file is an error. A paper that can't be fetched comes back with a `status` (`no-source-url`, `fetch-failed`, `hash-mismatch`, `build-failed`, `file-needed` for a paper added from a local file this Nix store no longer has, which `file` provides again, `timed-out`) and a `reason`. Gives up after five minutes.
- `check`: verify every fetched paper's PDF still reproduces from its recorded hash, as `pax check` does: one report per paper (`not-fetched`, `reproducible`, `mismatch`, `error`, `missing` for a local file this Nix store no longer has) plus totals.
- `list`: declared papers, filtered by author, year, and tag, each with its `source_url` and `hash`.
- `export`: BibTeX text (`format: "bibtex"`) for the whole library, or for the given `keys` (citation keys or aliases) and/or a `tag`. Returns the text as `bibtex`, with the exported `citation_keys`, rather than writing a file.
- `stats`: the review's PRISMA 2020 flow-diagram counts (records identified per provider from database searches, citation searching, and related works, and per label from papers added by hand (`other_sources`), duplicates removed, records screened, exclusions per stage with full-text reasons and both stages by criterion, papers awaiting a screening decision, reports sought, reports not retrieved, papers included, and decisions made outside the search log), then `by_label`: per search-log label, the records identified, distinct, new distinct (not identified under an earlier label), and duplicates. Computed by pax-core from the library's files.

A paper can carry an alias, such as a review's study ID `P17`, set with `edit` on a declared paper or `exclude` on an excluded one. Every tool that takes a citation key (`show`, `edit`, `fetch`, `export`, and the `refs`/`cited_by`/`related` seeds) also takes an alias; `show`, the seed tools, and `unexclude` also take an excluded paper's alias, while `edit` and `fetch` refuse one. Citation keys and aliases share one namespace across `papers.nix` and `excluded.nix`, so an alias another paper holds is refused, naming that paper. Declared papers and exclusion records carry their `alias` (null when unset).

`search`, `refs`, `cited_by`, and `related` are recorded in `research/search-log.jsonl`, the same log the `pax` CLI writes, and each result's `log` says whether it was recorded and, if not, why.

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
