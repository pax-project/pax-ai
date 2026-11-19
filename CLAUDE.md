# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`pax-ai` is the shared AI substrate for the `pax` project: summaries, semantic search, RAG, and an MCP server, all built on top of [`pax-core`](https://github.com/pax-project/pax-core) — never modifying it. [`lazy-pax`](https://github.com/pax-project/lazy-pax) is expected to depend on `pax-ai` for these features instead of implementing them internally, so every AI feature has one shared implementation instead of being duplicated between the TUI and any MCP client.

Full architecture rationale (why this is a separate crate instead of a `pax-core` feature flag) lives in the org's [`docs/README.md`](https://github.com/pax-project/.github/blob/main/docs/README.md#architecture-pax-ai) — read it before making structural decisions here, don't re-derive it locally.

## Non-negotiables (carried over from `pax-core`)

- Writes only ever go through `pax-core` functions — never hand-write `papers.nix`.
- Any persisted state (embeddings, cached summaries) is a rebuildable cache derived purely from `papers.nix` via `pax-core`'s read functions — never a second source of truth.
- Any LLM/API call runs off the UI/event thread (the same `spawn_blocking` pattern `pax-core`/`lazy-pax` use for provider search and `nix` calls) — callers must never block on a network round-trip.
- Ideas that don't need an LLM/embedding step aren't `pax-ai`'s concern — they belong in `pax-core` or `lazy-pax` directly. The one exception is the MCP server (`src/mcp.rs`, binary `pax-mcp`, behind the default `mcp` feature): the org docs place it here so AI clients have one integration point, but its tools only wrap `pax-core` calls (or, for the embedding tools such as `index`, pax-ai's own library API) — domain logic still goes in `pax-core`.

## Planned architecture: ports and adapters for the LLM capability

The LLM/embedding capability is a port (trait), not a hardcoded client — cost, privacy, and self-hosting are per-deployment decisions (see "Cost/privacy" in the org docs), so nothing above the port should know which backend is in use.

- **Port:** a trait covering the operations `pax-ai` actually needs (e.g. summarize, embed) — shaped by real call sites, not spec'd upfront.
- **First adapter:** Anthropic (hosted). Sends abstracts/paper text to a third-party API — this is an explicit, deliberate choice, not a default; a local-model adapter (e.g. via Ollama) is the expected second adapter once the port's shape has settled against a real client.
- Adapter choice is a runtime/config concern, not a compile-time feature flag split, so a deployment can switch backends without a rebuild.
- `Embedder` (`src/port.rs`) has a `model_id` (part of every embedding-store key, e.g. `ollama:nomic-embed-text`) and a batch `embed(texts, EmbedKind)` whose future is `Send`; it returns one vector per input, in order, unnormalized, and a count mismatch is `PortError::UnexpectedResponse`. `EmbedKind` says whether text is a document or a query: the adapter adds any model-specific prefix (`OllamaAdapter` adds `search_document: `/`search_query: ` for `nomic-embed-text`), so callers never do.

None of this is implemented yet — this repo is currently a bootstrap scaffold (`Cargo.toml`, `flake.nix`/`.envrc`, empty `src/lib.rs`). Update this section as the port/adapter split actually lands in code.

## Environment

Nix + direnv for the dev shell (`flake.nix`, `.envrc`), same pattern as `pax-core`/`lazy-pax`: `.envrc` does `use flake`, so `direnv allow` sets up cargo/rustc/rustfmt/clippy/rust-analyzer plus the OpenSSL env vars (`PKG_CONFIG_PATH`, `OPENSSL_DIR`) needed for native TLS dependencies.

## Commands

```bash
cargo build   # builds the pax-ai library + the pax-mcp binary (mcp is a default feature)
cargo build --no-default-features  # library alone, without the MCP SDK (tokio with only rt + time, for the index's blocking work)
cargo test    # unit tests
cargo fmt     # format
cargo clippy  # lint
```

## Dependency on `pax-core`

crates.io dependency on the published `pax-core-papers` package, aliased back to `pax-core` (its lib name is still `pax_core`), `default-features = false` (the `cli` feature — and its `clap`/`dotenvy`/`tokio` deps — stays off), same as `lazy-pax`. Moving to a later `pax-core` release is a deliberate, explicit step, not automatic.

pax-core's enums and several of its structs (since 3.0 also `Local` and `PaperEdits`) are `#[non_exhaustive]`: matches on them need a wildcard arm, and its structs are built with `new`/`with_*`, not literals.

## MCP server notes

- Tools return pax-core's output shapes (`pax_core::output`, and `Stats`), the ones the `pax` CLI prints with `--json`, so the CLI and MCP agree field for field; the `mcp` feature turns on pax-core's `schemars` feature for their JSON Schemas. Build them with pax-core's `new`/`From` (e.g. `SearchResults::new`, `Excluded::from`), never with a local copy of a type pax-core exports. There are three wrappers: `ShowOutput`, pax-core's `Shown` flattened plus `records` (a `DoiRecords`) for `show { doi }`, which the CLI's `show` lacks; `ListOutput`, which flattens exactly one of `PaperList` or `ExclusionList` (`list { excluded: true }`), so either serializes as the CLI's `list --json` or `list --excluded --json`; and `ExportOutput`, which flattens exactly one of `Exported` or `ExportedPdfs` (`export { format: "pdfs" }`), as `export bibtex --json` or `export pdfs --json`, plus `warnings` (skipped when empty) for what the CLI prints on stderr: a `pdfs` export's `PdfExport::staged_by_git`, worded by `staged_by_git_warning`. Where the MCP words a result differently (e.g. `fetch`'s reasons), it builds pax-core's type and sets the field: pax-core's texts name the CLI's flags (`--file`, `sync --from`), so a missing local artifact's reason or error in `fetch`, `sync`, `check`, and `export` is `file_needed_reason`, and `sync`'s failed papers go through `failed_fetch` as `fetch`'s do.
- `search`, `refs`, `cited_by`, and `related` call pax-core's `*_logged` variants (`references_many_logged`/`cited_by_many_logged`/`related_many_logged` for the seed tools, even for one seed), so an agent's searches land in `research/search-log.jsonl` with the CLI's; the result's `log` says when nothing was logged (no library, or the write failed). Never call the unlogged `search_*`/`references`/`cited_by`/`related` from a tool.
- `add` calls `add_candidate_logged` only when given a `label` (an "other methods" source, e.g. `seeds`, which `stats` reports under `other_sources`), and returns its `log` in `Added`; without one it calls the unlogged `add_candidate`, as the CLI's `pax add` without `--label` does, and `Added.log` is absent. A labelled add can't be tested offline, so tests write the `manual` line with `search_log::manual_log_entry` to check `stats`.
- `show { doi }` calls the unlogged `search_by_doi` (pax-core has no logged variant): it's a lookup of a known paper, not a search for PRISMA's identified count.
- `fetch`, `sync`, `check`, and `export { format: "pdfs" }` shell out to `nix`, so they run on tokio's blocking pool (`spawn_blocking`); `fetch` also has an overall timeout. A paper that can't be fetched is a `FetchResult` status with a reason, not a tool error. `fetch { file, force }` calls `fetch_file` instead of `fetch_paper` (`force` without `file` is refused, as the CLI's `--force` requires `--file`); `file` resolves against the library root, and one that can't be opened is a tool error before `nix` runs, as is pax-core's `NotAFile`. A `HashMismatch` is a result whose reason points at `force`.
- Backup/restore (pax-core's `backup.rs`): `export { format: "pdfs", dir }` calls `export_pdfs` (`keys`/`tag` refused, `dir` refused for `bibtex`; without `dir`, `default_backup_dir`, i.e. `research/.pdfs/`, which pax-core gitignores), and `sync { from }` or `sync { from_backup: true }` (the default dir; checked to exist first, and refused with `from`) calls `sync_library_from`, plain `sync` `sync_library`; `dir`/`from` resolve against the library root, and a blank one is refused rather than taken as the default. Tests of the tracked-folder warning `git init` the fixture and skip without `git` (`git_available`). A `BackupMismatch` is a `hash-mismatch` result saying the file was skipped, never pointing at `force`; `NotADirectory` is a tool error.
- Tests that add files to the Nix store need real `nix` and return early without it (`nix_available`). They hold their library in a `StoreScratch`, which on drop removes it (and so the GC roots pax-core put in `research/.gcroots`) and then `nix-store --delete`s just the paths the test tracked; each test adds its own file content, so no two share a path. Never run a whole-store garbage collection. pax-core rewrites `papers.nix` in key order when it records a hash, so look results up by citation key after a `fetch`.
- A library name is a citation key or an alias (pax-core's "Aliases": one namespace across `papers.nix` and `excluded.nix`, looked up as a citation key first). Tools that take a citation key pass the name straight to pax-core, which resolves it; `show` on an excluded paper's alias returns `Shown::excluded`, and `edit`/`fetch` get `PaxError::ExcludedNotDeclared` for one. Where a tool needs the citation key itself (`edit` re-reading the paper after an alias change, `fetch` labelling its result), `citation_key` resolves it the same way; `export` matches `keys` with `is_named`. `name_error` words `NameTaken`/`ExcludedNotDeclared` for the agent.
- `index` builds the embedding index (`src/index/`, #52) with `PaperIndex::build` and returns its `BuildReport` (one `indexed`/`fallback`/`failed` status per declared paper, flattened next to `citation_key`; serde and, under `mcp`, `JsonSchema` derive on the library's own types, since pax-core has no shape for them). The embedder is runtime state, not a feature: `PaxServer` holds an `OllamaAdapter::new()` (local Ollama, `nomic-embed-text`), and the tool calls `index_with(embedder)`, which tests call with `index::tests::FakeEmbedder`. Inside `build`, `resolve_artifact_path` and `pdftotext` run through `blocking::blocking_with_timeout` (shared with `fetch`, so tokio is a regular dependency with just `rt` + `time`); a timeout, a resolve or extract error, or a chunk-embedding error is that paper's `fallback` (its title + venue vector is stored), a title + venue embedding error its `failed`. Stored entries are reused without the embedder or `nix` (a full-text entry by the recorded hash, so a paper with one never resolves its PDF), and the build ends by pruning everything the current `papers.nix` doesn't imply for that model, so another model's entries go too. Tests that must not run `nix` use papers without a hash; the full-text test also needs `pdftotext` (`index::extract::tests::pdftotext_available`, `pdf_with_lines`).
- In-library/excluded flags come from `LibraryIndex`, which matches by DOI or arXiv id; don't use the deprecated `known_dois`/`excluded_keys`.
- Tests use a fixture library in a temp dir and no network: tools that need a provider are tested on argument validation and on pax-core's output shapes built from hand-built `SearchPage`s. pax-core's `SeedResult` can't be built outside it, so the multi-seed tools are tested with seeds that resolve offline (a declared paper with no DOI or arXiv id, an unknown citation key). A hash with no `source_url` is a local artifact, which `check` looks up in the Nix store, so a test that must not run `nix` uses papers without a hash.
