//! MCP server exposing a research library's `pax-core` operations as tools,
//! so an agent calls pax directly instead of going through a shell and
//! parsing CLI output.
//!
//! Most of these tools need no LLM: they wrap `pax-core` calls, and every
//! write goes through `pax-core`, so an agent's decisions land in the same
//! `research/` files the `pax` CLI and lazy-pax use — searches included,
//! which are recorded in `research/search-log.jsonl`. Results are
//! `pax-core`'s own output shapes (`pax_core::output`, and `Stats`), the
//! ones the `pax` CLI prints with `--json`, so the two agree field for
//! field; only `show`'s DOI lookup, which the CLI lacks, wraps one. The
//! exception is `index`, which builds pax-ai's own embedding index
//! ([`PaperIndex`]) through an [`Embedder`] and returns its [`BuildReport`].
//! `semantic_search` ranks over that index and returns its
//! [`SemanticResults`]. `rank_candidates` scores provider candidates
//! against one indexed library paper, and returns pax-core's
//! [`MergedHit`]s with their scores; `rank_by` on `search`, `refs`,
//! `cited_by` and `related` adds the same scores to their results as
//! `ranking` ([`RankedOutput`]), which is otherwise left out.

use std::collections::HashMap;
use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use crate::adapter::ollama::OllamaAdapter;
use crate::blocking::blocking_with_timeout;
use crate::index::candidates::{CandidateScore, best_first, score_candidates};
use crate::index::{BuildReport, PaperIndex, PaperVectors, SemanticResults, Similar};
use crate::port::Embedder;
use pax_core::merge::MERGE_ORDER;
use pax_core::output::{
    Added, CheckResults, CheckState, CitationResults, DeclaredPaper, DoiRecords, Excluded,
    ExclusionList, Exported, ExportedPdfs, FetchResult, FetchState, Lists, MergedHit, PaperList,
    PdfCopyState, SearchResults, Shown, SyncResults,
};
use pax_core::{
    CandidateId, CandidateWork, Config, DocType, ExclusionFilter, ExclusionOptions, ExclusionStage,
    FetchOutcome, Filters, Library, LibraryIndex, Limit, ListFilter, Logged, MergedWork,
    NameHolder, Paper, PaperEdits, PaxError, ProviderError, ProviderId, ResolvedArtifact,
    SearchOptions, SearchPage, SeedResults, Stats, SyncReport, YearRange,
};
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{Implementation, ServerCapabilities, ServerConfig};
use rmcp::{Json, ServerHandler, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// How many results to retrieve per provider, and the filters to apply —
/// the `pax` CLI's `--limit`/`--all`/`--year`/`--type`/`--lang`.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct PagingParams {
    /// Retrieve at most this many results per provider (default 50).
    pub limit: Option<usize>,
    /// Retrieve every result each provider serves instead of `limit` (can be
    /// slow).
    #[serde(default)]
    pub all: bool,
    /// Publication year: `2020`, or an inclusive range `2015-2020`, `2015-`
    /// (from 2015 on) or `-2020` (up to 2020).
    pub year: Option<String>,
    /// Document type: journal-article, conference-paper, preprint, review,
    /// book, book-chapter, thesis, or dataset.
    #[serde(rename = "type")]
    pub doc_type: Option<String>,
    /// Language, as an ISO 639-1 code (e.g. `en`).
    pub language: Option<String>,
    /// A label recorded on every search-log line this call writes, e.g.
    /// `R2-backward`, so `stats` can count each round apart (`by_label`).
    /// It doesn't change what is searched; a blank label is refused.
    pub label: Option<String>,
    /// Include each merged paper's `abstract`, for title-and-abstract
    /// screening without a `show` per paper. Makes the response several
    /// times larger.
    #[serde(default)]
    pub with_abstracts: bool,
}

impl PagingParams {
    /// The `pax-core` search options these parameters ask for; parsing is
    /// `pax-core`'s, so the accepted values match the CLI's.
    fn options(&self) -> Result<SearchOptions, String> {
        let defaults = SearchOptions::default();
        let limit = match (self.all, self.limit) {
            (true, Some(_)) => return Err("give `limit` or `all`, not both".to_string()),
            (true, None) => Limit::All,
            (false, Some(0)) => return Err("`limit` must be at least 1".to_string()),
            (false, Some(n)) => Limit::Count(n),
            (false, None) => defaults.limit,
        };
        let mut filters = Filters::default();
        if let Some(year) = &self.year {
            let years: YearRange = year.trim().parse().map_err(|e| format!("{e}"))?;
            filters = filters.with_years(years);
        }
        if let Some(doc_type) = &self.doc_type {
            let doc_type: DocType = doc_type.trim().parse().map_err(|e| format!("{e}"))?;
            filters = filters.with_doc_type(doc_type);
        }
        if let Some(language) = &self.language {
            let language = language.trim();
            if language.is_empty() {
                return Err("`language` must be an ISO 639-1 code, e.g. `en`".to_string());
            }
            filters = filters.with_language(language);
        }
        let options = defaults
            .with_limit(limit)
            .with_filters(filters)
            .with_abstracts(self.with_abstracts);
        match &self.label {
            Some(label) if label.trim().is_empty() => {
                Err("`label` must not be blank, e.g. `R2-backward`".to_string())
            }
            Some(label) => Ok(options.with_label(label.trim())),
            None => Ok(options),
        }
    }
}

/// `search`'s `providers`, parsed as the CLI's `--provider` is; none means
/// pax-core's default set.
fn providers(names: &[String]) -> Result<Vec<ProviderId>, String> {
    names
        .iter()
        .map(|name| {
            name.trim().parse().map_err(|e| {
                format!("{e}; `providers` takes openalex, crossref, semanticscholar, arxiv or dblp")
            })
        })
        .collect()
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct SearchParams {
    /// Free-text query. Give exactly one of `query` or `author`.
    pub query: Option<String>,
    /// Author name, for an author-scoped search instead of free text.
    pub author: Option<String>,
    /// Ask only these providers: `openalex`, `crossref`, `semanticscholar`,
    /// `arxiv` or `dblp`. Without it, every provider but dblp is asked;
    /// dblp (computer science only) is asked only when named, alone or
    /// beside others.
    #[serde(default)]
    pub providers: Vec<String>,
    /// Rank the returned results by how close in content each is to this
    /// library paper (a citation key or alias, which must be in the
    /// embedding index): see `ranking` in the result. Turns
    /// `with_abstracts` on.
    pub rank_by: Option<String>,
    #[serde(flatten)]
    pub paging: PagingParams,
}

/// What `search`, `refs`, `cited_by` and `related` return: pax-core's
/// shape ([`SearchResults`] or [`CitationResults`]) flattened, plus
/// `ranking` when asked for with `rank_by`; without it, exactly pax-core's
/// shape, as the CLI's `--json` prints it.
#[derive(Debug, Serialize, JsonSchema)]
pub struct RankedOutput<T> {
    #[serde(flatten)]
    pub output: T,
    /// With `rank_by`: every returned paper, by its `id`, best first, with
    /// its score against the `rank_by` paper (higher is closer, at most 1;
    /// not comparable with `similar`'s scores) and whether its abstract was
    /// embedded (otherwise only its title was). Absent without `rank_by`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ranking: Option<Vec<RankingEntry>>,
}

/// The results read as pax-core's own shape.
impl<T> std::ops::Deref for RankedOutput<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.output
    }
}

/// One paper's place in a `rank_by` ranking.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct RankingEntry {
    /// The paper's `id` in the results.
    pub id: String,
    pub score: f32,
    pub had_abstract: bool,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct CitationParams {
    /// One seed paper: a provider reference (e.g. `openalex:W2072794470`,
    /// from any provider), the citation key or alias of a paper in the
    /// library, or an excluded paper's alias. Give exactly one of
    /// `reference` or `references`.
    pub reference: Option<String>,
    /// Several seed papers, each as `reference`, for one snowballing round:
    /// each seed is reported on its own, and the results are merged across
    /// every seed.
    pub references: Option<Vec<String>>,
    /// Rank the returned papers by how close in content each is to this
    /// library paper (a citation key or alias, which must be in the
    /// embedding index): see `ranking` in the result. Turns
    /// `with_abstracts` on.
    pub rank_by: Option<String>,
    #[serde(flatten)]
    pub paging: PagingParams,
}

/// Which seed lookup a citation tool runs.
#[derive(Debug, Clone, Copy)]
enum CitationTool {
    Refs,
    CitedBy,
    Related,
}

impl CitationParams {
    /// The seeds to look up, in the order given.
    fn seeds(self) -> Result<Vec<String>, String> {
        match (self.reference, self.references) {
            (Some(reference), None) => Ok(vec![reference]),
            (None, Some(references)) if !references.is_empty() => Ok(references),
            (None, Some(_)) => Err("`references` must name at least one seed".to_string()),
            _ => Err("give exactly one of `reference` or `references`".to_string()),
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AddParams {
    /// A provider reference from `search`, e.g. `openalex:W2072794470`.
    pub reference: String,
    /// Where the paper came from outside the searches, e.g. `seeds` for
    /// papers the reviewer already knew. The add is then recorded in the
    /// search log under this label, so `stats` counts the paper as
    /// identified from that source (`other_sources`, and `by_label`) rather
    /// than as decided outside the log. It doesn't change what is added; a
    /// blank label is refused.
    pub label: Option<String>,
}

impl AddParams {
    /// The trimmed label, if one was given; a blank one is refused.
    fn label(&self) -> Result<Option<&str>, String> {
        match self.label.as_deref().map(str::trim) {
            Some("") => Err("`label` must not be blank, e.g. `seeds`".to_string()),
            label => Ok(label),
        }
    }
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct ShowParams {
    /// A provider reference from `search` (e.g. `openalex:W2072794470`), the
    /// citation key or alias of a paper in the library (e.g. `turing1936`
    /// or `P17`), or an excluded paper's alias. Give exactly one of
    /// `reference` or `doi`.
    pub reference: Option<String>,
    /// A DOI (bare or `https://doi.org/...`), looked up at every provider
    /// that resolves DOIs, since one DOI can match several providers'
    /// records.
    pub doi: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct FetchParams {
    /// The citation key or alias of a paper in the library, e.g.
    /// `turing1936` or `P17`.
    pub reference: String,
    /// A PDF the user already has (e.g. a paywalled paper downloaded through
    /// a university login), added as the paper's artifact instead of
    /// downloading it: an absolute path, or one relative to the library
    /// root.
    pub file: Option<String>,
    /// With `file`, replace an already-recorded hash that the file doesn't
    /// match (otherwise that's a `hash-mismatch`, and nothing is written).
    #[serde(default)]
    pub force: bool,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct SyncParams {
    /// A backup directory `export` wrote with `format: "pdfs"` (or any
    /// holding `<citation_key>.pdf` files) to restore PDFs added from local
    /// files from: an absolute path, or one relative to the library root.
    /// Not with `from_backup`.
    pub from: Option<String>,
    /// Restore from the default backup directory, `research/.pdfs/`, which
    /// `export` with `format: "pdfs"` and no `dir` writes. Not with `from`.
    #[serde(default)]
    pub from_backup: bool,
}

/// When in screening a paper was excluded; PRISMA reports the two apart.
#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum Stage {
    /// Excluded from its title and abstract.
    Screening,
    /// Excluded after reading the full text.
    FullText,
}

impl From<Stage> for ExclusionStage {
    fn from(stage: Stage) -> Self {
        match stage {
            Stage::Screening => ExclusionStage::Screening,
            Stage::FullText => ExclusionStage::FullText,
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExcludeParams {
    /// A DOI (bare or `https://doi.org/...`) or a provider reference from
    /// `search`, e.g. `openalex:W2072794470`.
    pub reference: String,
    /// Why the paper is excluded, e.g. "not empirical". Required.
    pub reason: String,
    pub stage: Stage,
    /// Title to record; overrides the provider's, and is needed when the
    /// reference can't be resolved. With a bare DOI, no provider is asked.
    pub title: Option<String>,
    /// Publication year to record; overrides the provider's. Give it with
    /// `title` for a reference that can't be resolved: without a year,
    /// `stats` can't match an exclusion keyed by a provider reference to
    /// its logged record, which then stays awaiting screening.
    pub year: Option<i32>,
    /// The protocol's exclusion criterion, e.g. `X3` (free text), so
    /// `stats` can group exclusions by criterion.
    pub criterion: Option<String>,
    /// The screening round, e.g. `R1` (free text); meant to match the
    /// search log's labels.
    pub round: Option<String>,
    /// The paper's alias, e.g. the review's study ID `P18`: letters,
    /// digits, `_`, `.` and `-`, held by no other paper as an alias or
    /// citation key. Omitted, an earlier exclusion's alias is kept; `""`
    /// clears it.
    pub alias: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct UnexcludeParams {
    /// The exclusion's key (as `exclude` returns it, or as `add`'s refusal
    /// names it), its alias, a DOI, or a provider reference from `search`.
    pub reference: String,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct EditParams {
    /// The citation key or alias of a paper in the library.
    pub reference: String,
    /// Tags to add; one the paper already has isn't repeated.
    #[serde(default)]
    pub add_tags: Vec<String>,
    /// Tags to remove, applied after `add_tags` (so a tag in both ends up
    /// removed).
    #[serde(default)]
    pub remove_tags: Vec<String>,
    /// Replaces the paper's notes.
    pub notes: Option<String>,
    /// Sets the paper's alias, e.g. the review's study ID `P17`: letters,
    /// digits, `_`, `.` and `-`, held by no other paper as an alias or
    /// citation key. `""` clears it.
    pub alias: Option<String>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct ListParams {
    /// List the excluded papers in `research/excluded.nix` instead of the
    /// declared ones (the `pax` CLI's `list --excluded`).
    #[serde(default)]
    pub excluded: bool,
    /// Case-insensitive substring of an author's name. Declared papers only.
    pub author: Option<String>,
    /// Publication year.
    pub year: Option<i32>,
    /// Exact tag. Declared papers only.
    pub tag: Option<String>,
    /// Exact exclusion criterion, e.g. `X3`. With `excluded: true` only.
    pub criterion: Option<String>,
    /// Exact screening round, e.g. `R1`. With `excluded: true` only.
    pub round: Option<String>,
}

/// What `list` returns: pax-core's [`PaperList`] (`papers`) for declared
/// papers, or its [`ExclusionList`] (`exclusions`) with `excluded: true`,
/// the shapes `pax list --json` and `pax list --excluded --json` print.
#[derive(Debug, Serialize, JsonSchema)]
pub struct ListOutput {
    #[serde(flatten)]
    pub declared: Option<PaperList>,
    #[serde(flatten)]
    pub excluded: Option<ExclusionList>,
}

/// What `export` produces.
#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ExportFormat {
    /// BibTeX text, returned in the result.
    Bibtex,
    /// A backup of the PDFs added from local files, written to `dir`.
    Pdfs,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExportParams {
    pub format: ExportFormat,
    /// Export only these papers, by citation key or alias; every one must be
    /// in the library.
    /// Without `keys` or `tag`, the whole library is exported. `bibtex`
    /// only.
    pub keys: Option<Vec<String>>,
    /// Export only papers with this exact tag (combined with `keys` by AND).
    /// `bibtex` only.
    pub tag: Option<String>,
    /// The backup directory, created if needed: an absolute path, or one
    /// relative to the library root. `pdfs` only; without it, the default
    /// `research/.pdfs/`, which git ignores.
    pub dir: Option<String>,
}

/// What `export` returns: pax-core's [`Exported`] (`citation_keys`,
/// `bibtex`) for `bibtex`, or its [`ExportedPdfs`] (`directory`,
/// `manifest`, `papers`, and counts) for `pdfs`, the shapes `pax export
/// bibtex --json` and `pax export pdfs --json` print. `warnings` is this
/// server's own addition: what the CLI prints on stderr, such as a `pdfs`
/// backup directory git would commit with the library.
#[derive(Debug, Serialize, JsonSchema)]
pub struct ExportOutput {
    #[serde(flatten)]
    pub text: Option<Exported>,
    #[serde(flatten)]
    pub pdfs: Option<ExportedPdfs>,
    /// What the agent should tell the user about this export; absent when
    /// there's nothing.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SemanticSearchParams {
    /// What to look for, in plain words: a theme, a claim, a method.
    pub query: String,
    /// Return at most this many papers (default 10).
    pub limit: Option<usize>,
}

/// What `show` found. For a `reference`, pax-core's [`Shown`]: exactly one of
/// `declared` (a paper in the library), `excluded` (an excluded paper named
/// by its alias) or `candidate` (a provider reference) is set. For a `doi`,
/// only `records` is: pax-core's `show` doesn't look DOIs up, so `records`
/// is this server's own addition.
#[derive(Debug, Serialize, JsonSchema)]
pub struct ShowOutput {
    #[serde(flatten)]
    pub shown: Option<Shown>,
    /// Every provider's record of the DOI looked up; null for a `reference`.
    pub records: Option<DoiRecords>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SimilarParams {
    /// The citation key or alias of a paper in the library, e.g.
    /// `turing1936` or `P17`.
    pub name: String,
    /// How many papers to return, best first (default 10).
    pub limit: Option<usize>,
}

/// How many papers `similar` returns without a `limit`.
const SIMILAR_LIMIT: usize = 10;

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct RankCandidatesParams {
    /// The library paper to rank against: a citation key or alias, e.g.
    /// `P17`. It must be in the embedding index (`index` adds it).
    pub seed: String,
    /// Candidates as `search`, `refs`, `cited_by` and `related` return
    /// them: provider references, e.g. `openalex:W2072794470`. Give at least
    /// one of `references` or `dois`.
    #[serde(default)]
    pub references: Vec<String>,
    /// Candidates by DOI (bare or `https://doi.org/...`).
    #[serde(default)]
    pub dois: Vec<String>,
}

/// The most candidates `rank_candidates` looks up in one call.
const RANK_LIMIT: usize = 100;

/// How many candidate lookups `rank_candidates` runs at once.
const LOOKUP_CONCURRENCY: usize = 4;

/// What every ranking says about its scores.
const RANKING_NOTE: &str = "scores rank candidates against this seed only; not comparable \
                            with `similar`";

/// One candidate `rank_candidates` ranked: pax-core's [`MergedHit`], as
/// `search` returns a paper, plus its score against the seed.
#[derive(Debug, Serialize, JsonSchema)]
pub struct RankedCandidate {
    #[serde(flatten)]
    pub hit: MergedHit,
    /// The candidate's best cosine among the seed's indexed vectors; higher
    /// is closer, at most 1.
    pub score: f32,
    /// Whether the candidate's abstract was embedded with its title; without
    /// one, only the title was, and the score says less.
    pub had_abstract: bool,
}

/// A candidate `rank_candidates` couldn't look up.
#[derive(Debug, Serialize, JsonSchema)]
pub struct FailedLookup {
    /// The reference or DOI, as given (a DOI normalized).
    pub reference: String,
    pub reason: String,
}

/// What `rank_candidates` returns.
#[derive(Debug, Serialize, JsonSchema)]
pub struct RankedCandidates {
    /// The seed's citation key.
    pub seed: String,
    /// Only the seed's title and venue are indexed (no full text), so the
    /// scores say less.
    pub seed_fallback: bool,
    /// Best first.
    pub ranked: Vec<RankedCandidate>,
    /// Lookups that failed; the other candidates are still ranked.
    pub failed: Vec<FailedLookup>,
    pub note: String,
}

/// One candidate `rank_candidates` looks up.
#[derive(Debug, Clone)]
enum Lookup {
    Reference(String, CandidateId),
    Doi(String),
}

impl Lookup {
    /// The candidate as given, for `failed`.
    fn reference(&self) -> &str {
        match self {
            Lookup::Reference(reference, _) | Lookup::Doi(reference) => reference,
        }
    }

    /// The provider record: a reference resolved directly, a DOI looked up
    /// at every provider that resolves DOIs (unlogged, as `show` does).
    async fn run(self, config: Config) -> Result<CandidateWork, String> {
        match self {
            Lookup::Reference(_, id) => pax_core::resolve_candidate(&id, &config)
                .await
                .map_err(|e| e.to_string()),
            Lookup::Doi(doi) => doi_record(&pax_core::search_by_doi(&doi, &config).await),
        }
    }
}

/// Serves one research library: `root` is the directory containing
/// `research/`, as for the `pax` CLI.
#[derive(Clone)]
pub struct PaxServer {
    root: PathBuf,
    config: Config,
    /// What `index` embeds with: a local Ollama server's `nomic-embed-text`.
    /// `similar` reads the entries of its `model_id`, without calling it.
    embedder: Arc<OllamaAdapter>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl PaxServer {
    pub fn new(root: PathBuf, config: Config) -> Self {
        PaxServer {
            root,
            config,
            embedder: Arc::new(OllamaAdapter::new()),
            tool_router: Self::tool_router(),
        }
    }

    /// Search the providers (OpenAlex, Crossref, Semantic Scholar, arXiv, or
    /// those named in `providers`, which can add dblp) for
    /// papers, by free text or by author, optionally filtered by year, type,
    /// and language. Reports each provider's retrieved/total counts, then one
    /// list merged across providers, each paper flagged if it's already in
    /// the library or excluded. Any result `id` works with `show` and `add`.
    /// `with_abstracts` adds each paper's `abstract`, making the response
    /// several times larger. The search is recorded in the library's search
    /// log, under `label` when given. `rank_by` (a library paper in the
    /// embedding index) also ranks the returned page by closeness in
    /// content to that paper, in `ranking`, with abstracts on; to rank
    /// more, fetch more pages.
    #[tool]
    async fn search(
        &self,
        Parameters(params): Parameters<SearchParams>,
    ) -> Result<Json<RankedOutput<SearchResults>>, String> {
        self.search_with(self.embedder.as_ref(), params).await
    }

    /// The papers one or more seed papers cite (backward snowballing), from
    /// OpenAlex and Semantic Scholar (the providers with real citation
    /// data). Reports each seed on its own: per-provider counts and which
    /// lists are `incomplete`, with why; then one list merged across every
    /// seed and provider, each paper naming the `seeds` and `providers` that
    /// returned it and flagged against the library. `complete` says whether
    /// every seed's lists are whole. When a seed's list is `withheld`,
    /// `failed`, or `unresolved` for a provider, that list is missing: take
    /// the paper's references from its own reference list (its
    /// bibliography) instead. `truncated` only means `limit` stopped
    /// retrieval; ask again with a higher `limit` or `all`.
    /// `with_abstracts` adds each paper's `abstract`, making the response
    /// several times larger. Recorded in the library's search log, under
    /// `label` when given. `rank_by` ranks the returned papers as in
    /// `search`.
    #[tool]
    async fn refs(
        &self,
        Parameters(params): Parameters<CitationParams>,
    ) -> Result<Json<RankedOutput<CitationResults>>, String> {
        self.citations_with(self.embedder.as_ref(), params, CitationTool::Refs)
            .await
    }

    /// The papers citing one or more seed papers (forward snowballing);
    /// otherwise as `refs`, `with_abstracts` and `rank_by` included
    /// (responses several times larger). A paper's own text can't list what
    /// cites it, so a missing list here has no fallback but the other
    /// provider's.
    #[tool]
    async fn cited_by(
        &self,
        Parameters(params): Parameters<CitationParams>,
    ) -> Result<Json<RankedOutput<CitationResults>>, String> {
        self.citations_with(self.embedder.as_ref(), params, CitationTool::CitedBy)
            .await
    }

    /// The papers the providers themselves recommend as related to one or
    /// more seed papers: OpenAlex's related works and Semantic Scholar's
    /// recommendations. They aren't citation edges, so they reach beyond the
    /// seeds' citation neighbourhood and make a snowballing start set more
    /// varied. Input and output are as `refs`, `with_abstracts`, `label`
    /// and `rank_by` included. Recommendations have no total, so a related
    /// list is never exhaustive: `incomplete` names only the missing lists
    /// (`failed`, `unresolved`), and `complete` means no list is missing,
    /// not that every related paper was returned. Recorded in the library's
    /// search log, and counted in `stats` as `identified.related_works`.
    #[tool]
    async fn related(
        &self,
        Parameters(params): Parameters<CitationParams>,
    ) -> Result<Json<RankedOutput<CitationResults>>, String> {
        self.citations_with(self.embedder.as_ref(), params, CitationTool::Related)
            .await
    }

    /// Show one paper: by a provider reference from `search` (resolved live),
    /// the citation key or alias of a paper already in the library, or a
    /// DOI. An excluded paper's alias returns its exclusion record
    /// (`excluded`), read from the library like a citation key. A DOI is
    /// looked up at every provider that resolves DOIs, and their records come
    /// back merged as in `search`, flagged if in the library or excluded.
    /// DOI lookups aren't recorded in the search log.
    #[tool]
    async fn show(
        &self,
        Parameters(params): Parameters<ShowParams>,
    ) -> Result<Json<ShowOutput>, String> {
        let reference = match (params.reference, params.doi) {
            (Some(reference), None) => reference,
            (None, Some(doi)) => {
                let doi = doi_param(&doi)?;
                let results = pax_core::search_by_doi(&doi, &self.config).await;
                let index = LibraryIndex::load(&self.root);
                return Ok(Json(ShowOutput {
                    shown: None,
                    records: Some(DoiRecords::new(&doi, &results, &index)),
                }));
            }
            _ => return Err("give exactly one of `reference` or `doi`".to_string()),
        };
        let shown = pax_core::show_reference(&reference, &self.root, &self.config)
            .await
            .map_err(|e| e.to_string())?;
        let index = LibraryIndex::load(&self.root);
        Ok(Json(ShowOutput {
            shown: Some(Shown::new(&shown, &index)),
            records: None,
        }))
    }

    /// Declare a search result in the library (metadata only; the PDF is
    /// fetched separately). Returns the new paper's citation key. A paper
    /// already in the library (same DOI or arXiv id) is refused, and so is
    /// one excluded from the review: the refusal names its exclusion, which
    /// only `unexclude` reverses, so ask the reviewer before calling it.
    /// With a `label` naming where the paper came from (e.g. `seeds`), the
    /// add is recorded in the search log as that "other methods" source,
    /// which `stats` counts; the result's `log` then says whether it was
    /// recorded.
    #[tool]
    async fn add(&self, Parameters(params): Parameters<AddParams>) -> Result<Json<Added>, String> {
        let label = params.label()?;
        let id: CandidateId = params.reference.parse().map_err(|e| format!("{e}"))?;
        let Some(label) = label else {
            let key = pax_core::add_candidate(&id, &self.root, &self.config)
                .await
                .map_err(add_error)?;
            return Ok(Json(Added::new(&key, None)));
        };
        let Logged { results, log, .. } =
            pax_core::add_candidate_logged(&id, label, &self.root, &self.config)
                .await
                .map_err(add_error)?;
        Ok(Json(Added::new(&results, Some(&log))))
    }

    /// Exclude a paper from the review: record it in research/excluded.nix
    /// with a reason and a stage (`screening` or `full-text`), and optionally
    /// the protocol's `criterion`, the screening `round` and the paper's
    /// `year`, replacing any earlier exclusion of it (so a criterion or round
    /// not given again is cleared). An `alias` (e.g. the review's study ID)
    /// names the excluded paper in `show`, `refs`, `cited_by`, `related` and
    /// `unexclude`; unlike the criterion and round, one not given again is
    /// kept, and `""` clears it. Excluded papers are flagged in `search`,
    /// `refs`, and `cited_by` results. A paper declared in the library is
    /// refused; it must be removed first. `add` refuses an excluded paper
    /// until `unexclude` reverses the decision.
    #[tool]
    async fn exclude(
        &self,
        Parameters(params): Parameters<ExcludeParams>,
    ) -> Result<Json<Excluded>, String> {
        let options = ExclusionOptions::new(params.reason, params.stage.into())
            .with_title(params.title)
            .with_year(params.year)
            .with_criterion(params.criterion)
            .with_round(params.round)
            .with_alias(params.alias);
        let exclusion =
            pax_core::exclude_paper_with(&params.reference, &options, &self.root, &self.config)
                .await
                .map_err(|e| match e {
                    PaxError::ExclusionTitleUnresolved(..) => format!("{e} with `title`"),
                    other => name_error(other),
                })?;
        Ok(Json(Excluded::from(&exclusion)))
    }

    /// Reverse a recorded screening decision: remove the paper's exclusion
    /// from research/excluded.nix, so `add` accepts it again. Returns the
    /// removed entry, with the alias it freed: give it back with `edit` once
    /// the paper is added. Excluding was the reviewer's decision, so only undo it
    /// when the reviewer asked to, not to get past `add`'s refusal. A paper
    /// with no matching exclusion is refused.
    #[tool]
    async fn unexclude(
        &self,
        Parameters(params): Parameters<UnexcludeParams>,
    ) -> Result<Json<Excluded>, String> {
        let exclusion = pax_core::unexclude_paper(&params.reference, &self.root, &self.config)
            .await
            .map_err(|e| e.to_string())?;
        Ok(Json(Excluded::from(&exclusion)))
    }

    /// Edit a declared paper's tags, notes and alias (e.g. tag it with a
    /// review's screening decision, or give it the review's study ID `P17`,
    /// which every tool then accepts in place of its citation key). An alias
    /// another paper already holds, as an alias or a citation key, is
    /// refused. Returns the paper as it is after the edit.
    #[tool]
    async fn edit(
        &self,
        Parameters(params): Parameters<EditParams>,
    ) -> Result<Json<DeclaredPaper>, String> {
        let path = self.papers_path()?;
        let edits = paper_edits(
            params.add_tags,
            params.remove_tags,
            params.notes,
            params.alias,
        )?;
        // Resolved before the edit, which may change the alias it was named by.
        let key = Library::load(&path)
            .ok()
            .and_then(|library| citation_key(&library, &params.reference))
            .unwrap_or_else(|| params.reference.clone());
        pax_core::edit_paper(&params.reference, &self.root, &edits).map_err(name_error)?;
        let library = Library::load(&path).map_err(|e| e.to_string())?;
        let paper = library
            .find(&key)
            .ok_or_else(|| format!("{:?} disappeared after the edit", params.reference))?;
        Ok(Json(DeclaredPaper::from(paper)))
    }

    /// The review's PRISMA 2020 flow-diagram counts: records identified per
    /// provider (database searches, citation searching and related works)
    /// and per label of papers added by hand (`other_sources`), duplicates
    /// removed, records screened, exclusions at each stage (full-text ones
    /// by reason, both stages by criterion), papers still awaiting a
    /// screening decision, reports sought (papers that passed screening),
    /// reports not retrieved, and papers included; then the records
    /// identified under each search-log `label` (`by_label`). Mid-round,
    /// report `awaiting_screening` as still to screen, not as passed.
    /// Computed from the library's files; nothing is written.
    #[tool]
    async fn stats(&self) -> Result<Json<Stats>, String> {
        self.papers_path()?;
        let stats = pax_core::stats(&self.root).map_err(|e| e.to_string())?;
        Ok(Json(stats))
    }

    /// List the papers declared in the library (`papers`), optionally
    /// filtered by author, year, and tag; or, with `excluded: true`, the
    /// papers excluded from the review in research/excluded.nix
    /// (`exclusions`: key, alias, title, year, reason, stage, criterion,
    /// round), optionally filtered by year, criterion, and round. Filters
    /// combine with AND; tag, criterion, and round match exactly.
    #[tool]
    async fn list(
        &self,
        Parameters(params): Parameters<ListParams>,
    ) -> Result<Json<ListOutput>, String> {
        let papers_path = self.papers_path()?;
        if params.excluded {
            if params.author.is_some() || params.tag.is_some() {
                return Err("`author` and `tag` filter declared papers: drop them, \
                            or drop `excluded`"
                    .to_string());
            }
            let exclusions = pax_core::Exclusions::load(&pax_core::nix::excluded_path(&self.root))
                .map_err(|e| e.to_string())?;
            let filter = ExclusionFilter::new()
                .with_year(params.year)
                .with_criterion(params.criterion)
                .with_round(params.round);
            let matching = pax_core::filter_exclusions(exclusions.entries(), &filter);
            return Ok(Json(ListOutput {
                declared: None,
                excluded: Some(ExclusionList::new(&matching)),
            }));
        }
        if params.criterion.is_some() || params.round.is_some() {
            return Err("`criterion` and `round` filter exclusions: \
                        pass `excluded: true` to list them"
                .to_string());
        }
        let library = pax_core::Library::load(&papers_path).map_err(|e| e.to_string())?;
        let filter = ListFilter {
            author: params.author,
            year: params.year,
            tag: params.tag,
        };
        let papers = pax_core::filter_papers(library.papers(), &filter);
        Ok(Json(ListOutput {
            declared: Some(PaperList::new(&papers)),
            excluded: None,
        }))
    }

    /// With `format: "bibtex"`, export declared papers as BibTeX text: the
    /// whole library, or only the papers named in `keys` (citation keys or
    /// aliases) and/or the papers with a `tag`. Returns the text (`bibtex`)
    /// and the exported papers' `citation_keys` instead of writing a file.
    ///
    /// With `format: "pdfs"`, back up the PDFs added from local files
    /// (`fetch` with `file`), which exist only in this machine's Nix store
    /// and can't be downloaded again: each is copied to
    /// `<dir>/<citation_key>.pdf`, and `<dir>/manifest.json` maps each
    /// exported citation key to its hash. Papers with a source URL aren't
    /// included, since Nix downloads them again. Without `dir`, the backup
    /// goes to `research/.pdfs/`, which git ignores, so the PDFs (often
    /// paywalled) are never committed with the library. Nothing is uploaded:
    /// that folder is on the same disk as the Nix store, so a real backup
    /// needs the user to copy it off this machine. A `dir` inside
    /// `research/` that git doesn't ignore is still written, but would be
    /// committed with the library, and the result's `warnings` says so:
    /// pass that on to the user. `sync` with `from_backup` (the default
    /// folder) or `from` (the directory) restores the backup, here or on
    /// another machine with the same library. A paper whose file isn't in
    /// this Nix store is reported as `missing` and the rest are still
    /// exported.
    #[tool]
    async fn export(
        &self,
        Parameters(params): Parameters<ExportParams>,
    ) -> Result<Json<ExportOutput>, String> {
        let papers_path = self.papers_path()?;
        if let ExportFormat::Pdfs = params.format {
            if params.keys.is_some() || params.tag.is_some() {
                return Err(
                    "`keys` and `tag` select papers for `bibtex`; `pdfs` backs up \
                            every PDF added from a local file"
                        .to_string(),
                );
            }
            let dir = match params.dir.as_deref().map(str::trim) {
                Some("") => {
                    return Err(
                        "`dir` is blank: give the backup directory, or leave it out for \
                         the default research/.pdfs/"
                            .to_string(),
                    );
                }
                Some(dir) => self.root.join(dir),
                None => pax_core::default_backup_dir(&self.root),
            };
            let root = self.root.clone();
            let (mut exported, staged_by_git) = tokio::task::spawn_blocking(move || {
                pax_core::export_pdfs(&root, &dir)
                    .map(|export| (ExportedPdfs::new(&export), export.staged_by_git))
            })
            .await
            .map_err(|e| format!("export didn't finish: {e}"))?
            .map_err(|e| e.to_string())?;
            let warnings = if staged_by_git {
                vec![staged_by_git_warning(&exported.directory)]
            } else {
                Vec::new()
            };
            // pax-core's error names the CLI's commands.
            for paper in &mut exported.papers {
                if paper.status == PdfCopyState::Missing {
                    paper.error = Some(file_needed_reason());
                }
            }
            return Ok(Json(ExportOutput {
                text: None,
                pdfs: Some(exported),
                warnings,
            }));
        }
        if params.dir.is_some() {
            return Err(
                "`dir` is where `pdfs` writes its backup; `bibtex` returns text".to_string(),
            );
        }
        let library = Library::load(&papers_path).map_err(|e| e.to_string())?;
        let papers = select_papers(
            library.papers(),
            params.keys.as_deref(),
            params.tag.as_deref(),
        )?;
        Ok(Json(ExportOutput {
            text: Some(Exported::new(&papers)),
            pdfs: None,
            warnings: Vec::new(),
        }))
    }

    /// Verify every fetched paper's PDF still reproduces from its recorded
    /// hash (one `nix store prefetch-file` per paper, so it needs the
    /// network), without writing anything. Papers not fetched yet are
    /// reported as such, and a paper added from a local file that isn't in
    /// this machine's Nix store as `missing` (`fetch` with `file` provides
    /// it again, or `sync` with `from` restores it from a backup); one
    /// paper's failure doesn't stop the rest.
    #[tool]
    async fn check(&self) -> Result<Json<CheckResults>, String> {
        self.papers_path()?;
        let root = self.root.clone();
        let mut checked = tokio::task::spawn_blocking(move || {
            pax_core::check_artifacts(&root).map(|checks| CheckResults::new(&checks))
        })
        .await
        .map_err(|e| format!("check didn't finish: {e}"))?
        .map_err(|e| e.to_string())?;
        // pax-core's error names the CLI's commands.
        for paper in &mut checked.papers {
            if paper.status == CheckState::Missing {
                paper.error = Some(file_needed_reason());
            }
        }
        Ok(Json(checked))
    }

    /// Download a declared paper's PDF through Nix: records its hash in
    /// `papers.nix` (unless already recorded), then builds it into the Nix
    /// store and returns the path. Can take minutes; it gives up after a
    /// timeout. A paper that can't be fetched (no source URL, download or
    /// build failure, hash mismatch, a local file no longer in the Nix
    /// store, which `file` or a backup restored with `sync`'s `from`
    /// provides again) is reported in `status` and `reason`, not as an
    /// error. An excluded paper's alias is refused: it isn't in the
    /// library.
    ///
    /// With `file`, declares a PDF the user already has instead of
    /// downloading one (as `pax fetch --file` does): it's added to the Nix
    /// store and its hash recorded. A file that doesn't match a hash
    /// already recorded is a `hash-mismatch` unless `force`. A missing or
    /// unreadable file is an error.
    #[tool]
    async fn fetch(
        &self,
        Parameters(params): Parameters<FetchParams>,
    ) -> Result<Json<FetchResult>, String> {
        let path = self.papers_path()?;
        let file = params
            .file
            .as_deref()
            .map(|file| readable_file(&self.root, file))
            .transpose()?;
        if params.force && file.is_none() {
            return Err("`force` only applies with `file`".to_string());
        }
        let source = file.map(|file| (file, params.force));
        // An alias is reported as the citation key it names; an unknown name
        // is left to pax-core to refuse.
        let citation_key = Library::load(&path)
            .ok()
            .and_then(|library| citation_key(&library, &params.reference))
            .unwrap_or(params.reference);
        let (key, root) = (citation_key.clone(), self.root.clone());
        let attempt =
            blocking_with_timeout(FETCH_TIMEOUT, move || fetch_and_build(&key, source, &root))
                .await?;
        let result = match attempt {
            Some(attempt) => fetch_result(&citation_key, attempt)?,
            None => FetchResult::timed_out(
                &citation_key,
                format!(
                    "nix didn't finish within {}s; it may still record the hash, so call \
                     `fetch` again later",
                    FETCH_TIMEOUT.as_secs()
                ),
            ),
        };
        Ok(Json(result))
    }

    /// Fetch every declared paper's PDF through Nix, as `fetch` does for
    /// one (as `pax sync` does): one result per paper, in library order, and
    /// one paper's failure doesn't stop the rest. Needs the network and can
    /// take a long time on a large library.
    ///
    /// With `from` (a directory) or `from_backup: true` (the default
    /// `research/.pdfs/`), first restore the PDFs added from local files
    /// from a backup directory on this machine, as `export` with
    /// `format: "pdfs"` writes one (`<citation_key>.pdf` files plus a
    /// manifest): each such
    /// paper whose file is there is added back to the Nix store when the
    /// file's hash is the one recorded in `papers.nix`, and reported as a
    /// `hash-mismatch` and skipped when it isn't; a recorded hash is never
    /// replaced. Every other paper is fetched as without `from`. A `from`
    /// that isn't a directory, or `from_backup` with no backup in
    /// `research/.pdfs/`, is an error.
    #[tool]
    async fn sync(
        &self,
        Parameters(params): Parameters<SyncParams>,
    ) -> Result<Json<SyncResults>, String> {
        self.papers_path()?;
        let from = match (params.from.as_deref().map(str::trim), params.from_backup) {
            (Some(_), true) => {
                return Err(
                    "give `from` (a backup directory) or `from_backup` (the default \
                     research/.pdfs/), not both"
                        .to_string(),
                );
            }
            (Some(""), false) => {
                return Err(
                    "`from` is blank: give the backup directory, or `from_backup: true` \
                     for the default research/.pdfs/"
                        .to_string(),
                );
            }
            (Some(dir), false) => Some(self.root.join(dir)),
            (None, true) => {
                let dir = pax_core::default_backup_dir(&self.root);
                if !dir.is_dir() {
                    return Err(format!(
                        "no backup in {}: `export` with `format: \"pdfs\"` and no `dir` \
                         writes one there",
                        dir.display()
                    ));
                }
                Some(dir)
            }
            (None, false) => None,
        };
        let root = self.root.clone();
        let reports = tokio::task::spawn_blocking(move || match from {
            Some(dir) => pax_core::sync_library_from(&root, &dir),
            None => pax_core::sync_library(&root),
        })
        .await
        .map_err(|e| format!("sync didn't finish: {e}"))?
        .map_err(|e| e.to_string())?;
        Ok(Json(sync_results(&reports)))
    }

    /// Build or update the library's embedding index (in
    /// research/.pax-ai/, which git ignores), which semantic tools search:
    /// every declared paper's title and venue, plus the full text of each
    /// PDF fetched on this machine, embedded by a local Ollama server.
    /// Incremental: what's already indexed is reused, so a rerun only
    /// embeds new or changed papers, and entries for papers no longer in
    /// the library are deleted. A scanned PDF with no usable text layer
    /// is read by OCR, which can take minutes per paper. Reports one
    /// status per paper: `indexed` with its chunk count (and `ocr: true`
    /// when its text was read by OCR), `fallback` with a reason when only the title
    /// and venue are indexed (no PDF on this machine, or its text couldn't
    /// be extracted or embedded; `fetch` provides a missing PDF, then
    /// `index` again), or `failed` with a reason when nothing could be
    /// embedded (e.g. Ollama isn't running). One paper's failure doesn't
    /// stop the rest. Can take minutes on a large library.
    #[tool]
    async fn index(&self) -> Result<Json<BuildReport>, String> {
        self.index_with(self.embedder.as_ref()).await
    }

    /// Search the library's declared papers by meaning, over the embedding
    /// index `index` builds: the query is embedded by the local Ollama
    /// server and each paper is scored by its closest passage of full text
    /// (or its title and venue, for a `fallback` paper with no PDF
    /// indexed). Returns one hit per paper, best first, with that passage
    /// as `chunk` and its cosine `score`, and lists the `unindexed` papers
    /// by citation key: run `index` to add them. Use it to find papers
    /// discussing a theme or arguing against a claim; `search` finds new
    /// papers from providers instead.
    #[tool]
    async fn semantic_search(
        &self,
        Parameters(params): Parameters<SemanticSearchParams>,
    ) -> Result<Json<SemanticResults>, String> {
        self.semantic_search_with(self.embedder.as_ref(), params)
            .await
    }

    /// Recommend the library's papers most related to one declared paper
    /// by meaning, from the embedding index (`index` builds it): each
    /// paper's full text and title, compared chunk by chunk, so it finds
    /// papers on the same topic whether or not they cite each other (for
    /// citation-based recommendations from providers, use `related`).
    /// Returns up to `limit` papers (default 10), best first, each with a
    /// `score` (higher is closer, at most 1) and `fallback` when only its
    /// title and venue are indexed, which makes its score less reliable;
    /// `seed_fallback` says the same of the paper asked about. `unindexed`
    /// lists the declared papers not in the index (the seed too, if it
    /// isn't, and then nothing is recommended): run `index` to add them.
    /// Reads the index only; nothing is embedded.
    #[tool]
    async fn similar(
        &self,
        Parameters(params): Parameters<SimilarParams>,
    ) -> Result<Json<Similar>, String> {
        self.similar_with(self.embedder.model_id(), params)
    }

    /// Rank papers that aren't in the library, such as `search`, `refs`,
    /// `cited_by` or `related` results, by how close in content each is to
    /// one library paper (`seed`, which must be in the embedding index:
    /// run `index` first). Give the candidates as provider `references`
    /// and/or `dois`, at most 100. Each is looked up again (not recorded in
    /// the search log: the search that found it already was), and embedded
    /// by the local Ollama server from its title and abstract, or its title
    /// alone (`had_abstract: false`, a less reliable score). Returns the
    /// candidates best first, as `search` lists papers plus a `score`
    /// (higher is closer, at most 1), and the lookups that `failed`. Scores
    /// only rank these candidates against this seed: they aren't comparable
    /// with `similar`'s. To rank one page of results in the same call, give
    /// `search`, `refs`, `cited_by` or `related` a `rank_by` instead.
    #[tool]
    async fn rank_candidates(
        &self,
        Parameters(params): Parameters<RankCandidatesParams>,
    ) -> Result<Json<RankedCandidates>, String> {
        self.rank_candidates_with(self.embedder.as_ref(), params)
            .await
    }
}

impl PaxServer {
    /// `rank_candidates` with `embedder`, so tests can refuse requests
    /// offline. Arguments and the seed are checked before any lookup.
    async fn rank_candidates_with(
        &self,
        embedder: &impl Embedder,
        params: RankCandidatesParams,
    ) -> Result<Json<RankedCandidates>, String> {
        let lookups = candidate_lookups(&params.references, &params.dois)?;
        let seed = self.ranking_seed(embedder.model_id(), "seed", &params.seed)?;
        let found = look_up(&self.config, lookups).await;
        self.rank_found(embedder, seed, found).await.map(Json)
    }

    /// The indexed vectors of the seed `param` names, refused when it isn't
    /// indexed for `model_id`, before anything is searched or looked up.
    fn ranking_seed(
        &self,
        model_id: &str,
        param: &str,
        name: &str,
    ) -> Result<PaperVectors, String> {
        let name = name.trim();
        if name.is_empty() {
            return Err(format!(
                "`{param}` must be a citation key or alias, e.g. `turing1936`"
            ));
        }
        self.papers_path()?;
        let seed = PaperIndex::open(&self.root)
            .vectors(model_id, name)
            .map_err(name_error)?;
        if seed.chunks.is_empty() {
            return Err(format!(
                "{name} isn't in the embedding index yet, so nothing can be ranked against it; \
                 run `index` first"
            ));
        }
        Ok(seed)
    }

    /// The looked-up candidates scored against `seed`, best first, flagged
    /// against the library; the failed lookups apart.
    async fn rank_found(
        &self,
        embedder: &impl Embedder,
        seed: PaperVectors,
        found: Vec<(String, Result<CandidateWork, String>)>,
    ) -> Result<RankedCandidates, String> {
        let mut works = Vec::new();
        let mut failed = Vec::new();
        for (reference, result) in found {
            match result {
                Ok(work) => works.push(work),
                Err(reason) => failed.push(FailedLookup { reference, reason }),
            }
        }
        let scores = score_candidates(embedder, &seed, &works)
            .await
            .map_err(|e| format!("embedding the candidates failed: {e}"))?;
        let index = LibraryIndex::load(&self.root);
        let ranked = best_first(&scores)
            .into_iter()
            .flat_map(|i| {
                let CandidateScore {
                    score,
                    had_abstract,
                } = scores[i];
                pax_core::merge_works([works[i].clone()])
                    .works
                    .iter()
                    .map(|merged| RankedCandidate {
                        hit: MergedHit::new(merged, &index, &|_| false, false),
                        score,
                        had_abstract,
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        Ok(RankedCandidates {
            seed: seed.citation_key,
            seed_fallback: seed.fallback,
            ranked,
            failed,
            note: RANKING_NOTE.to_string(),
        })
    }
}

/// The lookups `rank_candidates` asks for: every reference a provider
/// reference, every DOI a DOI, and between 1 and [`RANK_LIMIT`] in all.
fn candidate_lookups(references: &[String], dois: &[String]) -> Result<Vec<Lookup>, String> {
    let count = references.len() + dois.len();
    if count == 0 {
        return Err("give at least one of `references` or `dois` to rank".to_string());
    }
    if count > RANK_LIMIT {
        return Err(format!(
            "{count} candidates is more than {RANK_LIMIT} per call; rank them in several calls"
        ));
    }
    let mut lookups = Vec::with_capacity(count);
    for reference in references {
        let reference = reference.trim();
        let id: CandidateId = reference
            .parse()
            .map_err(|e| format!("{reference:?} in `references`: {e}"))?;
        lookups.push(Lookup::Reference(reference.to_string(), id));
    }
    for doi in dois {
        lookups.push(Lookup::Doi(doi_param(doi)?));
    }
    Ok(lookups)
}

/// Every lookup's result, in the order given, [`LOOKUP_CONCURRENCY`] at a
/// time on tokio's runtime, so no provider call blocks the event thread.
async fn look_up(
    config: &Config,
    lookups: Vec<Lookup>,
) -> Vec<(String, Result<CandidateWork, String>)> {
    let mut found = Vec::with_capacity(lookups.len());
    for batch in lookups.chunks(LOOKUP_CONCURRENCY) {
        let handles: Vec<_> = batch
            .iter()
            .map(|lookup| tokio::spawn(lookup.clone().run(config.clone())))
            .collect();
        for (lookup, handle) in batch.iter().zip(handles) {
            let result = handle
                .await
                .unwrap_or_else(|e| Err(format!("the lookup didn't finish: {e}")));
            found.push((lookup.reference().to_string(), result));
        }
    }
    found
}

/// The provider record of a DOI to rank: the first, in pax-core's merge
/// order, that has an abstract, else the first found; or why none was.
fn doi_record(
    results: &HashMap<ProviderId, Result<Vec<CandidateWork>, ProviderError>>,
) -> Result<CandidateWork, String> {
    let records: Vec<&CandidateWork> = MERGE_ORDER
        .iter()
        .filter_map(|provider| results.get(provider)?.as_ref().ok())
        .flatten()
        .collect();
    let has_abstract = |work: &CandidateWork| {
        work.abstract_text
            .as_deref()
            .is_some_and(|text| !text.trim().is_empty())
    };
    let chosen = records
        .iter()
        .find(|work| has_abstract(work))
        .or(records.first());
    if let Some(work) = chosen {
        return Ok((*work).clone());
    }
    let errors: Vec<String> = MERGE_ORDER
        .iter()
        .filter_map(|provider| match results.get(provider)? {
            Err(ProviderError::NotFound) | Ok(_) => None,
            Err(e) => Some(format!("{provider}: {e}")),
        })
        .collect();
    Err(if errors.is_empty() {
        "no provider has a record of this DOI".to_string()
    } else {
        format!(
            "no provider returned a record of this DOI ({})",
            errors.join("; ")
        )
    })
}

impl PaxServer {
    /// `similar` over the index entries of `model_id`, so tests can read
    /// what they indexed with a fake embedder.
    fn similar_with(&self, model_id: &str, params: SimilarParams) -> Result<Json<Similar>, String> {
        let limit = match params.limit {
            Some(0) => return Err("`limit` must be at least 1".to_string()),
            Some(n) => n,
            None => SIMILAR_LIMIT,
        };
        let name = params.name.trim();
        if name.is_empty() {
            return Err("`name` must be a citation key or alias, e.g. `turing1936`".to_string());
        }
        self.papers_path()?;
        PaperIndex::open(&self.root)
            .similar(model_id, name, limit)
            .map(Json)
            .map_err(name_error)
    }
}

impl PaxServer {
    /// `index` with `embedder`, so tests can index offline.
    async fn index_with(&self, embedder: &impl Embedder) -> Result<Json<BuildReport>, String> {
        self.papers_path()?;
        PaperIndex::open(&self.root)
            .build(embedder)
            .await
            .map(Json)
            .map_err(|e| e.to_string())
    }

    /// `semantic_search` with `embedder`, so tests can search offline.
    async fn semantic_search_with(
        &self,
        embedder: &impl Embedder,
        params: SemanticSearchParams,
    ) -> Result<Json<SemanticResults>, String> {
        let query = params.query.trim();
        if query.is_empty() {
            return Err("`query` is blank: say what to look for".to_string());
        }
        let limit = match params.limit {
            Some(0) => return Err("`limit` must be at least 1".to_string()),
            Some(n) => n,
            None => 10,
        };
        self.papers_path()?;
        PaperIndex::open(&self.root)
            .search(embedder, query, limit)
            .await
            .map(Json)
            .map_err(|e| e.to_string())
    }

    /// The library's `research/papers.nix`, or a clear error when the root
    /// has none.
    fn papers_path(&self) -> Result<PathBuf, String> {
        let path = pax_core::nix::papers_path(&self.root);
        if !path.exists() {
            return Err(format!(
                "no pax library at {} (research/papers.nix not found)",
                self.root.display()
            ));
        }
        Ok(path)
    }

    /// The `rank_by` seed's vectors, if one was given, refused before any
    /// provider is asked when it isn't indexed.
    fn rank_by_seed(
        &self,
        embedder: &impl Embedder,
        rank_by: Option<&str>,
    ) -> Result<Option<PaperVectors>, String> {
        rank_by
            .map(|name| self.ranking_seed(embedder.model_id(), "rank_by", name))
            .transpose()
    }

    /// `search` with `embedder` for `rank_by`, so tests can refuse a seed
    /// offline.
    async fn search_with(
        &self,
        embedder: &impl Embedder,
        params: SearchParams,
    ) -> Result<Json<RankedOutput<SearchResults>>, String> {
        let options = params.paging.options()?;
        let options = options.with_providers(providers(&params.providers)?);
        let (root, config) = (&self.root, &self.config);
        let (query, by_author) = match (params.query, params.author) {
            (Some(query), None) => (query, false),
            (None, Some(author)) => (author, true),
            _ => return Err("give exactly one of `query` or `author`".to_string()),
        };
        let seed = self.rank_by_seed(embedder, params.rank_by.as_deref())?;
        let options = match seed {
            Some(_) => options.with_abstracts(true),
            None => options,
        };
        let logged = if by_author {
            pax_core::search_by_author_logged(&query, &options, root, config).await
        } else {
            pax_core::search_all_logged(&query, &options, root, config).await
        };
        self.results(embedder, seed.as_ref(), logged, options.abstracts)
            .await
            .map(Json)
    }

    /// `refs`, `cited_by` or `related` with `embedder` for `rank_by`, so
    /// tests can rank offline.
    async fn citations_with(
        &self,
        embedder: &impl Embedder,
        mut params: CitationParams,
        tool: CitationTool,
    ) -> Result<Json<RankedOutput<CitationResults>>, String> {
        let options = params.paging.options()?;
        let rank_by = params.rank_by.take();
        let seeds = params.seeds()?;
        let seed = self.rank_by_seed(embedder, rank_by.as_deref())?;
        let options = match seed {
            Some(_) => options.with_abstracts(true),
            None => options,
        };
        let (root, config) = (&self.root, &self.config);
        let (logged, lists) = match tool {
            CitationTool::Refs => (
                pax_core::references_many_logged(&seeds, &options, root, config).await,
                Lists::Citations,
            ),
            CitationTool::CitedBy => (
                pax_core::cited_by_many_logged(&seeds, &options, root, config).await,
                Lists::Citations,
            ),
            CitationTool::Related => (
                pax_core::related_many_logged(&seeds, &options, root, config).await,
                Lists::Related,
            ),
        };
        self.citations(embedder, seed.as_ref(), logged, lists, options.abstracts)
            .await
            .map(Json)
    }

    /// A logged search's results, flagged against the library as it is now,
    /// with each paper's abstract when `abstracts` were asked for, ranked
    /// against `seed` when there is one.
    async fn results<E: Display>(
        &self,
        embedder: &impl Embedder,
        seed: Option<&PaperVectors>,
        logged: Logged<HashMap<ProviderId, Result<SearchPage, E>>>,
        abstracts: bool,
    ) -> Result<RankedOutput<SearchResults>, String> {
        let Logged { results, log, .. } = logged;
        let ranking = match seed {
            Some(seed) => {
                let merged = pax_core::merge_pages(&results);
                Some(rank_merged(embedder, seed, merged.works.iter()).await?)
            }
            None => None,
        };
        let index = LibraryIndex::load(&self.root);
        Ok(RankedOutput {
            output: SearchResults::new(&results, &index, &log, abstracts),
            ranking,
        })
    }

    /// A logged multi-seed citation lookup's results, flagged against the
    /// library as it is now, and ranked against `seed` when there is one.
    /// When no seed was valid, no provider was asked and nothing was logged,
    /// so it's an error naming each seed's problem.
    async fn citations(
        &self,
        embedder: &impl Embedder,
        seed: Option<&PaperVectors>,
        logged: Logged<SeedResults>,
        lists: Lists,
        abstracts: bool,
    ) -> Result<RankedOutput<CitationResults>, String> {
        let Logged { results, log, .. } = logged;
        let invalid: Vec<String> = results
            .seeds
            .iter()
            .filter_map(|seed| seed.results.as_ref().err().map(ToString::to_string))
            .collect();
        if invalid.len() == results.seeds.len() {
            return Err(invalid.join("; "));
        }
        let ranking = match seed {
            Some(seed) => {
                let merged = results.merge();
                let works = merged.works.iter().map(|work| &work.merged);
                Some(rank_merged(embedder, seed, works).await?)
            }
            None => None,
        };
        let index = LibraryIndex::load(&self.root);
        Ok(RankedOutput {
            output: CitationResults::new(&results, &index, &log, lists, abstracts),
            ranking,
        })
    }
}

/// `merged` (as pax-core merges a result list, so in its order) scored
/// against `seed`, best first, each by its representative record's `id`
/// and embedded with the group's abstract.
async fn rank_merged<'a>(
    embedder: &impl Embedder,
    seed: &PaperVectors,
    merged: impl Iterator<Item = &'a MergedWork>,
) -> Result<Vec<RankingEntry>, String> {
    let works: Vec<CandidateWork> = merged
        .map(|merged| {
            let mut work = merged.work.clone();
            work.abstract_text = merged.abstract_text().map(str::to_string);
            work
        })
        .collect();
    let scores = score_candidates(embedder, seed, &works)
        .await
        .map_err(|e| {
            format!(
                "the search ran and was logged, but embedding its results to rank them \
             failed: {e}"
            )
        })?;
    Ok(best_first(&scores)
        .into_iter()
        .map(|i| RankingEntry {
            id: works[i].id.to_string(),
            score: scores[i].score,
            had_abstract: scores[i].had_abstract,
        })
        .collect())
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for PaxServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("pax-mcp", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Tools over a pax research library: search providers for papers \
                 (search, with abstracts on request), follow papers' \
                 references and citations (refs, cited_by) or the providers' \
                 related-work recommendations (related), show one by \
                 reference, citation key, or DOI (show), \
                 add it to the library (add) or exclude it with a reason and \
                 the protocol's criterion (exclude, and unexclude to reverse \
                 an exclusion the reviewer reconsiders), tag and annotate \
                 declared papers (edit), fetch a paper's PDF through Nix \
                 (fetch, which can take minutes) or every paper's (sync, \
                 which with `from` also restores a backup), verify fetched PDFs still \
                 reproduce (check), list what's declared or excluded (list), export it as \
                 BibTeX text or back up the PDFs added from local files to a \
                 directory on this machine (export), and report PRISMA flow \
                 counts (stats). Build or update the library's embedding \
                 index of titles, venues and fetched PDFs' full text \
                 (index), through a local Ollama server, and recommend the \
                 library's papers closest in meaning to one of them from it \
                 (similar). \
                 Search the declared papers by meaning over that index, \
                 each with its best-matching passage (semantic_search). \
                 Rank papers that aren't in the library, such as search \
                 results, by how close in content each is to an indexed \
                 library paper (rank_candidates, or `rank_by` on search, refs, \
                 cited_by and related to rank the page they return). \
                 A paper can carry an alias, such as the review's study ID \
                 P17, set with edit (declared papers) or exclude (excluded \
                 ones): every tool that takes a citation key also takes an \
                 alias, and show, refs, cited_by, related and unexclude also \
                 take an excluded paper's alias. \
                 Writes go to research/ through pax-core, and searches are \
                 recorded in research/search-log.jsonl, under a `label` when \
                 given, which stats counts by.",
            )
    }
}

/// The DOI `show` looks up: trimmed and without a `https://doi.org/` prefix,
/// refused unless it looks like one (`10.<registrant>/<suffix>`), so a
/// provider reference passed as `doi` fails before any request.
fn doi_param(doi: &str) -> Result<String, String> {
    let doi = pax_core::normalize_doi(doi.trim());
    match doi.split_once('/') {
        Some((prefix, suffix)) if prefix.starts_with("10.") && !suffix.is_empty() => {
            Ok(doi.to_string())
        }
        _ => Err(format!(
            "{doi:?} is not a DOI (expected e.g. `10.1145/359545.359563`)"
        )),
    }
}

/// Why `add` failed; a paper already in the library or excluded from the
/// review says so plainly, since an agent is expected to move on to another
/// result (or, for an exclusion, ask the reviewer) rather than treat it as a
/// failure.
fn add_error(e: PaxError) -> String {
    match e {
        PaxError::AlreadyDeclared {
            reference,
            citation_key,
        } => format!("not added: {reference} is already in the library as {citation_key:?}"),
        PaxError::AlreadyExcluded {
            reference,
            key,
            stage,
            reason,
        } => format!(
            "not added: {reference} was excluded at {stage} ({reason:?}); call unexclude \
             with {key:?} first to reverse that decision"
        ),
        other => other.to_string(),
    }
}

/// The citation key of the declared paper `name` names in `library`: its
/// citation key or its alias, looked up in that order, as pax-core does.
fn citation_key(library: &Library, name: &str) -> Option<String> {
    library
        .find(name)
        .or_else(|| library.find_alias(name))
        .map(|paper| paper.local.citation_key.clone())
}

/// Whether `name` is `paper`'s citation key or alias.
fn is_named(paper: &Paper, name: &str) -> bool {
    paper.local.citation_key == name || paper.local.alias.as_deref() == Some(name)
}

/// A pax-core error from a tool that takes or sets a library name, worded
/// for the agent: a taken name says which paper holds it, and an excluded
/// paper's alias given where a declared paper is needed says what takes it.
fn name_error(e: PaxError) -> String {
    match e {
        PaxError::NameTaken { name, holder } => {
            let holder = match holder {
                NameHolder::Declared(key) => format!("the paper {key:?}"),
                NameHolder::Excluded(key) => format!("the excluded paper {key:?}"),
                other => other.to_string(),
            };
            format!(
                "{name:?} is already taken by {holder}, as its alias or citation key; aliases \
                 and citation keys share one namespace, so choose another"
            )
        }
        PaxError::ExcludedNotDeclared { name, key } => format!(
            "{name} names a paper excluded from the review (as {key:?}), not one in the \
             library; show, refs, cited_by, related and unexclude take it"
        ),
        other => other.to_string(),
    }
}

/// The tag, notes and alias edits `edit` asks for. Blank tags are refused,
/// since they'd be written into `papers.nix` as empty strings, and so is an
/// edit with nothing in it (pax-core's own message for that names the CLI's
/// flags). An empty `alias` clears the paper's alias.
fn paper_edits(
    add_tags: Vec<String>,
    remove_tags: Vec<String>,
    notes: Option<String>,
    alias: Option<String>,
) -> Result<PaperEdits, String> {
    if add_tags
        .iter()
        .chain(&remove_tags)
        .any(|tag| tag.trim().is_empty())
    {
        return Err("tags must not be blank".to_string());
    }
    if add_tags.is_empty() && remove_tags.is_empty() && notes.is_none() && alias.is_none() {
        return Err(
            "nothing to edit: give `add_tags`, `remove_tags`, `notes`, and/or `alias`".to_string(),
        );
    }
    Ok(PaperEdits::new()
        .with_add_tags(add_tags)
        .with_remove_tags(remove_tags)
        .with_notes(notes)
        .with_alias(alias))
}

/// The papers `export` renders, in library order: those named in `keys`
/// (by citation key or alias, when given) that carry `tag` (when given). A
/// name that isn't in the library is an error rather than silently missing
/// from the export.
fn select_papers(
    papers: &[Paper],
    keys: Option<&[String]>,
    tag: Option<&str>,
) -> Result<Vec<Paper>, String> {
    if let Some(keys) = keys {
        let unknown: Vec<&str> = keys
            .iter()
            .filter(|name| !papers.iter().any(|p| is_named(p, name)))
            .map(String::as_str)
            .collect();
        if !unknown.is_empty() {
            return Err(format!("not in the library: {}", unknown.join(", ")));
        }
    }
    let filter = ListFilter {
        tag: tag.map(str::to_string),
        ..ListFilter::default()
    };
    Ok(pax_core::filter_papers(papers, &filter)
        .into_iter()
        .filter(|p| keys.is_none_or(|names| names.iter().any(|name| is_named(p, name))))
        .collect())
}

/// How long `fetch` waits. pax-core already bounds each `nix` call it makes
/// (prefetch, then `git add` and build), so this only caps their sum.
const FETCH_TIMEOUT: Duration = Duration::from_secs(300);

/// `fetch`'s two pax-core steps: record the hash (downloading, unless one is
/// recorded already, or adding `file`), then build the PDF to get its store
/// path.
type FetchAttempt = Result<(FetchOutcome, Result<ResolvedArtifact, PaxError>), PaxError>;

/// `file` is a local PDF to add instead of downloading, with `force`.
fn fetch_and_build(citation_key: &str, file: Option<(PathBuf, bool)>, root: &Path) -> FetchAttempt {
    let outcome = match file {
        Some((file, force)) => pax_core::fetch_file(citation_key, &file, force, root)?,
        None => pax_core::fetch_paper(citation_key, root)?,
    };
    Ok((outcome, pax_core::resolve_artifact_path(citation_key, root)))
}

/// A paper that couldn't be fetched, as pax-core reports it, with the
/// reason worded for the agent where pax-core's names the CLI's flags.
fn failed_fetch(citation_key: &str, e: &PaxError) -> FetchResult {
    let mut result = FetchResult::failed(citation_key, e);
    let reason = match e {
        PaxError::NoSourceUrl(_) => "no PDF source is recorded (the provider found no \
             open-access copy when it was added), so there's nothing to fetch; if the user \
             has the PDF (e.g. through a university login), fetch again with `file` set to \
             its path, or record a source with `pax edit --source-url`"
            .to_string(),
        PaxError::HashMismatch {
            recorded, actual, ..
        } => format!(
            "a different file is already recorded (hash {recorded}; this file's is \
             {actual}), so nothing was written; fetch again with `force: true` to replace it"
        ),
        PaxError::BackupMismatch {
            file,
            recorded,
            actual,
            ..
        } => format!(
            "{} isn't the PDF recorded for this paper (hash {recorded}; the file's is \
             {actual}), so it was skipped and nothing was written; the backup may be of \
             another version of the paper",
            file.display()
        ),
        PaxError::LocalArtifactMissing(_) => file_needed_reason(),
        _ => return result,
    };
    result.reason = Some(reason);
    result
}

/// pax-core's [`SyncResults`], with each failed paper's reason worded as
/// `fetch` words it.
/// The warning for a `pdfs` export to `directory`, inside `research/` and
/// not ignored by git: pax-core's CLI warning, worded for the agent.
fn staged_by_git_warning(directory: &str) -> String {
    format!(
        "{directory} is in research/ but not ignored by git; pax stages research/ \
         for git (`git add -A`) before every build, so these PDFs would be \
         committed with the library. Leave `dir` out to use the default \
         research/.pdfs/, choose one outside research/, or add this one to \
         research/.gitignore"
    )
}

fn sync_results(reports: &[SyncReport]) -> SyncResults {
    let mut results = SyncResults::new(reports);
    for (result, report) in results.papers.iter_mut().zip(reports) {
        if let Err(e) = &report.result {
            result.reason = failed_fetch(&report.citation_key, e).reason;
        }
    }
    results
}

/// What a fetch attempt means for the agent. Only a bad citation key or an
/// unreadable library is an error; a paper that can't be fetched is a
/// result, with the reason.
fn fetch_result(citation_key: &str, attempt: FetchAttempt) -> Result<FetchResult, String> {
    let (outcome, built) = match attempt {
        Ok(fetched) => fetched,
        Err(
            e @ (PaxError::NoSourceUrl(_)
            | PaxError::HashMismatch { .. }
            | PaxError::LocalArtifactMissing(_)
            | PaxError::Fetch(_)),
        ) => return Ok(failed_fetch(citation_key, &e)),
        Err(e) => return Err(name_error(e)),
    };
    // The hash is recorded either way; only building the PDF can still fail.
    let not_built = |status: FetchState, reason: String| {
        let mut result = FetchResult::new(citation_key, &outcome);
        result.status = status;
        result.reason = Some(reason);
        result
    };
    Ok(match built {
        Ok(ResolvedArtifact::Path(path)) => FetchResult::new(citation_key, &outcome)
            .with_store_path(Some(path.display().to_string())),
        Err(PaxError::Build(message)) if message.contains("hash mismatch") => not_built(
            FetchState::HashMismatch,
            format!(
                "the source no longer matches the recorded hash; `check` reports it too: \
                 {message}"
            ),
        ),
        Err(PaxError::Build(message)) if message.contains("not tracked by Git") => not_built(
            FetchState::BuildFailed,
            format!(
                "research/ isn't tracked by git, and nix only sees tracked files; `git add` \
                 it in the repository that holds it, then fetch again: {message}"
            ),
        ),
        Err(PaxError::LocalArtifactMissing(_)) => {
            not_built(FetchState::FileNeeded, file_needed_reason())
        }
        Err(e) => not_built(FetchState::BuildFailed, e.to_string()),
        Ok(_) => not_built(
            FetchState::BuildFailed,
            "the hash is recorded, but the PDF still couldn't be resolved".to_string(),
        ),
    })
}

/// `fetch`'s `file`, resolved against the library `root` when relative, or
/// an error naming it when it can't be opened (missing, or unreadable).
/// pax-core refuses a path that opens but isn't a file (`NotAFile`).
fn readable_file(root: &Path, file: &str) -> Result<PathBuf, String> {
    let path = root.join(file);
    std::fs::File::open(&path).map_err(|e| format!("can't read {}: {e}", path.display()))?;
    Ok(path)
}

/// `fetch`'s and `sync`'s reason for `file-needed`, and `check`'s and
/// `export`'s error for `missing`: pax-core's names the CLI's commands
/// (`--file`, `sync --from`).
fn file_needed_reason() -> String {
    "this paper was added from a local file, which isn't in this Nix store; fetch again \
     with `file` set to the PDF's path to provide it again, or, if its PDFs were backed up \
     with `export` (`format: \"pdfs\"`), restore the backup with `sync`, `from` set to its \
     directory"
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::PaperStatus;
    use crate::index::extract::tests::{pdf_with_lines, pdftotext_available};
    use crate::index::tests::FakeEmbedder;
    use crate::port::EmbedKind;
    use pax_core::output::SearchLogStatus;
    use pax_core::{CandidateWork, Exclusion, Exclusions, LogOutcome, ProviderError};
    use std::fs;

    fn library_with_two_papers(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("pax-mcp-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("research")).unwrap();
        fs::write(
            root.join("research/papers.nix"),
            r#"{
  turing1936 = {
    doi = "10.1112/plms/s2-42.1.230";
    title = "On Computable Numbers";
    authors = [ "Alan Turing" ];
    year = 1936;
    venue = null;
    source_url = null;
    hash = "sha256-AAAA";
    tags = [ "computability" ];
    notes = null;
  };
  hewitt1973 = {
    doi = null;
    title = "A Universal Modular Actor Formalism";
    authors = [ "Carl Hewitt" ];
    year = 1973;
    venue = null;
    source_url = null;
    hash = null;
    tags = [ "concurrency" ];
    notes = null;
  };
}
"#,
        )
        .unwrap();
        root
    }

    fn work(
        provider: ProviderId,
        native_id: &str,
        title: &str,
        doi: Option<&str>,
    ) -> CandidateWork {
        CandidateWork {
            id: CandidateId {
                provider,
                native_id: native_id.to_string(),
            },
            title: title.to_string(),
            authors: Vec::new(),
            publish_date: "2017-06-12".to_string(),
            doi: doi.map(str::to_string),
            pdf_url: None,
            venue: None,
            abstract_text: None,
        }
    }

    #[test]
    fn registers_every_tool() {
        let server = PaxServer::new(PathBuf::from("."), Config::default());
        let mut names: Vec<String> = server
            .tool_router
            .list_all()
            .into_iter()
            .map(|tool| tool.name.to_string())
            .collect();
        names.sort();
        assert_eq!(
            names,
            [
                "add",
                "check",
                "cited_by",
                "edit",
                "exclude",
                "export",
                "fetch",
                "index",
                "list",
                "rank_candidates",
                "refs",
                "related",
                "search",
                "semantic_search",
                "show",
                "similar",
                "stats",
                "sync",
                "unexclude",
            ]
        );
    }

    #[tokio::test]
    async fn list_filters_declared_papers() {
        let root = library_with_two_papers("list");
        let server = PaxServer::new(root.clone(), Config::default());

        let Json(all) = server
            .list(Parameters(ListParams::default()))
            .await
            .unwrap();
        assert!(all.excluded.is_none());
        assert_eq!(all.declared.unwrap().papers.len(), 2);

        let Json(tagged) = server
            .list(Parameters(ListParams {
                tag: Some("computability".to_string()),
                ..ListParams::default()
            }))
            .await
            .unwrap();
        let tagged = tagged.declared.unwrap();
        assert_eq!(tagged.papers.len(), 1);
        assert_eq!(tagged.papers[0].citation_key, "turing1936");
        assert!(tagged.papers[0].fetched);

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn list_excluded_filters_exclusions_exactly() {
        let root = library_with_two_papers("list-excluded");
        let server = PaxServer::new(root.clone(), Config::default());

        let Json(none) = server
            .list(Parameters(ListParams {
                excluded: true,
                ..ListParams::default()
            }))
            .await
            .unwrap();
        assert!(none.declared.is_none());
        assert!(none.excluded.unwrap().exclusions.is_empty());

        for (doi, year, criterion, round) in [
            ("10.1000/a", 2019, "X3", "R1"),
            ("10.1000/b", 2020, "X3", "R2"),
            ("10.1000/c", 2020, "X1", "R1"),
        ] {
            let Parameters(params) = exclusion(doi, "off topic", Some("A Paper"));
            server
                .exclude(Parameters(ExcludeParams {
                    year: Some(year),
                    criterion: Some(criterion.to_string()),
                    round: Some(round.to_string()),
                    ..params
                }))
                .await
                .unwrap();
        }

        let keys = |params: ListParams| {
            let server = server.clone();
            async move {
                let Json(out) = server
                    .list(Parameters(ListParams {
                        excluded: true,
                        ..params
                    }))
                    .await
                    .unwrap();
                assert!(out.declared.is_none());
                out.excluded
                    .unwrap()
                    .exclusions
                    .into_iter()
                    .map(|e| e.key)
                    .collect::<Vec<_>>()
            }
        };
        assert_eq!(
            keys(ListParams::default()).await,
            ["10.1000/a", "10.1000/b", "10.1000/c"]
        );
        assert_eq!(
            keys(ListParams {
                year: Some(2020),
                ..ListParams::default()
            })
            .await,
            ["10.1000/b", "10.1000/c"]
        );
        assert_eq!(
            keys(ListParams {
                criterion: Some("X3".to_string()),
                round: Some("R1".to_string()),
                ..ListParams::default()
            })
            .await,
            ["10.1000/a"]
        );
        assert!(
            keys(ListParams {
                round: Some("r1".to_string()),
                ..ListParams::default()
            })
            .await
            .is_empty(),
            "round matches exactly"
        );

        let Json(out) = server
            .list(Parameters(ListParams {
                excluded: true,
                criterion: Some("X1".to_string()),
                ..ListParams::default()
            }))
            .await
            .unwrap();
        let json = serde_json::to_value(&out).unwrap();
        assert_eq!(json.as_object().unwrap().len(), 1, "{json}");
        let exclusion = &json["exclusions"][0];
        assert_eq!(
            (
                &exclusion["key"],
                &exclusion["title"],
                &exclusion["year"],
                &exclusion["reason"],
                &exclusion["stage"],
                &exclusion["criterion"],
                &exclusion["round"],
                &exclusion["alias"],
            ),
            (
                &serde_json::json!("10.1000/c"),
                &serde_json::json!("A Paper"),
                &serde_json::json!(2020),
                &serde_json::json!("off topic"),
                &serde_json::json!("full-text"),
                &serde_json::json!("X1"),
                &serde_json::json!("R1"),
                &serde_json::Value::Null,
            )
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn list_refuses_filters_of_the_other_set() {
        let root = library_with_two_papers("list-mixed");
        let server = PaxServer::new(root.clone(), Config::default());

        let err = server
            .list(Parameters(ListParams {
                excluded: true,
                tag: Some("computability".to_string()),
                ..ListParams::default()
            }))
            .await
            .err()
            .unwrap();
        assert!(err.contains("`tag`"), "{err}");
        let err = server
            .list(Parameters(ListParams {
                round: Some("R1".to_string()),
                ..ListParams::default()
            }))
            .await
            .err()
            .unwrap();
        assert!(err.contains("excluded: true"), "{err}");

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn show_resolves_a_citation_key_without_the_network() {
        let root = library_with_two_papers("show");
        let server = PaxServer::new(root.clone(), Config::default());

        let Json(shown) = server
            .show(Parameters(ShowParams {
                reference: Some("hewitt1973".to_string()),
                ..ShowParams::default()
            }))
            .await
            .unwrap();
        assert!(shown.records.is_none());
        let shown = shown.shown.expect("a reference is shown");
        assert!(shown.candidate.is_none());
        assert_eq!(shown.declared.unwrap().authors, ["Carl Hewitt"]);

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn search_rejects_both_or_neither_mode() {
        let server = PaxServer::new(PathBuf::from("."), Config::default());
        for params in [
            SearchParams::default(),
            SearchParams {
                query: Some("actor model".to_string()),
                author: Some("Hewitt".to_string()),
                ..SearchParams::default()
            },
        ] {
            assert!(server.search(Parameters(params)).await.is_err());
        }
    }

    #[test]
    fn search_providers_parse_as_the_clis_and_refuse_unknown_names() {
        assert_eq!(providers(&[]), Ok(vec![]));
        assert_eq!(
            providers(&["dblp".to_string(), " openalex ".to_string()]),
            Ok(vec![ProviderId::Dblp, ProviderId::OpenAlex])
        );
        let error = providers(&["scholar".to_string()]).unwrap_err();
        assert!(
            error.contains("scholar") && error.contains("dblp"),
            "{error}"
        );
        let options =
            SearchOptions::default().with_providers(providers(&["dblp".to_string()]).unwrap());
        assert_eq!(options.search_providers(), vec![ProviderId::Dblp]);
    }

    #[tokio::test]
    async fn search_refuses_an_unknown_provider_before_searching() {
        let server = PaxServer::new(PathBuf::from("."), Config::default());
        let error = server
            .search(Parameters(SearchParams {
                query: Some("actor model".to_string()),
                providers: vec!["scholar".to_string()],
                ..SearchParams::default()
            }))
            .await
            .err()
            .expect("the search should be refused");
        assert!(error.contains("unknown provider"), "{error}");
    }

    #[tokio::test]
    async fn search_rejects_invalid_paging_before_searching() {
        let server = PaxServer::new(PathBuf::from("."), Config::default());
        let error = server
            .search(Parameters(SearchParams {
                query: Some("actor model".to_string()),
                paging: PagingParams {
                    year: Some("1970s".to_string()),
                    ..PagingParams::default()
                },
                ..SearchParams::default()
            }))
            .await
            .err()
            .expect("the search should be refused");
        assert!(error.contains("1970s"), "{error}");
    }

    #[test]
    fn search_params_take_paging_fields_at_the_top_level() {
        let params: SearchParams = serde_json::from_value(serde_json::json!({
            "query": "actor model",
            "limit": 10,
            "year": "1970-1980",
            "type": "journal-article",
            "language": "EN",
            "label": " R2-backward ",
            "with_abstracts": true,
        }))
        .unwrap();
        let options = params.paging.options().unwrap();
        assert_eq!(options.label.as_deref(), Some("R2-backward"));
        assert!(options.abstracts);
        assert_eq!(options.limit, Limit::Count(10));
        assert_eq!(
            options.filters.years,
            YearRange::new(Some(1970), Some(1980))
        );
        assert_eq!(options.filters.doc_type, Some(DocType::JournalArticle));
        assert_eq!(options.filters.language.as_deref(), Some("en"));
    }

    #[test]
    fn paging_defaults_to_pax_cores_limit_and_no_filters() {
        let options = PagingParams::default().options().unwrap();
        assert_eq!(options, SearchOptions::default());

        let all = PagingParams {
            all: true,
            ..PagingParams::default()
        };
        assert_eq!(all.options().unwrap().limit, Limit::All);
    }

    #[test]
    fn paging_rejects_conflicting_or_invalid_values() {
        let invalid = [
            PagingParams {
                limit: Some(10),
                all: true,
                ..PagingParams::default()
            },
            PagingParams {
                limit: Some(0),
                ..PagingParams::default()
            },
            PagingParams {
                year: Some("2020-2015".to_string()),
                ..PagingParams::default()
            },
            PagingParams {
                doc_type: Some("blog-post".to_string()),
                ..PagingParams::default()
            },
            PagingParams {
                language: Some(" ".to_string()),
                ..PagingParams::default()
            },
            PagingParams {
                label: Some("  ".to_string()),
                ..PagingParams::default()
            },
        ];
        for params in invalid {
            assert!(params.options().is_err(), "{params:?}");
        }
    }

    #[test]
    fn search_results_merge_providers_and_flag_library_and_exclusions() {
        let root = library_with_two_papers("results");
        let library = Library::load(&pax_core::nix::papers_path(&root)).unwrap();
        let mut exclusions = Exclusions::default();
        exclusions.upsert(Exclusion::new(
            "10.1000/excluded".to_string(),
            "Excluded".to_string(),
            "off topic".to_string(),
            ExclusionStage::Screening,
        ));
        let index = LibraryIndex::new(&library, &exclusions);

        let turing = "10.1112/plms/s2-42.1.230";
        let openalex = SearchPage::new(
            vec![
                work(
                    ProviderId::OpenAlex,
                    "W1",
                    "On Computable Numbers",
                    Some(turing),
                ),
                work(
                    ProviderId::OpenAlex,
                    "W2",
                    "Excluded",
                    Some("10.1000/excluded"),
                ),
            ],
            Some(120),
            None,
        );
        let crossref = SearchPage::new(
            vec![
                work(
                    ProviderId::Crossref,
                    turing,
                    "On Computable Numbers",
                    Some(turing),
                ),
                work(
                    ProviderId::Crossref,
                    "10.1000/new",
                    "New",
                    Some("10.1000/new"),
                ),
            ],
            Some(2),
            None,
        );
        let results: HashMap<ProviderId, Result<SearchPage, ProviderError>> = HashMap::from([
            (ProviderId::OpenAlex, Ok(openalex)),
            (ProviderId::Crossref, Ok(crossref)),
            (
                ProviderId::ArXiv,
                Err(ProviderError::Request("timed out".to_string())),
            ),
        ]);

        let out = SearchResults::new(&results, &index, &LogOutcome::NoLibrary, false);

        let providers: Vec<_> = out
            .providers
            .iter()
            .map(|p| (p.provider.as_str(), p.retrieved, p.total, p.complete))
            .collect();
        assert_eq!(
            providers,
            [
                ("openalex", 2, Some(120), false),
                ("crossref", 2, Some(2), true),
                ("arxiv", 0, None, false),
            ]
        );
        assert!(
            out.providers[2]
                .error
                .as_deref()
                .unwrap()
                .contains("timed out")
        );

        assert_eq!(
            (out.merge.before, out.merge.after, out.merge.duplicates),
            (4, 3, 1)
        );
        let turing_hit = &out.results[0];
        assert_eq!(turing_hit.work.id, "openalex:W1");
        assert_eq!(turing_hit.providers, ["openalex", "crossref"]);
        assert_eq!(
            turing_hit.ids,
            ["openalex:W1", &format!("crossref:{turing}")]
        );
        assert!(turing_hit.work.in_library && !turing_hit.work.excluded);
        assert!(out.results[1].work.excluded && !out.results[1].work.in_library);
        assert!(!out.results[2].work.in_library && !out.results[2].work.excluded);

        assert!(!out.log.logged);
        assert!(out.log.reason.is_some());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn search_results_report_filters_and_unknown_languages() {
        use pax_core::{FilterApplication, FilterKind};

        let unknown = work(ProviderId::Crossref, "10.1000/a", "A", Some("10.1000/a"));
        let page = SearchPage::new(vec![unknown.clone()], Some(40), None)
            .with_scanned(20)
            .with_filters(vec![
                pax_core::AppliedFilter::new(FilterKind::Year, FilterApplication::Server),
                pax_core::AppliedFilter::new(
                    FilterKind::Language,
                    FilterApplication::PostRetrieval,
                ),
            ])
            .with_language_unknown(vec![unknown.id.clone()]);
        let results: HashMap<ProviderId, Result<SearchPage, ProviderError>> =
            HashMap::from([(ProviderId::Crossref, Ok(page))]);

        let out = SearchResults::new(
            &results,
            &LibraryIndex::default(),
            &LogOutcome::Written(PathBuf::from("research/search-log.jsonl")),
            false,
        );

        let report = &out.providers[0];
        assert_eq!((report.scanned, report.language_unknown), (20, 1));
        let filters: Vec<_> = report
            .filters
            .iter()
            .map(|f| (f.filter.as_str(), f.applied.as_str()))
            .collect();
        assert_eq!(
            filters,
            [("year", "server-side"), ("language", "post-retrieval")]
        );
        assert!(out.results[0].language_unknown);
        assert!(out.log.logged);
        assert_eq!(out.log.path.as_deref(), Some("research/search-log.jsonl"));
    }

    #[test]
    fn abstracts_are_sent_only_when_asked_for() {
        let turing = "10.1112/plms/s2-42.1.230";
        let openalex = CandidateWork {
            abstract_text: Some("From OpenAlex".to_string()),
            ..work(
                ProviderId::OpenAlex,
                "W1",
                "On Computable Numbers",
                Some(turing),
            )
        };
        let crossref = CandidateWork {
            abstract_text: Some("From Crossref".to_string()),
            ..work(
                ProviderId::Crossref,
                turing,
                "On Computable Numbers",
                Some(turing),
            )
        };
        let none = work(ProviderId::Crossref, "10.1000/a", "A", Some("10.1000/a"));
        let results: HashMap<ProviderId, Result<SearchPage, ProviderError>> = HashMap::from([
            (
                ProviderId::OpenAlex,
                Ok(SearchPage::new(vec![openalex], Some(1), None)),
            ),
            (
                ProviderId::Crossref,
                Ok(SearchPage::new(vec![crossref, none], Some(2), None)),
            ),
        ]);
        let out = |abstracts| {
            SearchResults::new(
                &results,
                &LibraryIndex::default(),
                &LogOutcome::NoLibrary,
                abstracts,
            )
        };

        let asked = serde_json::to_value(out(true)).unwrap();
        // Whichever record represents the paper, the abstract is taken in
        // pax-core's provider order: OpenAlex before Crossref.
        assert_eq!(
            asked["results"][0]["providers"].as_array().unwrap().len(),
            2
        );
        assert_eq!(asked["results"][0]["abstract"], "From OpenAlex");
        assert!(asked["results"][1]["abstract"].is_null());
        assert!(
            asked["results"][1]
                .as_object()
                .unwrap()
                .contains_key("abstract")
        );

        let unasked = serde_json::to_value(out(false)).unwrap();
        for hit in unasked["results"].as_array().unwrap() {
            assert!(!hit.as_object().unwrap().contains_key("abstract"), "{hit}");
        }
    }

    #[test]
    fn a_failed_log_write_is_reported_without_failing_the_results() {
        let status = SearchLogStatus::from(&LogOutcome::Failed(std::io::Error::other("disk full")));
        assert!(!status.logged);
        assert!(status.reason.unwrap().contains("disk full"));
    }

    fn seeds(references: &[&str]) -> Parameters<CitationParams> {
        Parameters(CitationParams {
            references: Some(references.iter().map(|r| r.to_string()).collect()),
            ..CitationParams::default()
        })
    }

    #[tokio::test]
    async fn refs_and_cited_by_refuse_only_invalid_seeds_without_logging_them() {
        let root = library_with_two_papers("citations");
        let server = PaxServer::new(root.clone(), Config::default());
        let seed = |reference: &str| {
            Parameters(CitationParams {
                reference: Some(reference.to_string()),
                ..CitationParams::default()
            })
        };

        let unknown_key = server
            .refs(seed("dijkstra1968"))
            .await
            .err()
            .expect("an unknown citation key should be refused");
        assert!(unknown_key.contains("dijkstra1968"), "{unknown_key}");
        let bad_references = server
            .cited_by(seeds(&["nowhere:123", "dijkstra1968"]))
            .await
            .err()
            .expect("a call with no valid seed should be refused");
        assert!(bad_references.contains("nowhere"), "{bad_references}");
        assert!(bad_references.contains("dijkstra1968"), "{bad_references}");
        assert!(!root.join("research/search-log.jsonl").exists());

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn an_unresolvable_seed_is_reported_on_that_seed_while_the_others_answer() {
        let root = library_with_two_papers("citations-unresolvable");
        let server = PaxServer::new(root.clone(), Config::default());

        // hewitt1973 has no DOI or arXiv id, so neither provider can look it
        // up and no request is made; dijkstra1968 isn't in the library.
        let Json(out) = server
            .refs(seeds(&["hewitt1973", "dijkstra1968", "hewitt1973"]))
            .await
            .unwrap();

        let hewitt = &out.seeds[0];
        assert_eq!(hewitt.seed, "hewitt1973");
        assert!(hewitt.error.is_none() && !hewitt.complete);
        let incomplete: Vec<_> = hewitt
            .incomplete
            .iter()
            .map(|list| (list.provider.as_str(), list.reason.as_str()))
            .collect();
        assert_eq!(
            incomplete,
            [
                ("openalex", "unresolved"),
                ("semanticscholar", "unresolved")
            ]
        );
        assert_eq!(hewitt.providers.len(), 2);

        let dijkstra = &out.seeds[1];
        assert!(dijkstra.error.as_deref().unwrap().contains("dijkstra1968"));
        assert!(dijkstra.providers.is_empty() && !dijkstra.complete);

        assert_eq!(out.seeds[2].same_as.as_deref(), Some("hewitt1973"));
        assert!(!out.complete);
        assert!(out.results.is_empty());
        assert!(out.log.logged, "the valid seeds' lookups are logged");

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn a_label_lands_on_every_log_line_the_tool_writes() {
        let root = library_with_two_papers("citations-label");
        let server = PaxServer::new(root.clone(), Config::default());

        // hewitt1973 resolves at no provider, so nothing is requested, but
        // each provider's lookup is still logged.
        let Json(out) = server
            .cited_by(Parameters(CitationParams {
                reference: Some("hewitt1973".to_string()),
                paging: PagingParams {
                    label: Some("R2-forward".to_string()),
                    ..PagingParams::default()
                },
                ..CitationParams::default()
            }))
            .await
            .unwrap();
        assert!(out.log.logged);

        let log = fs::read_to_string(root.join("research/search-log.jsonl")).unwrap();
        let lines: Vec<serde_json::Value> = log
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines.len(), 2, "{log}");
        for line in &lines {
            assert_eq!(line["label"], "R2-forward", "{line}");
        }

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn citation_params_take_one_seed_or_several() {
        let params = |value: serde_json::Value| {
            serde_json::from_value::<CitationParams>(value)
                .unwrap()
                .seeds()
        };
        assert_eq!(
            params(serde_json::json!({ "reference": "openalex:W1" })).unwrap(),
            ["openalex:W1"]
        );
        assert_eq!(
            params(serde_json::json!({ "references": ["openalex:W1", "turing1936"], "limit": 5 }))
                .unwrap(),
            ["openalex:W1", "turing1936"]
        );
        for invalid in [
            serde_json::json!({}),
            serde_json::json!({ "references": [] }),
            serde_json::json!({ "reference": "openalex:W1", "references": ["openalex:W2"] }),
        ] {
            assert!(params(invalid.clone()).is_err(), "{invalid}");
        }
    }

    #[tokio::test]
    async fn refs_and_cited_by_reject_invalid_paging_before_any_lookup() {
        let server = PaxServer::new(PathBuf::from("."), Config::default());
        let params = || {
            Parameters(CitationParams {
                reference: Some("openalex:W2072794470".to_string()),
                paging: PagingParams {
                    doc_type: Some("blog-post".to_string()),
                    ..PagingParams::default()
                },
                ..CitationParams::default()
            })
        };
        assert!(server.refs(params()).await.is_err());
        assert!(server.cited_by(params()).await.is_err());
        assert!(server.related(params()).await.is_err());
    }

    #[tokio::test]
    async fn related_reports_unresolvable_seeds_and_logs_them() {
        let root = library_with_two_papers("related");
        let server = PaxServer::new(root.clone(), Config::default());

        // As for refs: hewitt1973 resolves at no provider, so no request is
        // made, and dijkstra1968 isn't in the library.
        let Json(out) = server
            .related(seeds(&["hewitt1973", "dijkstra1968"]))
            .await
            .unwrap();
        let reasons: Vec<_> = out.seeds[0]
            .incomplete
            .iter()
            .map(|list| list.reason.as_str())
            .collect();
        assert_eq!(reasons, ["unresolved", "unresolved"]);
        assert!(
            out.seeds[1]
                .error
                .as_deref()
                .unwrap()
                .contains("dijkstra1968")
        );
        assert!(!out.complete && out.results.is_empty());
        assert!(out.log.logged);
        let log = fs::read_to_string(root.join("research/search-log.jsonl")).unwrap();
        assert!(
            log.lines()
                .all(|line| line.contains(r#""command":"related""#)),
            "{log}"
        );

        let Json(stats) = server.stats().await.unwrap();
        assert_eq!(stats.identified.total, 0);
        assert!(stats.identified.related_works.is_empty());

        fs::remove_dir_all(root).unwrap();
    }

    fn exclusion(reference: &str, reason: &str, title: Option<&str>) -> Parameters<ExcludeParams> {
        Parameters(ExcludeParams {
            reference: reference.to_string(),
            reason: reason.to_string(),
            stage: Stage::FullText,
            title: title.map(str::to_string),
            year: None,
            criterion: None,
            round: None,
            alias: None,
        })
    }

    #[tokio::test]
    async fn exclude_records_a_doi_with_a_title_without_the_network() {
        let root = library_with_two_papers("exclude");
        let server = PaxServer::new(root.clone(), Config::default());

        let Parameters(params) = exclusion(
            "https://doi.org/10.1000/Off-Topic",
            "not empirical",
            Some("An Essay"),
        );
        let Json(excluded) = server
            .exclude(Parameters(ExcludeParams {
                year: Some(2019),
                criterion: Some("X3".to_string()),
                round: Some(" R1 ".to_string()),
                ..params
            }))
            .await
            .unwrap();
        assert_eq!(excluded.key, "10.1000/off-topic");
        assert_eq!(excluded.stage, "full-text");
        assert_eq!(
            (excluded.title.as_str(), excluded.reason.as_str()),
            ("An Essay", "not empirical")
        );
        assert_eq!(excluded.year, Some(2019));
        assert_eq!(excluded.criterion.as_deref(), Some("X3"));
        assert_eq!(excluded.round.as_deref(), Some("R1"));

        let excluded_work = work(
            ProviderId::OpenAlex,
            "W9",
            "An Essay",
            Some("10.1000/off-topic"),
        );
        assert!(LibraryIndex::load(&root).is_excluded(&excluded_work));
        let recorded = Exclusions::load(&pax_core::nix::excluded_path(&root)).unwrap();
        let recorded = recorded.get("10.1000/off-topic").unwrap();
        assert_eq!(
            (
                recorded.year,
                recorded.criterion.as_deref(),
                recorded.round.as_deref()
            ),
            (Some(2019), Some("X3"), Some("R1"))
        );
        let Json(listed) = server
            .list(Parameters(ListParams::default()))
            .await
            .unwrap();
        let listed = listed.declared.unwrap();
        assert_eq!(listed.papers.len(), 2, "excluding doesn't declare anything");

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn exclude_refuses_a_missing_reason_and_a_declared_paper() {
        let root = library_with_two_papers("exclude-refused");
        let server = PaxServer::new(root.clone(), Config::default());

        let no_reason = server
            .exclude(exclusion("10.1000/x", "  ", Some("X")))
            .await
            .err()
            .expect("an empty reason should be refused");
        assert!(no_reason.contains("reason"), "{no_reason}");

        let declared = server
            .exclude(exclusion(
                "10.1112/plms/s2-42.1.230",
                "out of scope",
                Some("On Computable Numbers"),
            ))
            .await
            .err()
            .expect("a declared paper should be refused");
        assert!(declared.contains("turing1936"), "{declared}");
        assert!(!root.join("research/excluded.nix").exists());

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn unexclude_removes_an_exclusion_by_its_doi_without_the_network() {
        let root = library_with_two_papers("unexclude");
        let server = PaxServer::new(root.clone(), Config::default());
        let Parameters(params) = exclusion("10.1000/essay", "not empirical", Some("An Essay"));
        server
            .exclude(Parameters(ExcludeParams {
                criterion: Some("X1".to_string()),
                round: Some("R2".to_string()),
                ..params
            }))
            .await
            .unwrap();

        let Json(removed) = server
            .unexclude(Parameters(UnexcludeParams {
                reference: "https://doi.org/10.1000/Essay".to_string(),
            }))
            .await
            .unwrap();
        assert_eq!(removed.key, "10.1000/essay");
        assert_eq!(removed.reason, "not empirical");
        assert_eq!(removed.criterion.as_deref(), Some("X1"));
        assert_eq!(removed.round.as_deref(), Some("R2"));
        let essay = work(
            ProviderId::OpenAlex,
            "W2",
            "An Essay",
            Some("10.1000/essay"),
        );
        assert!(!LibraryIndex::load(&root).is_excluded(&essay));

        let error = server
            .unexclude(Parameters(UnexcludeParams {
                reference: "10.1000/essay".to_string(),
            }))
            .await
            .err()
            .expect("a paper that isn't excluded should be refused");
        assert!(error.contains("isn't excluded"), "{error}");

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn exclude_params_take_only_known_stages() {
        let params = |stage: &str| {
            serde_json::from_value::<ExcludeParams>(serde_json::json!({
                "reference": "10.1000/x",
                "reason": "r",
                "stage": stage,
            }))
        };
        assert!(matches!(
            ExclusionStage::from(params("screening").unwrap().stage),
            ExclusionStage::Screening
        ));
        assert!(matches!(
            ExclusionStage::from(params("full-text").unwrap().stage),
            ExclusionStage::FullText
        ));
        assert!(params("abstract").is_err());
    }

    #[tokio::test]
    async fn stats_counts_the_logged_search_and_the_exclusions() {
        let root = library_with_two_papers("stats");
        let server = PaxServer::new(root.clone(), Config::default());
        let page = SearchPage::new(
            vec![
                work(
                    ProviderId::OpenAlex,
                    "W1",
                    "On Computable Numbers",
                    Some("10.1112/plms/s2-42.1.230"),
                ),
                work(
                    ProviderId::OpenAlex,
                    "W2",
                    "An Essay",
                    Some("10.1000/essay"),
                ),
                work(
                    ProviderId::OpenAlex,
                    "W3",
                    "Undecided",
                    Some("10.1000/undecided"),
                ),
            ],
            Some(3),
            None,
        );
        let results: HashMap<ProviderId, Result<SearchPage, ProviderError>> =
            HashMap::from([(ProviderId::OpenAlex, Ok(page))]);
        let logged = pax_core::log_results(
            &root,
            pax_core::SearchCommand::Search,
            "computable",
            &SearchOptions::default().with_label("R1"),
            &results,
        );
        assert!(matches!(logged, LogOutcome::Written(_)));
        let Parameters(params) = exclusion("10.1000/essay", "not empirical", Some("An Essay"));
        server
            .exclude(Parameters(ExcludeParams {
                criterion: Some("X2".to_string()),
                ..params
            }))
            .await
            .unwrap();

        let Json(stats) = server.stats().await.unwrap();
        assert_eq!(stats.identified.databases.get("openalex"), Some(&3));
        assert_eq!(stats.identified.total, 3);
        assert_eq!(stats.records_screened, 3);
        assert_eq!(stats.duplicates_removed, 0);
        // The undecided paper awaits screening rather than counting as
        // passed: only the declared papers and the full-text exclusion did.
        assert_eq!(stats.awaiting_screening, 1);
        assert_eq!(stats.reports_sought, 3);
        // hewitt1973 was declared without a logged search finding it.
        assert_eq!(stats.decided_outside_log, 1);
        assert_eq!(stats.excluded_full_text.total, 1);
        assert_eq!(stats.excluded_full_text.reasons[0].reason, "not empirical");
        assert_eq!(stats.excluded_full_text.reasons[0].count, 1);
        assert!(stats.excluded_by_criterion.screening.is_empty());
        let full_text: Vec<_> = stats
            .excluded_by_criterion
            .full_text
            .iter()
            .map(|c| (c.criterion.as_deref(), c.count))
            .collect();
        assert_eq!(full_text, [(Some("X2"), 1)]);
        let by_label: Vec<_> = stats
            .by_label
            .iter()
            .map(|l| {
                (
                    l.label.as_deref(),
                    l.identified,
                    l.distinct,
                    l.new_distinct,
                    l.duplicates,
                )
            })
            .collect();
        assert_eq!(by_label, [(Some("R1"), 3, 3, 3, 0)]);

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn stats_counts_related_works_apart_from_citation_searching() {
        let root = library_with_two_papers("stats-related");
        let server = PaxServer::new(root.clone(), Config::default());
        let page = SearchPage::new(
            vec![
                work(ProviderId::SemanticScholar, "S1", "A", Some("10.1000/a")),
                work(ProviderId::SemanticScholar, "S2", "B", Some("10.1000/b")),
            ],
            None,
            None,
        );
        let results: HashMap<ProviderId, Result<SearchPage, ProviderError>> =
            HashMap::from([(ProviderId::SemanticScholar, Ok(page))]);
        let logged = pax_core::log_results(
            &root,
            pax_core::SearchCommand::Related,
            "openalex:W1",
            &SearchOptions::default(),
            &results,
        );
        assert!(matches!(logged, LogOutcome::Written(_)));

        let Json(stats) = server.stats().await.unwrap();
        assert_eq!(
            stats.identified.related_works.get("semanticscholar"),
            Some(&2)
        );
        assert!(stats.identified.citation_searching.is_empty());
        assert!(stats.identified.databases.is_empty());
        assert_eq!(stats.identified.total, 2);

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn stats_without_a_library_is_a_clear_error() {
        let root =
            std::env::temp_dir().join(format!("pax-mcp-test-no-library-{}", std::process::id()));
        let server = PaxServer::new(root, Config::default());
        let error = server.stats().await.err().expect("stats needs a library");
        assert!(error.contains("no pax library"), "{error}");
    }

    #[tokio::test]
    async fn add_refuses_a_paper_already_in_the_library() {
        let root = library_with_two_papers("add-duplicate");
        let server = PaxServer::new(root.clone(), Config::default());
        // A Crossref reference names its DOI, so pax-core refuses it before
        // resolving anything over the network.
        let error = server
            .add(Parameters(AddParams {
                reference: "crossref:10.1112/plms/s2-42.1.230".to_string(),
                label: None,
            }))
            .await
            .err()
            .expect("a declared paper should be refused");
        assert!(error.contains("already in the library"), "{error}");
        assert!(error.contains("turing1936"), "{error}");

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn add_refuses_an_excluded_paper_and_points_to_unexclude() {
        let root = library_with_two_papers("add-excluded");
        let server = PaxServer::new(root.clone(), Config::default());
        server
            .exclude(exclusion(
                "10.1000/essay",
                "not empirical",
                Some("An Essay"),
            ))
            .await
            .unwrap();
        // As above, the Crossref reference's DOI matches the exclusion before
        // anything is resolved.
        let error = server
            .add(Parameters(AddParams {
                reference: "crossref:10.1000/essay".to_string(),
                label: None,
            }))
            .await
            .err()
            .expect("an excluded paper should be refused");
        assert_eq!(
            error,
            "not added: crossref:10.1000/essay was excluded at full-text (\"not empirical\"); \
             call unexclude with \"10.1000/essay\" first to reverse that decision"
        );
        let Json(listed) = server
            .list(Parameters(ListParams::default()))
            .await
            .unwrap();
        let listed = listed.declared.unwrap();
        assert_eq!(listed.papers.len(), 2, "nothing was declared");

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn add_rejects_a_reference_that_is_not_a_provider_id() {
        let server = PaxServer::new(PathBuf::from("."), Config::default());
        let result = server
            .add(Parameters(AddParams {
                reference: "10.1145/359545.359563".to_string(),
                label: None,
            }))
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn add_refuses_a_blank_label() {
        let server = PaxServer::new(PathBuf::from("."), Config::default());
        let error = server
            .add(Parameters(AddParams {
                reference: "openalex:W2072794470".to_string(),
                label: Some("  ".to_string()),
            }))
            .await
            .err()
            .expect("a blank label should be refused");
        assert!(error.contains("`label` must not be blank"), "{error}");
    }

    #[tokio::test]
    async fn a_refused_labelled_add_logs_nothing() {
        let root = library_with_two_papers("add-label-refused");
        let server = PaxServer::new(root.clone(), Config::default());
        // As in add_refuses_a_paper_already_in_the_library, refused offline.
        let error = server
            .add(Parameters(AddParams {
                reference: "crossref:10.1112/plms/s2-42.1.230".to_string(),
                label: Some("seeds".to_string()),
            }))
            .await
            .err()
            .expect("a declared paper should be refused");
        assert!(error.contains("already in the library"), "{error}");
        assert!(!root.join("research/search-log.jsonl").exists());

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn stats_counts_papers_added_under_a_label_as_other_sources() {
        let root = library_with_two_papers("stats-manual");
        let server = PaxServer::new(root.clone(), Config::default());
        // The line `add { reference, label: "seeds" }` writes; the add itself
        // needs a provider, so the line is built by pax-core directly.
        let id: CandidateId = "crossref:10.1112/plms/s2-42.1.230".parse().unwrap();
        let entry = pax_core::search_log::manual_log_entry(
            &id,
            &work(
                ProviderId::Crossref,
                "10.1112/plms/s2-42.1.230",
                "On Computable Numbers",
                Some("10.1112/plms/s2-42.1.230"),
            ),
            "seeds",
            std::time::SystemTime::now(),
        );
        pax_core::SearchLog::for_library(&root)
            .unwrap()
            .append(&[entry])
            .unwrap();

        let Json(stats) = server.stats().await.unwrap();
        assert_eq!(stats.identified.other_sources.get("seeds"), Some(&1));
        assert!(stats.identified.databases.is_empty());
        assert_eq!(stats.identified.total, 1);
        let by_label: Vec<_> = stats
            .by_label
            .iter()
            .map(|l| (l.label.as_deref(), l.identified))
            .collect();
        assert_eq!(by_label, [(Some("seeds"), 1)]);
        // Only hewitt1973 is still decided outside the log.
        assert_eq!(stats.decided_outside_log, 1);

        fs::remove_dir_all(root).unwrap();
    }

    fn edit_of(reference: &str) -> EditParams {
        EditParams {
            reference: reference.to_string(),
            ..EditParams::default()
        }
    }

    #[tokio::test]
    async fn edit_round_trips_tags_and_notes_through_papers_nix() {
        let root = library_with_two_papers("edit");
        let server = PaxServer::new(root.clone(), Config::default());

        let Json(edited) = server
            .edit(Parameters(EditParams {
                add_tags: vec!["review-1".to_string(), "computability".to_string()],
                remove_tags: vec!["concurrency".to_string()],
                notes: Some("included: meets criterion C2".to_string()),
                ..edit_of("hewitt1973")
            }))
            .await
            .unwrap();
        assert_eq!(edited.tags, ["review-1", "computability"]);
        assert_eq!(
            edited.notes.as_deref(),
            Some("included: meets criterion C2")
        );

        let library = Library::load(&pax_core::nix::papers_path(&root)).unwrap();
        let reloaded = DeclaredPaper::from(library.find("hewitt1973").unwrap());
        assert_eq!(reloaded.tags, edited.tags);
        assert_eq!(reloaded.notes, edited.notes);
        let Json(tagged) = server
            .list(Parameters(ListParams {
                tag: Some("review-1".to_string()),
                ..ListParams::default()
            }))
            .await
            .unwrap();
        assert_eq!(tagged.declared.unwrap().papers.len(), 1);

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn edit_refuses_an_empty_edit_blank_tags_and_unknown_keys() {
        let root = library_with_two_papers("edit-refused");
        let server = PaxServer::new(root.clone(), Config::default());
        let papers = pax_core::nix::papers_path(&root);
        let before = fs::read_to_string(&papers).unwrap();

        let empty = server
            .edit(Parameters(edit_of("hewitt1973")))
            .await
            .err()
            .expect("an empty edit should be refused");
        assert!(empty.contains("add_tags"), "{empty}");
        let blank = server
            .edit(Parameters(EditParams {
                add_tags: vec![" ".to_string()],
                ..edit_of("hewitt1973")
            }))
            .await;
        assert!(blank.is_err());
        let unknown = server
            .edit(Parameters(EditParams {
                notes: Some("n".to_string()),
                ..edit_of("dijkstra1968")
            }))
            .await
            .err()
            .expect("an unknown citation key should be refused");
        assert!(unknown.contains("dijkstra1968"), "{unknown}");
        assert_eq!(fs::read_to_string(&papers).unwrap(), before);

        fs::remove_dir_all(root).unwrap();
    }

    fn export_of(keys: Option<&[&str]>, tag: Option<&str>) -> Parameters<ExportParams> {
        Parameters(ExportParams {
            format: ExportFormat::Bibtex,
            keys: keys.map(|keys| keys.iter().map(|k| k.to_string()).collect()),
            tag: tag.map(str::to_string),
            dir: None,
        })
    }

    /// A `bibtex` export's result.
    fn bibtex(output: Result<Json<ExportOutput>, String>) -> Exported {
        let Json(output) = output.unwrap();
        assert!(output.pdfs.is_none());
        output.text.expect("a bibtex export")
    }

    #[tokio::test]
    async fn export_renders_the_library_or_a_selection_as_bibtex() {
        let root = library_with_two_papers("export");
        let server = PaxServer::new(root.clone(), Config::default());

        let all = bibtex(server.export(export_of(None, None)).await);
        assert_eq!(all.citation_keys, ["turing1936", "hewitt1973"]);
        assert!(
            all.bibtex.contains("@article{turing1936,"),
            "{}",
            all.bibtex
        );
        assert!(
            all.bibtex.contains("@article{hewitt1973,"),
            "{}",
            all.bibtex
        );
        assert!(
            all.bibtex.contains("doi = {10.1112/plms/s2-42.1.230}"),
            "{}",
            all.bibtex
        );

        let by_key = bibtex(server.export(export_of(Some(&["hewitt1973"]), None)).await);
        assert_eq!(by_key.citation_keys, ["hewitt1973"]);
        assert!(!by_key.bibtex.contains("turing1936"));
        assert!(
            by_key
                .bibtex
                .contains("title = {A Universal Modular Actor Formalism}")
        );

        let by_tag = bibtex(server.export(export_of(None, Some("computability"))).await);
        assert_eq!(by_tag.citation_keys, ["turing1936"]);

        let none = bibtex(
            server
                .export(export_of(Some(&["hewitt1973"]), Some("computability")))
                .await,
        );
        assert!(none.citation_keys.is_empty() && none.bibtex.is_empty());

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn export_refuses_unknown_keys_and_formats() {
        let root = library_with_two_papers("export-refused");
        let server = PaxServer::new(root.clone(), Config::default());
        let error = server
            .export(export_of(Some(&["turing1936", "dijkstra1968"]), None))
            .await
            .err()
            .expect("an unknown key should be refused");
        assert!(
            error.contains("dijkstra1968") && !error.contains("turing1936"),
            "{error}"
        );

        let ris = serde_json::from_value::<ExportParams>(serde_json::json!({ "format": "ris" }));
        assert!(ris.is_err());

        fs::remove_dir_all(root).unwrap();
    }

    fn pdfs_to(dir: Option<&str>) -> Parameters<ExportParams> {
        Parameters(ExportParams {
            format: ExportFormat::Pdfs,
            keys: None,
            tag: None,
            dir: dir.map(str::to_string),
        })
    }

    /// A `pdfs` export's result, with no warnings.
    fn pdfs(output: Result<Json<ExportOutput>, String>) -> ExportedPdfs {
        let (exported, warnings) = pdfs_warned(output);
        assert!(warnings.is_empty(), "{warnings:?}");
        exported
    }

    /// A `pdfs` export's result and its warnings.
    fn pdfs_warned(output: Result<Json<ExportOutput>, String>) -> (ExportedPdfs, Vec<String>) {
        let Json(output) = output.unwrap();
        assert!(output.text.is_none());
        (output.pdfs.expect("a pdfs export"), output.warnings)
    }

    /// Clears the fixture's hashes, so it has no local artifact (turing1936's
    /// hash has no `source_url`) and pax-core runs no `nix` for it.
    fn without_hashes(root: &Path) {
        let papers = pax_core::nix::papers_path(root);
        let text = fs::read_to_string(&papers)
            .unwrap()
            .replace(r#"hash = "sha256-AAAA";"#, "hash = null;");
        fs::write(&papers, text).unwrap();
    }

    #[tokio::test]
    async fn export_pdfs_refuses_a_blank_dir_and_selects_no_papers() {
        let root = library_with_two_papers("export-pdfs-refused");
        let server = PaxServer::new(root.clone(), Config::default());

        for dir in [Some(""), Some("  ")] {
            let error = server.export(pdfs_to(dir)).await.err().unwrap();
            assert!(error.contains("`dir`"), "{error}");
        }
        let mut selected = pdfs_to(Some("backup"));
        selected.0.tag = Some("computability".to_string());
        let error = server.export(selected).await.err().unwrap();
        assert!(error.contains("`tag`"), "{error}");
        let mut bibtex_to_dir = export_of(None, None);
        bibtex_to_dir.0.dir = Some("backup".to_string());
        let error = server.export(bibtex_to_dir).await.err().unwrap();
        assert!(error.contains("`dir`"), "{error}");
        assert!(!root.join("backup").exists());
        assert!(!pax_core::default_backup_dir(&root).exists());

        let parsed: ExportParams =
            serde_json::from_value(serde_json::json!({ "format": "pdfs", "dir": "backup" }))
                .unwrap();
        assert!(matches!(parsed.format, ExportFormat::Pdfs));
        let parsed: ExportParams =
            serde_json::from_value(serde_json::json!({ "format": "pdfs" })).unwrap();
        assert!(parsed.dir.is_none());

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn export_pdfs_leaves_out_papers_nix_can_download() {
        let root = library_with_two_papers("export-pdfs-none");
        without_hashes(&root);
        let server = PaxServer::new(root.clone(), Config::default());

        let exported = pdfs(server.export(pdfs_to(Some("backup/nested"))).await);
        let dir = root.join("backup/nested");
        assert_eq!(exported.directory, dir.display().to_string());
        assert_eq!(
            exported.manifest,
            dir.join("manifest.json").display().to_string()
        );
        assert!(exported.papers.is_empty());
        assert_eq!(
            (exported.exported, exported.missing, exported.failed),
            (0, 0, 0)
        );
        assert_eq!(
            fs::read_to_string(dir.join("manifest.json")).unwrap(),
            "{}\n"
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn export_pdfs_defaults_to_the_ignored_research_pdfs() {
        let root = library_with_two_papers("export-pdfs-default");
        without_hashes(&root);
        let server = PaxServer::new(root.clone(), Config::default());

        let exported = pdfs(server.export(pdfs_to(None)).await);
        let dir = pax_core::default_backup_dir(&root);
        assert_eq!(dir, root.join("research/.pdfs"));
        assert_eq!(exported.directory, dir.display().to_string());
        assert_eq!(
            fs::read_to_string(dir.join("manifest.json")).unwrap(),
            "{}\n"
        );
        let gitignore = fs::read_to_string(root.join("research/.gitignore")).unwrap();
        assert!(gitignore.lines().any(|l| l == ".pdfs/"), "{gitignore}");

        fs::remove_dir_all(root).unwrap();
    }

    /// Whether `git` can run here; the tracked-folder warning needs it.
    fn git_available() -> bool {
        std::process::Command::new("git")
            .arg("--version")
            .output()
            .is_ok_and(|output| output.status.success())
    }

    #[tokio::test]
    async fn export_pdfs_warns_about_a_dir_git_would_commit() {
        if !git_available() {
            eprintln!("skipped: git isn't available");
            return;
        }
        let root = library_with_two_papers("export-pdfs-staged");
        without_hashes(&root);
        // Cleared so a git hook's GIT_DIR can't point `git init` at this repo.
        let init = std::process::Command::new("git")
            .arg("init")
            .arg("-q")
            .arg(&root)
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .env_remove("GIT_INDEX_FILE")
            .output()
            .unwrap();
        assert!(init.status.success());
        let server = PaxServer::new(root.clone(), Config::default());

        let (exported, warnings) = pdfs_warned(server.export(pdfs_to(Some("research/pdfs"))).await);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        let warning = &warnings[0];
        assert!(
            warning.contains(&exported.directory)
                && warning.contains("committed")
                && warning.contains("research/.pdfs/"),
            "{warning}"
        );
        let Json(output) = server.export(pdfs_to(Some("research/pdfs"))).await.unwrap();
        let json = serde_json::to_value(&output).unwrap();
        assert_eq!(json["warnings"][0], serde_json::json!(warning));

        // The default and a directory outside research/ aren't committed.
        pdfs(server.export(pdfs_to(None)).await);
        let Json(output) = server.export(pdfs_to(Some("backup"))).await.unwrap();
        assert!(output.warnings.is_empty(), "{:?}", output.warnings);
        let json = serde_json::to_value(&output).unwrap();
        assert!(json.get("warnings").is_none(), "{json}");

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn sync_from_backup_takes_the_default_dir_alone() {
        let root = library_with_two_papers("sync-from-backup");
        without_hashes(&root);
        let server = PaxServer::new(root.clone(), Config::default());

        let both = server
            .sync(Parameters(SyncParams {
                from: Some("backup".to_string()),
                from_backup: true,
            }))
            .await
            .err()
            .unwrap();
        assert!(both.contains("not both"), "{both}");
        let none_yet = server.sync(sync_from_backup()).await.err().unwrap();
        assert!(
            none_yet.contains("research/.pdfs") && none_yet.contains("`export`"),
            "{none_yet}"
        );

        // No paper has a hash or a source URL, so nothing runs `nix`.
        pdfs(server.export(pdfs_to(None)).await);
        let Json(synced) = server.sync(sync_from_backup()).await.unwrap();
        let keys: Vec<_> = synced.papers.iter().map(|p| &p.citation_key).collect();
        assert_eq!(keys, ["turing1936", "hewitt1973"]);

        let parsed: SyncParams =
            serde_json::from_value(serde_json::json!({ "from_backup": true })).unwrap();
        assert!(parsed.from_backup && parsed.from.is_none());
        let parsed: SyncParams =
            serde_json::from_value(serde_json::json!({ "from": "backup" })).unwrap();
        assert!(!parsed.from_backup);

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn sync_refuses_a_blank_or_missing_backup_directory() {
        let root = library_with_two_papers("sync-refused");
        without_hashes(&root);
        let server = PaxServer::new(root.clone(), Config::default());
        let papers = pax_core::nix::papers_path(&root);
        let before = fs::read_to_string(&papers).unwrap();

        let blank = server.sync(sync_from(Some(" "))).await.err().unwrap();
        assert!(blank.contains("`from`"), "{blank}");
        let missing = server
            .sync(sync_from(Some("no-such-dir")))
            .await
            .err()
            .unwrap();
        assert!(
            missing.contains(&root.join("no-such-dir").display().to_string())
                && missing.contains("not a directory"),
            "{missing}"
        );
        assert_eq!(fs::read_to_string(&papers).unwrap(), before);

        let elsewhere = PaxServer::new(root.join("research"), Config::default());
        let no_library = elsewhere.sync(sync_from(None)).await.err().unwrap();
        assert!(no_library.contains("no pax library"), "{no_library}");

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn sync_reports_every_paper_with_or_without_a_backup() {
        // No paper has a hash or a source URL, so nothing runs `nix`.
        let root = library_with_two_papers("sync");
        without_hashes(&root);
        fs::create_dir_all(root.join("backup")).unwrap();
        let server = PaxServer::new(root.clone(), Config::default());

        for from in [None, Some("backup")] {
            let Json(synced) = server.sync(sync_from(from)).await.unwrap();
            let keys: Vec<_> = synced.papers.iter().map(|p| &p.citation_key).collect();
            assert_eq!(keys, ["turing1936", "hewitt1973"]);
            for paper in &synced.papers {
                assert_eq!(paper.status, FetchState::NoSourceUrl);
                let reason = paper.reason.as_deref().unwrap();
                assert!(reason.contains("`file`"), "{reason}");
            }
        }

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn sync_results_say_why_a_backup_was_skipped() {
        let report = |citation_key: &str, result| SyncReport {
            citation_key: citation_key.to_string(),
            result,
        };
        let synced = sync_results(&[
            report(
                "turing1936",
                Ok(FetchOutcome::Fetched {
                    hash: "sha256-ok".to_string(),
                }),
            ),
            report(
                "hewitt1973",
                Err(PaxError::BackupMismatch {
                    citation_key: "hewitt1973".to_string(),
                    file: PathBuf::from("/backup/hewitt1973.pdf"),
                    recorded: "sha256-old".to_string(),
                    actual: "sha256-new".to_string(),
                }),
            ),
            report(
                "knuth1974",
                Err(PaxError::LocalArtifactMissing("knuth1974".to_string())),
            ),
        ]);
        let statuses: Vec<_> = synced.papers.iter().map(|p| p.status).collect();
        assert_eq!(
            statuses,
            [
                FetchState::Fetched,
                FetchState::HashMismatch,
                FetchState::FileNeeded
            ]
        );
        assert!(synced.papers[0].reason.is_none());
        let skipped = synced.papers[1].reason.as_deref().unwrap();
        assert!(
            [
                "/backup/hewitt1973.pdf",
                "sha256-old",
                "sha256-new",
                "skipped"
            ]
            .iter()
            .all(|part| skipped.contains(part)),
            "{skipped}"
        );
        assert!(!skipped.contains("force"), "{skipped}");
        let needed = synced.papers[2].reason.as_deref().unwrap();
        assert!(
            needed.contains("`file`") && needed.contains("`from`") && !needed.contains("--"),
            "{needed}"
        );
    }

    #[tokio::test]
    async fn check_reports_papers_without_a_hash_as_not_fetched() {
        // Neither paper has a hash, so pax-core runs no `nix` at all. (A hash
        // without a `source_url`, as turing1936's in the fixture, is a local
        // artifact, which pax-core looks up in the Nix store.)
        let root = library_with_two_papers("check");
        let papers = root.join("research/papers.nix");
        let without_hash = fs::read_to_string(&papers)
            .unwrap()
            .replace(r#"hash = "sha256-AAAA";"#, "hash = null;");
        fs::write(&papers, without_hash).unwrap();
        let server = PaxServer::new(root.clone(), Config::default());

        let Json(checked) = server.check().await.unwrap();
        let statuses: Vec<_> = checked
            .papers
            .iter()
            .map(|p| (p.citation_key.as_str(), p.status))
            .collect();
        assert_eq!(
            statuses,
            [
                ("turing1936", CheckState::NotFetched),
                ("hewitt1973", CheckState::NotFetched),
            ]
        );
        assert_eq!((checked.checked, checked.not_fetched), (2, 2));
        assert_eq!(
            (
                checked.reproducible,
                checked.mismatched,
                checked.errored,
                checked.missing
            ),
            (0, 0, 0, 0)
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn fetch_reports_a_paper_without_a_source_url_without_running_nix() {
        let root = library_with_two_papers("fetch");
        let server = PaxServer::new(root.clone(), Config::default());
        let papers = pax_core::nix::papers_path(&root);
        let before = fs::read_to_string(&papers).unwrap();

        let Json(result) = server
            .fetch(Parameters(FetchParams {
                reference: "hewitt1973".to_string(),
                file: None,
                force: false,
            }))
            .await
            .unwrap();
        assert_eq!(result.status, FetchState::NoSourceUrl);
        assert!(result.store_path.is_none() && result.hash.is_none());
        assert!(result.reason.unwrap().contains("source"));
        assert_eq!(fs::read_to_string(&papers).unwrap(), before);

        let unknown = server
            .fetch(Parameters(FetchParams {
                reference: "dijkstra1968".to_string(),
                file: None,
                force: false,
            }))
            .await
            .err()
            .expect("an unknown citation key should be refused");
        assert!(unknown.contains("dijkstra1968"), "{unknown}");

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn fetch_results_say_what_happened_and_what_to_do() {
        let built = fetch_result(
            "turing1936",
            Ok((
                FetchOutcome::Fetched {
                    hash: "sha256-new".to_string(),
                },
                Ok(ResolvedArtifact::Path(PathBuf::from("/nix/store/x-turing"))),
            )),
        )
        .unwrap();
        assert_eq!(built.status, FetchState::Fetched);
        assert_eq!(built.hash.as_deref(), Some("sha256-new"));
        assert_eq!(built.store_path.as_deref(), Some("/nix/store/x-turing"));
        assert!(built.reason.is_none());

        let mismatch = fetch_result(
            "turing1936",
            Ok((
                FetchOutcome::AlreadyFetched {
                    hash: "sha256-old".to_string(),
                },
                Err(PaxError::Build(
                    "error: hash mismatch in fixed-output derivation".to_string(),
                )),
            )),
        )
        .unwrap();
        assert_eq!(mismatch.status, FetchState::HashMismatch);
        assert_eq!(mismatch.hash.as_deref(), Some("sha256-old"));
        assert!(mismatch.store_path.is_none());

        let untracked = fetch_result(
            "turing1936",
            Ok((
                FetchOutcome::AlreadyFetched {
                    hash: "sha256-old".to_string(),
                },
                Err(PaxError::Build(
                    "error: path 'research/papers.nix' is not tracked by Git".to_string(),
                )),
            )),
        )
        .unwrap();
        assert_eq!(untracked.status, FetchState::BuildFailed);
        assert!(untracked.reason.unwrap().contains("git add"));

        let failed = fetch_result(
            "turing1936",
            Err(PaxError::Fetch("`nix` timed out after 60s".to_string())),
        )
        .unwrap();
        assert_eq!(failed.status, FetchState::FetchFailed);
        assert!(failed.hash.is_none());
        assert!(failed.reason.unwrap().contains("timed out"));

        // A paper added from a local file that this Nix store no longer has.
        let missing = PaxError::LocalArtifactMissing("turing1936".to_string());
        let file_needed = fetch_result("turing1936", Err(missing)).unwrap();
        assert_eq!(file_needed.status, FetchState::FileNeeded);
        let reason = file_needed.reason.unwrap();
        assert!(
            reason.contains("`file`") && reason.contains("`sync`") && reason.contains("`from`"),
            "{reason}"
        );
        let unbuilt = fetch_result(
            "turing1936",
            Ok((
                FetchOutcome::AlreadyFetched {
                    hash: "sha256-old".to_string(),
                },
                Err(PaxError::LocalArtifactMissing("turing1936".to_string())),
            )),
        )
        .unwrap();
        assert_eq!(unbuilt.status, FetchState::FileNeeded);
        assert_eq!(unbuilt.hash.as_deref(), Some("sha256-old"));

        assert!(fetch_result("x", Err(PaxError::NoSuchPaper("x".to_string()))).is_err());
    }

    #[test]
    fn file_fetch_results_point_at_file_and_force() {
        let no_source = fetch_result(
            "hewitt1973",
            Err(PaxError::NoSourceUrl("hewitt1973".to_string())),
        )
        .unwrap();
        assert_eq!(no_source.status, FetchState::NoSourceUrl);
        let reason = no_source.reason.unwrap();
        assert!(
            reason.contains("`file`") && reason.contains("--source-url"),
            "{reason}"
        );

        let mismatch = fetch_result(
            "turing1936",
            Err(PaxError::HashMismatch {
                citation_key: "turing1936".to_string(),
                recorded: "sha256-old".to_string(),
                actual: "sha256-new".to_string(),
            }),
        )
        .unwrap();
        assert_eq!(mismatch.status, FetchState::HashMismatch);
        assert!(mismatch.hash.is_none());
        let reason = mismatch.reason.unwrap();
        assert!(
            reason.contains("sha256-old") && reason.contains("sha256-new"),
            "{reason}"
        );
        assert!(reason.contains("`force: true`"), "{reason}");

        let not_a_file = fetch_result("x", Err(PaxError::NotAFile(PathBuf::from("/tmp"))));
        assert!(not_a_file.unwrap_err().contains("not a file"));
    }

    #[tokio::test]
    async fn fetch_refuses_a_bad_file_before_running_nix() {
        let root = library_with_two_papers("fetch-bad-file");
        let server = PaxServer::new(root.clone(), Config::default());
        let papers = pax_core::nix::papers_path(&root);
        let before = fs::read_to_string(&papers).unwrap();
        let fetch = |file: Option<&str>, force| {
            server.fetch(Parameters(FetchParams {
                reference: "hewitt1973".to_string(),
                file: file.map(str::to_string),
                force,
            }))
        };

        let missing = fetch(Some("missing.pdf"), false).await.err().unwrap();
        assert!(
            missing.contains(&root.join("missing.pdf").display().to_string()),
            "{missing}"
        );
        // A directory opens, so pax-core is the one refusing it.
        let directory = fetch(Some("research"), false).await.err().unwrap();
        assert!(directory.contains("not a file"), "{directory}");
        let force = fetch(None, true).await.err().unwrap();
        assert!(force.contains("`file`"), "{force}");
        assert_eq!(fs::read_to_string(&papers).unwrap(), before);

        fs::remove_dir_all(root).unwrap();
    }

    /// Whether `nix` can run here; tests that add files to the Nix store
    /// are skipped without it.
    fn nix_available() -> bool {
        std::process::Command::new("nix")
            .arg("--version")
            .output()
            .is_ok_and(|output| output.status.success())
    }

    /// A test library whose papers a test adds to the Nix store. On drop it
    /// removes the library, and with it the GC roots pax-core registered in
    /// `research/.gcroots`, then deletes each store path in `store_paths`
    /// (only those: never a whole-store garbage collection). Each test adds
    /// files with its own content, so no other test shares a path.
    struct StoreScratch {
        root: PathBuf,
        store_paths: Vec<String>,
    }

    impl StoreScratch {
        fn new(root: PathBuf) -> Self {
            StoreScratch {
                root,
                store_paths: Vec::new(),
            }
        }

        /// Records a path a tool reported, for deletion.
        fn track(&mut self, store_path: Option<&str>) {
            if let Some(path) = store_path
                && !self.store_paths.iter().any(|p| p == path)
            {
                self.store_paths.push(path.to_string());
            }
        }
    }

    impl Drop for StoreScratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
            for path in &self.store_paths {
                let deleted = std::process::Command::new("nix-store")
                    .args(["--delete", path])
                    .output();
                if !deleted.is_ok_and(|output| output.status.success()) {
                    eprintln!("couldn't delete {path} from the Nix store");
                }
            }
        }
    }

    #[tokio::test]
    async fn fetch_with_a_file_records_its_hash_and_force_replaces_one() {
        if !nix_available() {
            eprintln!("skipped: nix isn't available");
            return;
        }
        let mut scratch = StoreScratch::new(library_with_two_papers("fetch-file"));
        let root = scratch.root.clone();
        let server = PaxServer::new(root.clone(), Config::default());
        let papers = pax_core::nix::papers_path(&root);
        let pdf = root.join("paper.pdf");
        fs::write(&pdf, "%PDF-1.4 pax-ai fetch test\n").unwrap();
        let fetch = |reference: &str, force| {
            server.fetch(Parameters(FetchParams {
                reference: reference.to_string(),
                file: Some("paper.pdf".to_string()),
                force,
            }))
        };

        let Json(added) = fetch("hewitt1973", false).await.unwrap();
        scratch.track(added.store_path.as_deref());
        assert_eq!(added.status, FetchState::Fetched, "{:?}", added.reason);
        let hash = added.hash.expect("the file's hash");
        assert!(fs::read_to_string(&papers).unwrap().contains(&hash));
        assert!(
            added
                .store_path
                .is_some_and(|path| path.starts_with("/nix/store/"))
        );

        // turing1936 already records another hash.
        let before = fs::read_to_string(&papers).unwrap();
        let Json(mismatch) = fetch("turing1936", false).await.unwrap();
        assert_eq!(mismatch.status, FetchState::HashMismatch);
        assert!(mismatch.reason.unwrap().contains("force"));
        assert_eq!(fs::read_to_string(&papers).unwrap(), before);

        let Json(forced) = fetch("turing1936", true).await.unwrap();
        scratch.track(forced.store_path.as_deref());
        assert_eq!(forced.status, FetchState::Fetched, "{:?}", forced.reason);
        assert_eq!(forced.hash.as_deref(), Some(hash.as_str()));
        assert!(!fs::read_to_string(&papers).unwrap().contains("sha256-AAAA"));
    }

    /// Adds `content` as `reference`'s PDF through `fetch`'s `file`, tracking
    /// its store path in `scratch`, and returns its hash.
    async fn add_local_pdf(
        server: &PaxServer,
        scratch: &mut StoreScratch,
        reference: &str,
        content: &str,
    ) -> String {
        let file = format!("{reference}-source.pdf");
        fs::write(scratch.root.join(&file), content).unwrap();
        let Json(added) = server
            .fetch(Parameters(FetchParams {
                reference: reference.to_string(),
                file: Some(file),
                force: false,
            }))
            .await
            .unwrap();
        scratch.track(added.store_path.as_deref());
        assert_eq!(added.status, FetchState::Fetched, "{:?}", added.reason);
        added.hash.expect("the file's hash")
    }

    #[tokio::test]
    async fn export_pdfs_copies_local_artifacts_and_writes_a_manifest() {
        if !nix_available() {
            eprintln!("skipped: nix isn't available");
            return;
        }
        let mut scratch = StoreScratch::new(library_with_two_papers("export-pdfs"));
        let root = scratch.root.clone();
        without_hashes(&root);
        let server = PaxServer::new(root.clone(), Config::default());
        let content = "%PDF-1.4 pax-ai export test\n";
        let hash = add_local_pdf(&server, &mut scratch, "hewitt1973", content).await;

        let dir = root.join("backup");
        let exported = pdfs(server.export(pdfs_to(Some(dir.to_str().unwrap()))).await);
        assert_eq!(
            (exported.exported, exported.missing, exported.failed),
            (1, 0, 0)
        );
        let copy = &exported.papers[0];
        assert_eq!(copy.citation_key, "hewitt1973");
        assert_eq!(copy.status, PdfCopyState::Exported);
        assert_eq!(copy.hash, hash);
        let file = dir.join("hewitt1973.pdf");
        assert_eq!(copy.file.as_deref(), Some(file.to_str().unwrap()));
        assert_eq!(fs::read_to_string(&file).unwrap(), content);
        let manifest: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest, serde_json::json!({ "hewitt1973": hash }));
    }

    /// Removes `citation_key`'s GC root and its PDF from the Nix store, as
    /// on a machine that never had it.
    fn drop_from_store(root: &Path, citation_key: &str, store_path: &str) {
        fs::remove_file(root.join("research/.gcroots").join(citation_key)).unwrap();
        let deleted = std::process::Command::new("nix-store")
            .args(["--delete", store_path])
            .output()
            .unwrap();
        assert!(
            deleted.status.success(),
            "{}",
            String::from_utf8_lossy(&deleted.stderr)
        );
    }

    fn sync_from(from: Option<&str>) -> Parameters<SyncParams> {
        Parameters(SyncParams {
            from: from.map(str::to_string),
            from_backup: false,
        })
    }

    fn sync_from_backup() -> Parameters<SyncParams> {
        Parameters(SyncParams {
            from: None,
            from_backup: true,
        })
    }

    #[tokio::test]
    async fn the_default_backup_dir_round_trips_through_export_and_sync() {
        if !nix_available() {
            eprintln!("skipped: nix isn't available");
            return;
        }
        let mut scratch = StoreScratch::new(library_with_two_papers("default-backup"));
        let root = scratch.root.clone();
        without_hashes(&root);
        let server = PaxServer::new(root.clone(), Config::default());
        let content = "%PDF-1.4 pax-ai default backup test\n";
        let hash = add_local_pdf(&server, &mut scratch, "hewitt1973", content).await;
        let store_path = scratch.store_paths[0].clone();

        let exported = pdfs(server.export(pdfs_to(None)).await);
        assert_eq!(exported.exported, 1);
        let file = pax_core::default_backup_dir(&root).join("hewitt1973.pdf");
        assert_eq!(exported.papers[0].file.as_deref(), file.to_str());
        assert_eq!(fs::read_to_string(&file).unwrap(), content);

        drop_from_store(&root, "hewitt1973", &store_path);
        let Json(restored) = server.sync(sync_from_backup()).await.unwrap();
        let hewitt = restored
            .papers
            .iter()
            .find(|p| p.citation_key == "hewitt1973")
            .unwrap();
        assert_eq!(hewitt.status, FetchState::Fetched, "{:?}", hewitt.reason);
        assert_eq!(hewitt.hash.as_deref(), Some(hash.as_str()));
        assert!(Path::new(&store_path).exists());
    }

    #[tokio::test]
    async fn sync_from_restores_a_backup_and_skips_a_mismatching_file() {
        if !nix_available() {
            eprintln!("skipped: nix isn't available");
            return;
        }
        let mut scratch = StoreScratch::new(library_with_two_papers("sync-from"));
        let root = scratch.root.clone();
        without_hashes(&root);
        let server = PaxServer::new(root.clone(), Config::default());
        let papers = pax_core::nix::papers_path(&root);
        let hash = add_local_pdf(
            &server,
            &mut scratch,
            "hewitt1973",
            "%PDF-1.4 pax-ai sync test\n",
        )
        .await;
        let store_path = scratch.store_paths[0].clone();
        let exported = pdfs(server.export(pdfs_to(Some("backup"))).await);
        assert_eq!(exported.exported, 1);

        // On a machine without the PDF, check and export point at the backup.
        // (pax-core rewrites papers.nix in key order when it records a hash.)
        drop_from_store(&root, "hewitt1973", &store_path);
        let Json(checked) = server.check().await.unwrap();
        let missing = checked
            .papers
            .iter()
            .find(|p| p.citation_key == "hewitt1973")
            .unwrap();
        assert_eq!(missing.status, CheckState::Missing);
        let reason = missing.error.as_deref().unwrap();
        assert!(
            reason.contains("`file`") && reason.contains("`from`"),
            "{reason}"
        );
        let again = pdfs(server.export(pdfs_to(Some("backup-again"))).await);
        assert_eq!((again.exported, again.missing), (0, 1));
        assert_eq!(again.papers[0].status, PdfCopyState::Missing);
        assert!(again.papers[0].error.as_deref().unwrap().contains("`from`"));

        let by_key = |results: SyncResults| -> HashMap<String, FetchResult> {
            results
                .papers
                .into_iter()
                .map(|p| (p.citation_key.clone(), p))
                .collect()
        };
        let Json(restored) = server.sync(sync_from(Some("backup"))).await.unwrap();
        let restored = by_key(restored);
        assert_eq!(restored.len(), 2);
        assert_eq!(restored["turing1936"].status, FetchState::NoSourceUrl);
        let hewitt = &restored["hewitt1973"];
        assert_eq!(hewitt.status, FetchState::Fetched, "{:?}", hewitt.reason);
        assert_eq!(hewitt.hash.as_deref(), Some(hash.as_str()));
        assert!(Path::new(&store_path).exists());
        let Json(checked) = server.check().await.unwrap();
        assert_eq!(checked.missing, 0);

        // A backup file with other content is skipped, never forced.
        let before = fs::read_to_string(&papers).unwrap();
        fs::write(root.join("backup/hewitt1973.pdf"), "%PDF-1.4 another\n").unwrap();
        let Json(skipped) = server.sync(sync_from(Some("backup"))).await.unwrap();
        let mismatch = &by_key(skipped)["hewitt1973"];
        assert_eq!(mismatch.status, FetchState::HashMismatch);
        let reason = mismatch.reason.as_deref().unwrap();
        assert!(
            reason.contains("skipped") && reason.contains(&hash),
            "{reason}"
        );
        assert_eq!(fs::read_to_string(&papers).unwrap(), before);
    }

    #[tokio::test]
    async fn show_takes_exactly_one_of_reference_or_doi() {
        let server = PaxServer::new(PathBuf::from("."), Config::default());
        let both_or_neither = [
            ShowParams::default(),
            ShowParams {
                reference: Some("openalex:W2072794470".to_string()),
                doi: Some("10.1145/359545.359563".to_string()),
            },
        ];
        for params in both_or_neither {
            let error = server
                .show(Parameters(params))
                .await
                .err()
                .expect("both or neither should be refused");
            assert!(error.contains("exactly one"), "{error}");
        }

        let not_a_doi = server
            .show(Parameters(ShowParams {
                doi: Some("openalex:W2072794470".to_string()),
                ..ShowParams::default()
            }))
            .await
            .err()
            .expect("a provider reference isn't a DOI");
        assert!(not_a_doi.contains("not a DOI"), "{not_a_doi}");
    }

    #[test]
    fn doi_params_are_normalized() {
        assert_eq!(
            doi_param(" https://doi.org/10.1112/plms/s2-42.1.230 ").unwrap(),
            "10.1112/plms/s2-42.1.230"
        );
        for invalid in ["", "10.1112", "10.1112/", "turing1936"] {
            assert!(doi_param(invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn doi_records_merge_providers_and_flag_the_library() {
        let root = library_with_two_papers("doi-records");
        let index = LibraryIndex::load(&root);
        let turing = "10.1112/plms/s2-42.1.230";
        let results: HashMap<ProviderId, Result<Vec<CandidateWork>, ProviderError>> =
            HashMap::from([
                (
                    ProviderId::OpenAlex,
                    Ok(vec![work(
                        ProviderId::OpenAlex,
                        "W1",
                        "On Computable Numbers",
                        Some(turing),
                    )]),
                ),
                (
                    ProviderId::Crossref,
                    Ok(vec![work(
                        ProviderId::Crossref,
                        turing,
                        "On Computable Numbers",
                        Some(turing),
                    )]),
                ),
                (ProviderId::SemanticScholar, Err(ProviderError::NotFound)),
            ]);

        let out = DoiRecords::new(turing, &results, &index);

        let lookups: Vec<_> = out
            .providers
            .iter()
            .map(|p| (p.provider.as_str(), p.found, p.error.is_some()))
            .collect();
        assert_eq!(
            lookups,
            [
                ("openalex", true, false),
                ("crossref", true, false),
                ("semanticscholar", false, false),
            ]
        );
        assert_eq!(out.results.len(), 1);
        let hit = &out.results[0];
        assert_eq!(hit.providers, ["openalex", "crossref"]);
        assert!(hit.work.in_library && !hit.work.excluded);

        let failed: HashMap<ProviderId, Result<Vec<CandidateWork>, ProviderError>> =
            HashMap::from([(
                ProviderId::OpenAlex,
                Err(ProviderError::Request("timed out".to_string())),
            )]);
        let out = DoiRecords::new(turing, &failed, &index);
        assert!(out.results.is_empty());
        assert!(
            out.providers[0]
                .error
                .as_deref()
                .unwrap()
                .contains("timed out")
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn show_output_is_pax_cores_shown_or_a_dois_records() {
        let root = library_with_two_papers("show-output");
        let library = Library::load(&pax_core::nix::papers_path(&root)).unwrap();
        let index = LibraryIndex::load(&root);
        let hewitt = library.find("hewitt1973").unwrap().clone();

        let reference = ShowOutput {
            shown: Some(Shown::new(&pax_core::ShowResult::Declared(hewitt), &index)),
            records: None,
        };
        let json = serde_json::to_value(&reference).unwrap();
        assert_eq!(json["declared"]["citation_key"], "hewitt1973");
        for absent in ["excluded", "candidate", "records"] {
            assert!(json[absent].is_null(), "{absent}: {json}");
        }

        let doi = ShowOutput {
            shown: None,
            records: Some(DoiRecords::new("10.1000/x", &HashMap::new(), &index)),
        };
        let json = serde_json::to_value(&doi).unwrap();
        assert_eq!(json["records"]["doi"], "10.1000/x");
        assert!(json.get("declared").is_none(), "{json}");

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn every_tool_publishes_an_object_output_schema() {
        let server = PaxServer::new(PathBuf::from("."), Config::default());
        for tool in server.tool_router.list_all() {
            let schema = tool
                .output_schema
                .unwrap_or_else(|| panic!("{} has no output schema", tool.name));
            assert_eq!(schema["type"], "object", "{}", tool.name);
            if tool.name == "show" {
                let properties = schema["properties"].as_object().unwrap();
                for field in ["declared", "excluded", "candidate", "records"] {
                    assert!(properties.contains_key(field), "{field}");
                }
            }
            if tool.name == "index" {
                let papers = &schema["properties"]["papers"];
                assert_eq!(papers["type"], "array", "{schema:?}");
            }
            if tool.name == "similar" {
                let properties = schema["properties"].as_object().unwrap();
                for field in ["seed", "seed_fallback", "papers", "unindexed"] {
                    assert!(properties.contains_key(field), "{field}");
                }
            }
            if tool.name == "semantic_search" {
                for field in ["hits", "unindexed"] {
                    assert_eq!(schema["properties"][field]["type"], "array", "{schema:?}");
                }
            }
            if tool.name == "rank_candidates" {
                let properties = schema["properties"].as_object().unwrap();
                for field in ["seed", "seed_fallback", "ranked", "failed", "note"] {
                    assert!(properties.contains_key(field), "{field}");
                }
            }
        }
    }

    #[tokio::test]
    async fn index_without_a_library_is_a_clear_error() {
        let server = PaxServer::new(PathBuf::from("/nonexistent/pax-mcp"), Config::default());
        let Err(error) = server.index_with(&FakeEmbedder::new()).await else {
            panic!("indexed without a library");
        };
        assert!(error.contains("no pax library"), "{error}");
    }

    #[tokio::test]
    async fn index_reports_papers_without_a_pdf_as_fallbacks_without_running_nix() {
        let root = library_with_two_papers("index-fallback");
        without_hashes(&root);
        let server = PaxServer::new(root.clone(), Config::default());

        let Json(report) = server.index_with(&FakeEmbedder::new()).await.unwrap();
        let keys: Vec<&str> = report
            .papers
            .iter()
            .map(|p| p.citation_key.as_str())
            .collect();
        assert_eq!(keys, ["turing1936", "hewitt1973"]);
        for paper in &report.papers {
            assert!(
                matches!(&paper.status, PaperStatus::Fallback { reason } if reason.contains("no PDF source")),
                "{paper:?}"
            );
        }
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["papers"][0]["status"], "fallback");

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn index_embeds_a_fetched_pdfs_full_text_and_falls_back_when_it_cant() {
        if !nix_available() || !pdftotext_available() {
            eprintln!("skipped: nix or pdftotext isn't available");
            return;
        }
        let mut scratch = StoreScratch::new(library_with_two_papers("index-full-text"));
        let root = scratch.root.clone();
        let server = PaxServer::new(root.clone(), Config::default());
        let pdf = pdf_with_lines(&[
            "Introduction",
            "Indexing full texts lets an agent find passages by meaning.",
            "This line is unique to the pax-ai index test.",
        ]);
        let pdf = String::from_utf8(pdf).unwrap();
        add_local_pdf(&server, &mut scratch, "hewitt1973", &pdf).await;
        let status = |report: &BuildReport, key: &str| {
            report
                .papers
                .iter()
                .find(|p| p.citation_key == key)
                .unwrap_or_else(|| panic!("{key} in {report:?}"))
                .status
                .clone()
        };

        let Json(report) = server.index_with(&FakeEmbedder::new()).await.unwrap();
        assert!(
            matches!(status(&report, "hewitt1973"), PaperStatus::Indexed { chunks, .. } if chunks >= 1),
            "{report:?}"
        );
        // turing1936's recorded hash isn't even a valid one, so nix can't
        // resolve its PDF: a fallback, not a failed run.
        assert!(
            matches!(status(&report, "turing1936"), PaperStatus::Fallback { reason } if reason.contains("couldn't be resolved")),
            "{report:?}"
        );
        let vectors = PaperIndex::open(&root)
            .vectors("fake:model", "hewitt1973")
            .unwrap();
        assert!(!vectors.fallback);
        assert!(
            vectors.chunks[1..]
                .iter()
                .any(|c| c.text.contains("unique"))
        );

        // Without the stored entries, a failure embedding the chunks leaves
        // the title + venue vector: a fallback, not a failure.
        fs::remove_dir_all(root.join("research/.pax-ai")).unwrap();
        let Json(report) = server
            .index_with(&FakeEmbedder::failing_on("unique"))
            .await
            .unwrap();
        assert!(
            matches!(status(&report, "hewitt1973"), PaperStatus::Fallback { reason } if reason.contains("embedding the full text failed")),
            "{report:?}"
        );
    }

    fn similar_to(name: &str, limit: Option<usize>) -> SimilarParams {
        SimilarParams {
            name: name.to_string(),
            limit,
        }
    }

    fn semantic(query: &str, limit: Option<usize>) -> SemanticSearchParams {
        SemanticSearchParams {
            query: query.to_string(),
            limit,
        }
    }

    #[tokio::test]
    async fn similar_recommends_the_other_indexed_papers() {
        let root = library_with_two_papers("similar");
        without_hashes(&root);
        let server = PaxServer::new(root.clone(), Config::default());

        let Json(before) = server
            .similar_with("fake:model", similar_to("hewitt1973", None))
            .unwrap();
        assert!(before.papers.is_empty());
        assert_eq!(before.unindexed, ["turing1936", "hewitt1973"]);

        server.index_with(&FakeEmbedder::new()).await.unwrap();
        let Json(after) = server
            .similar_with("fake:model", similar_to("hewitt1973", None))
            .unwrap();
        assert_eq!(after.seed, "hewitt1973");
        assert!(after.seed_fallback);
        assert!(after.unindexed.is_empty());
        let [paper] = &after.papers[..] else {
            panic!("{after:?}");
        };
        assert_eq!(paper.citation_key, "turing1936");
        assert!(paper.fallback);
        let json = serde_json::to_value(&after).unwrap();
        assert_eq!(json["papers"][0]["citation_key"], "turing1936");

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn similar_refuses_a_zero_limit_a_blank_name_and_unknown_papers() {
        let root = library_with_two_papers("similar-refused");
        let server = PaxServer::new(root.clone(), Config::default());

        for (params, expected) in [
            (similar_to("hewitt1973", Some(0)), "`limit`"),
            (similar_to("  ", None), "`name`"),
            (similar_to("nobody", None), "nobody"),
        ] {
            let Err(error) = server.similar_with("fake:model", params) else {
                panic!("accepted a bad request");
            };
            assert!(error.contains(expected), "{error}");
        }
        let server = PaxServer::new(PathBuf::from("/nonexistent/pax-mcp"), Config::default());
        let Err(error) = server.similar_with("fake:model", similar_to("hewitt1973", None)) else {
            panic!("ran without a library");
        };
        assert!(error.contains("no pax library"), "{error}");

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn semantic_search_refuses_a_blank_query_a_zero_limit_and_no_library() {
        let root = library_with_two_papers("semantic-refused");
        let server = PaxServer::new(root.clone(), Config::default());
        let embedder = FakeEmbedder::new();
        for (params, expected) in [
            (semantic("  ", None), "blank"),
            (semantic("actors", Some(0)), "at least 1"),
        ] {
            let Err(error) = server.semantic_search_with(&embedder, params).await else {
                panic!("searched with {expected}");
            };
            assert!(error.contains(expected), "{error}");
        }
        assert!(embedder.kinds.lock().unwrap().is_empty());

        let server = PaxServer::new(PathBuf::from("/nonexistent/pax-mcp"), Config::default());
        let Err(error) = server
            .semantic_search_with(&embedder, semantic("actors", None))
            .await
        else {
            panic!("searched without a library");
        };
        assert!(error.contains("no pax library"), "{error}");

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn semantic_search_ranks_indexed_papers_and_lists_the_unindexed() {
        let root = library_with_two_papers("semantic");
        without_hashes(&root);
        let server = PaxServer::new(root.clone(), Config::default());
        // turing1936's title fails to embed, so it isn't indexed at all.
        let Json(report) = server
            .index_with(&FakeEmbedder::failing_on("Computable"))
            .await
            .unwrap();
        assert!(
            matches!(report.papers[0].status, PaperStatus::Failed { .. }),
            "{report:?}"
        );

        let embedder = FakeEmbedder::new();
        let Json(results) = server
            .semantic_search_with(
                &embedder,
                semantic(" A Universal Modular Actor Formalism ", None),
            )
            .await
            .unwrap();
        assert_eq!(*embedder.kinds.lock().unwrap(), [EmbedKind::Query]);
        let [hit] = &results.hits[..] else {
            panic!("{results:?}");
        };
        assert_eq!(hit.citation_key, "hewitt1973");
        assert_eq!(hit.chunk, "A Universal Modular Actor Formalism");
        assert!(hit.fallback);
        assert!((hit.score - 1.0).abs() < 1e-5, "{hit:?}");
        assert_eq!(results.unindexed, ["turing1936"]);
        let json = serde_json::to_value(&results).unwrap();
        assert_eq!(json["hits"][0]["citation_key"], "hewitt1973");

        fs::remove_dir_all(root).unwrap();
    }

    // Ranking provider candidates against a library paper (#65).

    /// A library whose two papers are indexed by [`FakeEmbedder`] (as
    /// fallbacks: neither has a PDF), with `hewitt1973` aliased `P17`.
    async fn indexed_library(name: &str) -> (PathBuf, PaxServer) {
        let root = library_with_two_papers(name);
        without_hashes(&root);
        let server = PaxServer::new(root.clone(), Config::default());
        server
            .edit(alias("hewitt1973", "P17"))
            .await
            .expect("alias set");
        server.index_with(&FakeEmbedder::new()).await.unwrap();
        (root, server)
    }

    fn ranking(seed: &str, references: &[&str], dois: &[&str]) -> RankCandidatesParams {
        RankCandidatesParams {
            seed: seed.to_string(),
            references: references.iter().map(|r| r.to_string()).collect(),
            dois: dois.iter().map(|d| d.to_string()).collect(),
        }
    }

    #[tokio::test]
    async fn rank_candidates_refuses_bad_arguments_and_an_unindexed_seed_before_any_lookup() {
        let root = library_with_two_papers("rank-refused");
        let server = PaxServer::new(root.clone(), Config::default());
        let embedder = FakeEmbedder::new();
        let too_many: Vec<String> = (0..=RANK_LIMIT).map(|n| format!("openalex:W{n}")).collect();
        let too_many: Vec<&str> = too_many.iter().map(String::as_str).collect();

        for (params, expected) in [
            (ranking("hewitt1973", &[], &[]), "at least one"),
            (ranking("hewitt1973", &too_many, &[]), "more than 100"),
            (ranking("hewitt1973", &["W123"], &[]), "`references`"),
            (ranking("hewitt1973", &[], &["openalex:W1"]), "not a DOI"),
            (ranking("  ", &["openalex:W1"], &[]), "`seed`"),
            (ranking("nobody", &["openalex:W1"], &[]), "nobody"),
            // Valid candidates, but the library isn't indexed: refused
            // before any provider is asked.
            (
                ranking("turing1936", &["openalex:W1"], &["10.1/x"]),
                "run `index` first",
            ),
            (ranking("hewitt1973", &["openalex:W1"], &[]), "`index`"),
        ] {
            let Err(error) = server.rank_candidates_with(&embedder, params).await else {
                panic!("ranked with {expected}");
            };
            assert!(error.contains(expected), "{error}");
        }
        assert!(embedder.kinds.lock().unwrap().is_empty());

        let server = PaxServer::new(PathBuf::from("/nonexistent/pax-mcp"), Config::default());
        let Err(error) = server
            .rank_candidates_with(&embedder, ranking("hewitt1973", &["openalex:W1"], &[]))
            .await
        else {
            panic!("ranked without a library");
        };
        assert!(error.contains("no pax library"), "{error}");

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn found_candidates_rank_best_first_flagged_against_the_library() {
        let (root, server) = indexed_library("rank-found").await;
        let seed = server.ranking_seed("fake:model", "seed", "P17").unwrap();
        let mut with_abstract = work(ProviderId::Crossref, "10.1/b", "Actors", Some("10.1/b"));
        with_abstract.abstract_text = Some("Message passing between actors.".to_string());
        let found = vec![
            (
                "openalex:W1".to_string(),
                Ok(work(
                    ProviderId::OpenAlex,
                    "W1",
                    "On Computable Numbers",
                    Some("10.1112/plms/s2-42.1.230"),
                )),
            ),
            ("10.1/b".to_string(), Ok(with_abstract)),
            (
                "openalex:W3".to_string(),
                Ok(work(
                    ProviderId::OpenAlex,
                    "W3",
                    "A Universal Modular Actor Formalism",
                    None,
                )),
            ),
            ("openalex:W4".to_string(), Err("not found".to_string())),
        ];
        let embedder = FakeEmbedder::new();

        let out = server.rank_found(&embedder, seed, found).await.unwrap();

        assert_eq!(*embedder.kinds.lock().unwrap(), [EmbedKind::Document]);
        assert_eq!(embedder.embedded(), 3, "the failed lookup isn't embedded");
        assert_eq!((out.seed.as_str(), out.seed_fallback), ("hewitt1973", true));
        assert_eq!(out.ranked[0].hit.work.id, "openalex:W3");
        assert!((out.ranked[0].score - 1.0).abs() < 1e-5, "{out:?}");
        assert!(out.ranked.windows(2).all(|w| w[0].score >= w[1].score));
        let turing = out
            .ranked
            .iter()
            .find(|r| r.hit.work.id == "openalex:W1")
            .unwrap();
        assert!(turing.hit.work.in_library && !turing.had_abstract);
        let actors = out
            .ranked
            .iter()
            .find(|r| r.hit.work.id == "crossref:10.1/b")
            .unwrap();
        assert!(actors.had_abstract && !actors.hit.work.in_library);
        let [failed] = &out.failed[..] else {
            panic!("{out:?}");
        };
        assert_eq!(
            (failed.reference.as_str(), failed.reason.as_str()),
            ("openalex:W4", "not found")
        );

        let json = serde_json::to_value(&out).unwrap();
        let first = &json["ranked"][0];
        assert_eq!(first["id"], "openalex:W3");
        assert_eq!(first["providers"], serde_json::json!(["openalex"]));
        assert_eq!(first["had_abstract"], false);
        assert!(first["score"].is_number());
        assert!(first.get("abstract").is_none(), "{first}");
        assert!(json["note"].as_str().unwrap().contains("not comparable"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_doi_ranks_the_record_with_an_abstract_or_says_why_there_is_none() {
        let doi = "10.1/a";
        let bare = work(ProviderId::OpenAlex, "W1", "A", Some(doi));
        let mut described = work(ProviderId::SemanticScholar, "S1", "A", Some(doi));
        described.abstract_text = Some("An abstract.".to_string());
        let results = HashMap::from([
            (ProviderId::OpenAlex, Ok(vec![bare.clone()])),
            (ProviderId::Crossref, Err(ProviderError::NotFound)),
            (ProviderId::SemanticScholar, Ok(vec![described.clone()])),
        ]);
        assert_eq!(doi_record(&results), Ok(described));

        let results = HashMap::from([
            (ProviderId::OpenAlex, Ok(vec![bare.clone()])),
            (ProviderId::Crossref, Err(ProviderError::NotFound)),
        ]);
        assert_eq!(doi_record(&results), Ok(bare));

        let results: HashMap<ProviderId, Result<Vec<CandidateWork>, ProviderError>> =
            HashMap::from([
                (ProviderId::OpenAlex, Err(ProviderError::NotFound)),
                (ProviderId::Crossref, Err(ProviderError::NotFound)),
            ]);
        assert_eq!(
            doi_record(&results),
            Err("no provider has a record of this DOI".to_string())
        );

        let results: HashMap<ProviderId, Result<Vec<CandidateWork>, ProviderError>> =
            HashMap::from([(
                ProviderId::OpenAlex,
                Err(ProviderError::Request("timed out".to_string())),
            )]);
        let Err(reason) = doi_record(&results) else {
            panic!("found a record");
        };
        assert!(
            reason.contains("openalex") && reason.contains("timed out"),
            "{reason}"
        );
    }

    fn ranked_seeds(seed: &str, rank_by: Option<&str>) -> CitationParams {
        CitationParams {
            reference: Some(seed.to_string()),
            rank_by: rank_by.map(str::to_string),
            ..CitationParams::default()
        }
    }

    #[tokio::test]
    async fn rank_by_refuses_an_unindexed_seed_before_any_provider_call() {
        let root = library_with_two_papers("rank-by-refused");
        let server = PaxServer::new(root.clone(), Config::default());
        let embedder = FakeEmbedder::new();
        let log = root.join("research/search-log.jsonl");

        for (rank_by, expected) in [
            ("hewitt1973", "run `index` first"),
            (" ", "`rank_by`"),
            ("nobody", "nobody"),
        ] {
            let params = SearchParams {
                query: Some("actors".to_string()),
                rank_by: Some(rank_by.to_string()),
                ..SearchParams::default()
            };
            let Err(error) = server.search_with(&embedder, params).await else {
                panic!("searched with rank_by {rank_by:?}");
            };
            assert!(error.contains(expected), "{error}");
            for tool in [
                CitationTool::Refs,
                CitationTool::CitedBy,
                CitationTool::Related,
            ] {
                // hewitt1973 would resolve offline and be logged, were the
                // seed not refused first.
                let params = ranked_seeds("hewitt1973", Some(rank_by));
                let Err(error) = server.citations_with(&embedder, params, tool).await else {
                    panic!("{tool:?} ran with rank_by {rank_by:?}");
                };
                assert!(error.contains(expected), "{error}");
            }
        }
        assert!(!log.exists(), "nothing was searched, so nothing is logged");
        assert!(embedder.kinds.lock().unwrap().is_empty());

        // Paging is still checked first, as without `rank_by`.
        let params = CitationParams {
            paging: PagingParams {
                limit: Some(0),
                ..PagingParams::default()
            },
            ..ranked_seeds("hewitt1973", Some("hewitt1973"))
        };
        let Err(error) = server
            .citations_with(&embedder, params, CitationTool::Refs)
            .await
        else {
            panic!("accepted a zero limit");
        };
        assert!(error.contains("`limit`"), "{error}");

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn rank_by_adds_a_ranking_and_leaves_the_rest_as_without_it() {
        let (root, server) = indexed_library("rank-by-citations").await;
        let embedder = FakeEmbedder::new();
        let log = root.join("research/search-log.jsonl");

        // hewitt1973 has no DOI, so no provider is asked and no paper
        // comes back, but the lookup is logged.
        let Json(plain) = server
            .citations_with(
                &embedder,
                ranked_seeds("hewitt1973", None),
                CitationTool::Refs,
            )
            .await
            .unwrap();
        assert!(plain.ranking.is_none());
        assert_eq!(
            serde_json::to_string(&plain).unwrap(),
            serde_json::to_string(&plain.output).unwrap(),
            "without rank_by, the output is pax-core's shape exactly"
        );
        let logged = fs::read_to_string(&log).unwrap().lines().count();

        let Json(ranked) = server
            .citations_with(
                &embedder,
                ranked_seeds("hewitt1973", Some("P17")),
                CitationTool::Related,
            )
            .await
            .unwrap();
        assert_eq!(ranked.ranking.as_deref(), Some(&[][..]));
        assert!(ranked.log.logged);
        assert!(fs::read_to_string(&log).unwrap().lines().count() > logged);
        let json = serde_json::to_value(&ranked).unwrap();
        assert_eq!(json["ranking"], serde_json::json!([]));
        assert_eq!(json["seeds"][0]["seed"], "hewitt1973");

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn a_ranked_page_scores_each_merged_paper_with_its_groups_abstract() {
        let (root, server) = indexed_library("rank-by-page").await;
        let seed = server
            .ranking_seed("fake:model", "rank_by", "hewitt1973")
            .unwrap();
        let actor_doi = "10.1/actors";
        let mut abstractless = work(
            ProviderId::OpenAlex,
            "W1",
            "Actor Systems in Practice",
            Some(actor_doi),
        );
        abstractless.venue = Some("IJCAI".to_string());
        let mut described = work(
            ProviderId::SemanticScholar,
            "S1",
            "Actor Systems in Practice",
            Some(actor_doi),
        );
        described.abstract_text = Some("Actors exchange messages.".to_string());
        let results: HashMap<ProviderId, Result<SearchPage, ProviderError>> = HashMap::from([
            (
                ProviderId::OpenAlex,
                Ok(SearchPage::new(
                    vec![
                        abstractless,
                        work(
                            ProviderId::OpenAlex,
                            "W2",
                            "A Universal Modular Actor Formalism",
                            None,
                        ),
                    ],
                    Some(2),
                    None,
                )),
            ),
            (
                ProviderId::SemanticScholar,
                Ok(SearchPage::new(vec![described], Some(1), None)),
            ),
        ]);
        let embedder = FakeEmbedder::new();

        let merged = pax_core::merge_pages(&results);
        let ranking = rank_merged(&embedder, &seed, merged.works.iter())
            .await
            .unwrap();

        let index = LibraryIndex::load(&root);
        let page = SearchResults::new(&results, &index, &LogOutcome::NoLibrary, true);
        let mut ids: Vec<&str> = page
            .results
            .iter()
            .map(|hit| hit.work.id.as_str())
            .collect();
        let mut ranked: Vec<&str> = ranking.iter().map(|entry| entry.id.as_str()).collect();
        assert_eq!(ranked.len(), 2, "one entry per merged paper: {ranking:?}");
        // W2's title alone is exactly the seed's title + venue text; the
        // merged W1 carries Semantic Scholar's abstract.
        assert_eq!(
            ranking[0],
            RankingEntry {
                id: "openalex:W2".to_string(),
                score: ranking[0].score,
                had_abstract: false,
            }
        );
        assert!((ranking[0].score - 1.0).abs() < 1e-5, "{ranking:?}");
        assert!(ranking[1].had_abstract && ranking[1].score < ranking[0].score);
        ids.sort();
        ranked.sort();
        assert_eq!(ids, ranked, "the ranking names the page's own ids");

        let output = RankedOutput {
            output: page.clone(),
            ranking: Some(ranking),
        };
        let json = serde_json::to_value(&output).unwrap();
        assert_eq!(json["ranking"][0]["id"], "openalex:W2");
        assert_eq!(
            json["results"],
            serde_json::to_value(&page).unwrap()["results"]
        );
        let unranked = RankedOutput {
            output: page.clone(),
            ranking: None,
        };
        assert_eq!(
            serde_json::to_string(&unranked).unwrap(),
            serde_json::to_string(&page).unwrap()
        );

        fs::remove_dir_all(root).unwrap();
    }

    fn alias(reference: &str, alias: &str) -> Parameters<EditParams> {
        Parameters(EditParams {
            alias: Some(alias.to_string()),
            ..edit_of(reference)
        })
    }

    #[tokio::test]
    async fn show_resolves_an_alias_without_the_network() {
        let root = library_with_two_papers("show-alias");
        let server = PaxServer::new(root.clone(), Config::default());

        let Json(edited) = server.edit(alias("hewitt1973", "P17")).await.unwrap();
        assert_eq!(edited.alias.as_deref(), Some("P17"));

        let Json(shown) = server
            .show(Parameters(ShowParams {
                reference: Some("P17".to_string()),
                ..ShowParams::default()
            }))
            .await
            .unwrap();
        let shown = shown.shown.expect("a reference is shown");
        assert!(shown.candidate.is_none() && shown.excluded.is_none());
        let shown = shown.declared.unwrap();
        assert_eq!(
            (shown.citation_key.as_str(), shown.alias.as_deref()),
            ("hewitt1973", Some("P17"))
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn every_tool_taking_a_citation_key_takes_an_alias() {
        let root = library_with_two_papers("alias-tools");
        let server = PaxServer::new(root.clone(), Config::default());
        server.edit(alias("hewitt1973", "P17")).await.unwrap();

        let Json(tagged) = server
            .edit(Parameters(EditParams {
                add_tags: vec!["included".to_string()],
                ..edit_of("P17")
            }))
            .await
            .unwrap();
        assert_eq!(tagged.citation_key, "hewitt1973");
        assert_eq!(tagged.tags, ["concurrency", "included"]);

        let Json(fetched) = server
            .fetch(Parameters(FetchParams {
                reference: "P17".to_string(),
                file: None,
                force: false,
            }))
            .await
            .unwrap();
        assert_eq!(fetched.citation_key, "hewitt1973");
        assert_eq!(fetched.status, FetchState::NoSourceUrl);

        let exported = bibtex(
            server
                .export(Parameters(ExportParams {
                    format: ExportFormat::Bibtex,
                    keys: Some(vec!["P17".to_string()]),
                    tag: None,
                    dir: None,
                }))
                .await,
        );
        assert_eq!(exported.citation_keys, ["hewitt1973"]);

        let Json(refs) = server
            .refs(Parameters(CitationParams {
                reference: Some("P17".to_string()),
                ..CitationParams::default()
            }))
            .await
            .unwrap();
        assert!(refs.seeds[0].error.is_none(), "{:?}", refs.seeds[0].error);

        let Json(listed) = server
            .list(Parameters(ListParams::default()))
            .await
            .unwrap();
        let listed = listed.declared.unwrap();
        let mut aliases: Vec<_> = listed
            .papers
            .iter()
            .map(|p| (p.citation_key.as_str(), p.alias.as_deref()))
            .collect();
        aliases.sort();
        assert_eq!(aliases, [("hewitt1973", Some("P17")), ("turing1936", None)]);

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn edit_changes_an_alias_by_its_old_name_and_an_empty_one_clears_it() {
        let root = library_with_two_papers("alias-change");
        let server = PaxServer::new(root.clone(), Config::default());
        server.edit(alias("hewitt1973", "P17")).await.unwrap();

        let Json(renamed) = server.edit(alias("P17", "P18")).await.unwrap();
        assert_eq!(
            (renamed.citation_key.as_str(), renamed.alias.as_deref()),
            ("hewitt1973", Some("P18"))
        );
        let Json(cleared) = server.edit(alias("P18", "")).await.unwrap();
        assert_eq!(cleared.alias, None);
        let library = Library::load(&pax_core::nix::papers_path(&root)).unwrap();
        assert_eq!(library.find("hewitt1973").unwrap().local.alias, None);

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn an_alias_another_paper_holds_or_an_invalid_one_is_refused() {
        let root = library_with_two_papers("alias-refused");
        let server = PaxServer::new(root.clone(), Config::default());
        server.edit(alias("hewitt1973", "P17")).await.unwrap();
        let papers = pax_core::nix::papers_path(&root);
        let before = fs::read_to_string(&papers).unwrap();

        let taken = server
            .edit(alias("turing1936", "P17"))
            .await
            .err()
            .expect("another paper's alias should be refused");
        assert!(taken.contains("already taken"), "{taken}");
        assert!(taken.contains("hewitt1973"), "{taken}");
        let a_key = server
            .edit(alias("turing1936", "hewitt1973"))
            .await
            .err()
            .expect("another paper's citation key should be refused");
        assert!(a_key.contains("already taken"), "{a_key}");
        let invalid = server
            .edit(alias("turing1936", "P 17"))
            .await
            .err()
            .expect("an invalid alias should be refused");
        assert!(invalid.contains("not a valid alias"), "{invalid}");
        assert_eq!(fs::read_to_string(&papers).unwrap(), before);

        let Parameters(params) = exclusion("10.1000/x", "off topic", Some("X"));
        let excluded = server
            .exclude(Parameters(ExcludeParams {
                alias: Some("P17".to_string()),
                ..params
            }))
            .await
            .err()
            .expect("exclude should refuse a declared paper's alias");
        assert!(excluded.contains("hewitt1973"), "{excluded}");
        assert!(!pax_core::nix::excluded_path(&root).exists());

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn an_excluded_alias_shows_the_exclusion_and_is_refused_by_edit_and_fetch() {
        let root = library_with_two_papers("alias-excluded");
        let server = PaxServer::new(root.clone(), Config::default());
        let excluded_as = |alias: Option<&str>| {
            let Parameters(params) = exclusion("10.1000/off-topic", "not empirical", Some("X"));
            Parameters(ExcludeParams {
                alias: alias.map(str::to_string),
                ..params
            })
        };

        let Json(excluded) = server.exclude(excluded_as(Some("P18"))).await.unwrap();
        assert_eq!(excluded.alias.as_deref(), Some("P18"));
        let Json(again) = server.exclude(excluded_as(None)).await.unwrap();
        assert_eq!(
            again.alias.as_deref(),
            Some("P18"),
            "an omitted alias is kept"
        );

        let Json(shown) = server
            .show(Parameters(ShowParams {
                reference: Some("P18".to_string()),
                ..ShowParams::default()
            }))
            .await
            .unwrap();
        let shown = shown.shown.expect("a reference is shown");
        assert!(shown.declared.is_none() && shown.candidate.is_none());
        let shown = shown.excluded.unwrap();
        assert_eq!(
            (shown.key.as_str(), shown.alias.as_deref()),
            ("10.1000/off-topic", Some("P18"))
        );

        let edit = server
            .edit(Parameters(EditParams {
                notes: Some("x".to_string()),
                ..edit_of("P18")
            }))
            .await
            .err()
            .expect("edit should refuse an excluded paper's alias");
        assert!(edit.contains("excluded"), "{edit}");
        let fetch = server
            .fetch(Parameters(FetchParams {
                reference: "P18".to_string(),
                file: None,
                force: false,
            }))
            .await
            .err()
            .expect("fetch should refuse an excluded paper's alias");
        assert!(fetch.contains("excluded"), "{fetch}");

        let Json(cleared) = server.exclude(excluded_as(Some(""))).await.unwrap();
        assert_eq!(cleared.alias, None);
        let Json(_) = server.exclude(excluded_as(Some("P18"))).await.unwrap();
        let Json(unexcluded) = server
            .unexclude(Parameters(UnexcludeParams {
                reference: "P18".to_string(),
            }))
            .await
            .unwrap();
        assert_eq!(unexcluded.alias.as_deref(), Some("P18"));

        fs::remove_dir_all(root).unwrap();
    }
}
