//! MCP server exposing a research library's `pax-core` operations as tools,
//! so an agent calls pax directly instead of going through a shell and
//! parsing CLI output.
//!
//! None of these tools need an LLM: they wrap `pax-core` calls, and every
//! write goes through `pax-core`, so an agent's decisions land in the same
//! `research/` files the `pax` CLI and lazy-pax use — searches included,
//! which are recorded in `research/search-log.jsonl`. Results are this
//! module's own serializable types, since `pax-core`'s don't derive
//! `Serialize`.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::time::Duration;

use pax_core::merge::PROVIDER_ORDER;
use pax_core::{
    CITATION_PROVIDERS, CandidateId, CandidateWork, CheckReport, CheckStatus, Config, DocType,
    Exclusion, ExclusionOptions, ExclusionStage, FetchOutcome, Filters, Library, LibraryIndex,
    Limit, ListFilter, LogOutcome, Logged, MergedWork, Paper, PaperEdits, PaxError, ProviderError,
    ProviderId, ResolvedArtifact, SearchOptions, SearchPage, SeedResults, ShowResult, YearRange,
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
        let options = defaults.with_limit(limit).with_filters(filters);
        match &self.label {
            Some(label) if label.trim().is_empty() => {
                Err("`label` must not be blank, e.g. `R2-backward`".to_string())
            }
            Some(label) => Ok(options.with_label(label.trim())),
            None => Ok(options),
        }
    }
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct SearchParams {
    /// Free-text query. Give exactly one of `query` or `author`.
    pub query: Option<String>,
    /// Author name, for an author-scoped search instead of free text.
    pub author: Option<String>,
    #[serde(flatten)]
    pub paging: PagingParams,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct CitationParams {
    /// One seed paper: a provider reference (e.g. `openalex:W2072794470`,
    /// from any provider) or the citation key of a paper in the library.
    /// Give exactly one of `reference` or `references`.
    pub reference: Option<String>,
    /// Several seed papers, each as `reference`, for one snowballing round:
    /// each seed is reported on its own, and the results are merged across
    /// every seed.
    pub references: Option<Vec<String>>,
    #[serde(flatten)]
    pub paging: PagingParams,
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
pub struct ReferenceParams {
    /// A provider reference from `search`, e.g. `openalex:W2072794470`.
    pub reference: String,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct ShowParams {
    /// A provider reference from `search` (e.g. `openalex:W2072794470`) or the
    /// citation key of a paper in the library. Give exactly one of
    /// `reference` or `doi`.
    pub reference: Option<String>,
    /// A DOI (bare or `https://doi.org/...`), looked up at every provider
    /// that resolves DOIs, since one DOI can match several providers'
    /// records.
    pub doi: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CitationKeyParams {
    /// The citation key of a paper in the library, e.g. `turing1936`.
    pub reference: String,
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
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct UnexcludeParams {
    /// The exclusion's key (as `exclude` returns it, or as `add`'s refusal
    /// names it), a DOI, or a provider reference from `search`.
    pub reference: String,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct EditParams {
    /// The citation key of a paper in the library.
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
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct ListParams {
    /// Case-insensitive substring of an author's name.
    pub author: Option<String>,
    /// Publication year.
    pub year: Option<i32>,
    /// Exact tag.
    pub tag: Option<String>,
}

/// The text format `export` renders.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ExportFormat {
    Bibtex,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExportParams {
    pub format: ExportFormat,
    /// Export only these citation keys; every one must be in the library.
    /// Without `keys` or `tag`, the whole library is exported.
    pub keys: Option<Vec<String>>,
    /// Export only papers with this exact tag (combined with `keys` by AND).
    pub tag: Option<String>,
}

/// One provider's record of a paper.
#[derive(Debug, Serialize, JsonSchema)]
pub struct Candidate {
    /// The reference `show` and `add` accept, e.g. `openalex:W2072794470`.
    pub id: String,
    pub title: String,
    pub authors: Vec<String>,
    pub publish_date: String,
    pub doi: Option<String>,
    pub venue: Option<String>,
    pub pdf_available: bool,
    /// Whether the paper is already declared in the library (matched by DOI
    /// or arXiv id, so any provider's record of it counts).
    pub in_library: bool,
    /// Whether the paper is excluded in `research/excluded.nix`.
    pub excluded: bool,
}

/// One paper in a result list, merged across every provider that returned
/// it. The flattened fields are its representative record (the published
/// version, if any); `in_library` and `excluded` hold if any record is.
#[derive(Debug, Serialize, JsonSchema)]
pub struct MergedHit {
    #[serde(flatten)]
    pub work: Candidate,
    /// The providers that returned this paper, without repeats.
    pub providers: Vec<String>,
    /// Every merged record's reference, representative (`id`) first; any of
    /// them works with `show` and `add`.
    pub ids: Vec<String>,
    /// Kept by a language filter although no record carries a language to
    /// check it against.
    pub language_unknown: bool,
}

/// How one provider applied one requested filter.
#[derive(Debug, Serialize, JsonSchema)]
pub struct AppliedFilter {
    /// `year`, `type`, or `language`.
    pub filter: String,
    /// `server-side` (the provider's `total` reflects it), `post-retrieval`
    /// (applied to the records fetched; `total` counts records before it),
    /// or `not applied` (the provider has no such data, so its results
    /// ignore the filter).
    pub applied: String,
}

/// What one provider returned.
#[derive(Debug, Serialize, JsonSchema)]
pub struct ProviderReport {
    pub provider: String,
    /// Records retrieved (0 when the provider failed).
    pub retrieved: usize,
    /// Records the provider reports matching in total; null when it doesn't
    /// report one.
    pub total: Option<u64>,
    /// Whether every matching record was retrieved; false when `total` is
    /// unknown. If not, ask again with a higher `limit` or `all`.
    pub complete: bool,
    /// The most this provider serves for one query, however far it's paged.
    pub max_retrievable: Option<u64>,
    /// Records fetched before any post-retrieval filter dropped some.
    pub scanned: u64,
    pub filters: Vec<AppliedFilter>,
    /// Retrieved records the language filter kept without a language to
    /// check.
    pub language_unknown: usize,
    /// Set when this provider failed; the other providers' results still
    /// stand.
    pub error: Option<String>,
}

/// Record counts before and after merging duplicates across providers.
#[derive(Debug, Serialize, JsonSchema)]
pub struct MergeCounts {
    /// Records retrieved from every provider.
    pub before: usize,
    /// Distinct papers (`results.len()`).
    pub after: usize,
    /// Records merged into another (`before - after`).
    pub duplicates: usize,
    /// Records with no DOI, arXiv id, or title and year, kept as their own
    /// rows since they can't be matched.
    pub unmatched: usize,
}

/// Whether a search was recorded in the library's search log.
#[derive(Debug, Serialize, JsonSchema)]
pub struct SearchLogStatus {
    pub logged: bool,
    /// The log file, when logged.
    pub path: Option<String>,
    /// Why it wasn't logged.
    pub reason: Option<String>,
}

/// Per-provider counts and one list merged across providers.
#[derive(Debug, Serialize, JsonSchema)]
pub struct SearchResults {
    pub providers: Vec<ProviderReport>,
    pub merge: MergeCounts,
    pub results: Vec<MergedHit>,
    pub log: SearchLogStatus,
}

/// One provider whose citation list for a seed is incomplete, and why.
#[derive(Debug, Serialize, JsonSchema)]
pub struct IncompleteList {
    pub provider: String,
    /// `withheld` (the provider holds the list back, e.g. at the publisher's
    /// request), `failed` (the request failed), `unresolved` (the provider
    /// can't look the seed up, or has no record of it), or `truncated`
    /// (`limit` stopped retrieval before the provider's total; the list is
    /// good, just short).
    pub reason: String,
}

/// One seed's part in a `refs` or `cited_by` lookup.
#[derive(Debug, Serialize, JsonSchema)]
pub struct SeedReport {
    /// The seed as given.
    pub seed: String,
    /// An earlier seed that is the same paper, whose lists these are.
    pub same_as: Option<String>,
    /// What each citation provider returned for this seed.
    pub providers: Vec<ProviderReport>,
    /// This seed's provider lists that are incomplete, with why.
    pub incomplete: Vec<IncompleteList>,
    /// Whether every provider returned this seed's whole list.
    pub complete: bool,
    /// Set when the seed itself is invalid (a malformed reference, or a
    /// citation key not in the library); no provider was asked for it, and
    /// the other seeds still answer.
    pub error: Option<String>,
}

/// One paper in a list merged across seeds and providers.
#[derive(Debug, Serialize, JsonSchema)]
pub struct CitationHit {
    #[serde(flatten)]
    pub hit: MergedHit,
    /// The seeds whose lists returned this paper, as given, in seed order.
    pub seeds: Vec<String>,
}

/// Per-seed reports and one list merged across every seed and provider.
#[derive(Debug, Serialize, JsonSchema)]
pub struct CitationResults {
    pub seeds: Vec<SeedReport>,
    /// Whether every seed's lists are complete.
    pub complete: bool,
    pub merge: MergeCounts,
    pub results: Vec<CitationHit>,
    pub log: SearchLogStatus,
}

/// A paper declared in the library.
#[derive(Debug, Serialize, JsonSchema)]
pub struct DeclaredPaper {
    pub citation_key: String,
    pub title: String,
    pub authors: Vec<String>,
    pub year: Option<i32>,
    pub doi: Option<String>,
    pub venue: Option<String>,
    pub tags: Vec<String>,
    pub notes: Option<String>,
    /// Whether the PDF has been fetched (its Nix hash is recorded).
    pub fetched: bool,
}

/// What one provider returned for a DOI.
#[derive(Debug, Serialize, JsonSchema)]
pub struct DoiLookup {
    pub provider: String,
    /// Whether the provider has a record for the DOI.
    pub found: bool,
    /// Set when the lookup failed (not when the provider just has no
    /// record); the other providers' records still stand.
    pub error: Option<String>,
}

/// Every provider's record of a DOI, merged as `search` merges results.
#[derive(Debug, Serialize, JsonSchema)]
pub struct DoiRecords {
    /// The DOI looked up, normalized.
    pub doi: String,
    /// The providers that resolve DOIs (arXiv doesn't).
    pub providers: Vec<DoiLookup>,
    /// Normally one paper; more if the providers' records don't merge.
    pub results: Vec<MergedHit>,
}

/// Exactly one field is set: `declared` for a citation key, `candidate` for
/// a provider reference, `records` for a DOI.
#[derive(Debug, Serialize, JsonSchema)]
pub struct ShowOutput {
    pub declared: Option<DeclaredPaper>,
    pub candidate: Option<Candidate>,
    pub records: Option<DoiRecords>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct Added {
    pub citation_key: String,
}

/// An exclusion as recorded in `research/excluded.nix`.
#[derive(Debug, Serialize, JsonSchema)]
pub struct Excluded {
    /// The key it's recorded under: the paper's DOI when known (so it's
    /// excluded from every provider), else its provider reference.
    pub key: String,
    pub title: String,
    pub reason: String,
    /// `screening` or `full-text`.
    pub stage: String,
    /// The paper's publication year, when known.
    pub year: Option<i32>,
    /// The protocol's exclusion criterion, when recorded.
    pub criterion: Option<String>,
    /// The screening round, when recorded.
    pub round: Option<String>,
}

/// A library's PRISMA 2020 flow-diagram counts, computed by pax-core from
/// the search log, the exclusions, and the declared papers.
#[derive(Debug, Serialize, JsonSchema)]
pub struct PrismaStats {
    pub identified: Identified,
    /// `identified.total - records_screened`, across the whole log.
    pub duplicates_removed: usize,
    /// Distinct papers among every record identified.
    pub records_screened: usize,
    /// Papers excluded at title/abstract screening.
    pub excluded_screening: usize,
    /// Distinct logged papers with no screening decision yet: neither
    /// excluded nor declared. Report these as still to screen, not as
    /// passed.
    pub awaiting_screening: usize,
    /// Papers that passed screening: declared papers plus papers excluded at
    /// full text.
    pub reports_sought: usize,
    /// Declared papers with no fetchable artifact (no `source_url` or no
    /// `hash`).
    pub not_retrieved: usize,
    /// `reports_sought - not_retrieved`.
    pub reports_assessed: usize,
    pub excluded_full_text: FullTextExclusions,
    /// Exclusions at each stage, by the protocol's criterion.
    pub excluded_by_criterion: CriterionExclusions,
    /// Declared papers with a fetchable artifact.
    pub included: usize,
    /// Search-log lines whose provider failed; they count nowhere else.
    pub skipped_log_lines: usize,
    /// Declared or excluded papers that match no logged record (e.g. added
    /// from a DOI the reviewer already knew); still counted in their box.
    /// When 0, `records_screened = excluded_screening + reports_sought +
    /// awaiting_screening`.
    pub decided_outside_log: usize,
}

/// Records identified, per provider tag (e.g. `openalex`).
#[derive(Debug, Serialize, JsonSchema)]
pub struct Identified {
    /// From database searches (`search`).
    pub databases: BTreeMap<String, usize>,
    /// From citation searching (`refs`, `cited_by`).
    pub citation_searching: BTreeMap<String, usize>,
    pub total: usize,
}

/// Papers excluded after reading the full text, by reason.
#[derive(Debug, Serialize, JsonSchema)]
pub struct FullTextExclusions {
    pub total: usize,
    pub reasons: Vec<ReasonCount>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct ReasonCount {
    pub reason: String,
    pub count: usize,
}

/// Exclusions at each stage, grouped by criterion. Each stage has one entry
/// per distinct criterion (sorted), then one with `criterion: null` for the
/// exclusions recording none; a stage with no exclusions has no entries.
#[derive(Debug, Serialize, JsonSchema)]
pub struct CriterionExclusions {
    /// Title/abstract screening exclusions.
    pub screening: Vec<CriterionCount>,
    /// Full-text exclusions.
    pub full_text: Vec<CriterionCount>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct CriterionCount {
    pub criterion: Option<String>,
    pub count: usize,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct PaperList {
    pub papers: Vec<DeclaredPaper>,
}

/// Selected papers rendered as text; nothing is written, so the client
/// decides where it goes.
#[derive(Debug, Serialize, JsonSchema)]
pub struct Exported {
    pub format: ExportFormat,
    /// The exported papers, in library order.
    pub citation_keys: Vec<String>,
    /// Empty when nothing matched.
    pub text: String,
}

/// Where one paper's artifact stands, as `pax check` reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum CheckState {
    /// No hash recorded yet (or no `source_url` to check it against); `fetch`
    /// hasn't run for it.
    NotFetched,
    /// The source still resolves to the recorded hash.
    Reproducible,
    /// The source now resolves to different content than recorded.
    Mismatch,
    /// The source couldn't be resolved (network failure, 404, ...).
    Error,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct PaperCheck {
    pub citation_key: String,
    pub status: CheckState,
    /// The recorded hash, on a mismatch.
    pub expected: Option<String>,
    /// The hash the source resolves to now, on a mismatch.
    pub actual: Option<String>,
    /// Why the source couldn't be resolved, on an error.
    pub error: Option<String>,
}

/// One report per declared paper, then the totals `pax check` prints.
#[derive(Debug, Serialize, JsonSchema)]
pub struct CheckResults {
    pub papers: Vec<PaperCheck>,
    pub checked: usize,
    pub reproducible: usize,
    pub not_fetched: usize,
    pub mismatched: usize,
    pub errored: usize,
}

/// How a `fetch` ended. Only `fetched` and `already-fetched` have a
/// `store_path`; every other state says in `reason` what to do next.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum FetchState {
    /// Downloaded now: its hash is recorded in `papers.nix` and the PDF is in
    /// the Nix store.
    Fetched,
    /// Its hash was already recorded; the PDF is in the Nix store.
    AlreadyFetched,
    /// No `source_url` is recorded (the provider found no open-access copy),
    /// so there's nothing to download.
    NoSourceUrl,
    /// Downloading failed (network error, 404, `nix` failure); nothing was
    /// recorded.
    FetchFailed,
    /// The source no longer matches the recorded hash.
    HashMismatch,
    /// The hash is recorded, but `nix build` couldn't produce the PDF.
    BuildFailed,
    /// `nix` didn't finish in time; the hash may still be recorded once it
    /// does.
    TimedOut,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct FetchResult {
    pub citation_key: String,
    pub status: FetchState,
    /// The recorded Nix hash, when there is one.
    pub hash: Option<String>,
    /// The PDF's Nix store path.
    pub store_path: Option<String>,
    /// Why there's no store path, and what to do about it.
    pub reason: Option<String>,
}

/// Serves one research library: `root` is the directory containing
/// `research/`, as for the `pax` CLI.
#[derive(Clone)]
pub struct PaxServer {
    root: PathBuf,
    config: Config,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl PaxServer {
    pub fn new(root: PathBuf, config: Config) -> Self {
        PaxServer {
            root,
            config,
            tool_router: Self::tool_router(),
        }
    }

    /// Search every provider (OpenAlex, Crossref, Semantic Scholar, arXiv) for
    /// papers, by free text or by author, optionally filtered by year, type,
    /// and language. Reports each provider's retrieved/total counts, then one
    /// list merged across providers, each paper flagged if it's already in
    /// the library or excluded. Any result `id` works with `show` and `add`.
    /// The search is recorded in the library's search log, under `label`
    /// when given.
    #[tool]
    async fn search(
        &self,
        Parameters(params): Parameters<SearchParams>,
    ) -> Result<Json<SearchResults>, String> {
        let options = params.paging.options()?;
        let (root, config) = (&self.root, &self.config);
        let logged = match (params.query, params.author) {
            (Some(query), None) => {
                pax_core::search_all_logged(&query, &options, root, config).await
            }
            (None, Some(author)) => {
                pax_core::search_by_author_logged(&author, &options, root, config).await
            }
            _ => return Err("give exactly one of `query` or `author`".to_string()),
        };
        Ok(Json(self.results(logged, &PROVIDER_ORDER)))
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
    /// retrieval; ask again with a higher `limit` or `all`. Recorded in the
    /// library's search log, under `label` when given.
    #[tool]
    async fn refs(
        &self,
        Parameters(params): Parameters<CitationParams>,
    ) -> Result<Json<CitationResults>, String> {
        let options = params.paging.options()?;
        let seeds = params.seeds()?;
        let logged =
            pax_core::references_many_logged(&seeds, &options, &self.root, &self.config).await;
        self.citations(logged).map(Json)
    }

    /// The papers citing one or more seed papers (forward snowballing);
    /// otherwise as `refs`. A paper's own text can't list what cites it, so
    /// a missing list here has no fallback but the other provider's.
    #[tool]
    async fn cited_by(
        &self,
        Parameters(params): Parameters<CitationParams>,
    ) -> Result<Json<CitationResults>, String> {
        let options = params.paging.options()?;
        let seeds = params.seeds()?;
        let logged =
            pax_core::cited_by_many_logged(&seeds, &options, &self.root, &self.config).await;
        self.citations(logged).map(Json)
    }

    /// Show one paper: by a provider reference from `search` (resolved live),
    /// the citation key of a paper already in the library, or a DOI. A DOI is
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
                    declared: None,
                    candidate: None,
                    records: Some(doi_records(doi, &results, &index)),
                }));
            }
            _ => return Err("give exactly one of `reference` or `doi`".to_string()),
        };
        let shown = pax_core::show_reference(&reference, &self.root, &self.config)
            .await
            .map_err(|e| e.to_string())?;
        Ok(Json(match shown {
            ShowResult::Declared(paper) => ShowOutput {
                declared: Some(declared(&paper)),
                candidate: None,
                records: None,
            },
            ShowResult::Candidate(work) => {
                let index = LibraryIndex::load(&self.root);
                ShowOutput {
                    declared: None,
                    candidate: Some(candidate(
                        &work,
                        index.contains(&work),
                        index.is_excluded(&work),
                    )),
                    records: None,
                }
            }
        }))
    }

    /// Declare a search result in the library (metadata only; the PDF is
    /// fetched separately). Returns the new paper's citation key. A paper
    /// already in the library (same DOI or arXiv id) is refused, and so is
    /// one excluded from the review: the refusal names its exclusion, which
    /// only `unexclude` reverses, so ask the reviewer before calling it.
    #[tool]
    async fn add(
        &self,
        Parameters(params): Parameters<ReferenceParams>,
    ) -> Result<Json<Added>, String> {
        let id: CandidateId = params.reference.parse().map_err(|e| format!("{e}"))?;
        let key = pax_core::add_candidate(&id, &self.root, &self.config)
            .await
            .map_err(add_error)?;
        Ok(Json(Added {
            citation_key: key.0,
        }))
    }

    /// Exclude a paper from the review: record it in research/excluded.nix
    /// with a reason and a stage (`screening` or `full-text`), and optionally
    /// the protocol's `criterion`, the screening `round` and the paper's
    /// `year`, replacing any earlier exclusion of it (so a criterion or round
    /// not given again is cleared). Excluded papers are flagged in `search`,
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
            .with_round(params.round);
        let exclusion =
            pax_core::exclude_paper_with(&params.reference, &options, &self.root, &self.config)
                .await
                .map_err(|e| match e {
                    PaxError::ExclusionTitleUnresolved(..) => format!("{e} with `title`"),
                    other => other.to_string(),
                })?;
        Ok(Json(excluded(exclusion)))
    }

    /// Reverse a recorded screening decision: remove the paper's exclusion
    /// from research/excluded.nix, so `add` accepts it again. Returns the
    /// removed entry. Excluding was the reviewer's decision, so only undo it
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
        Ok(Json(excluded(exclusion)))
    }

    /// Edit a declared paper's tags and notes (e.g. tag it with a review's
    /// screening decision). Returns the paper as it is after the edit.
    #[tool]
    async fn edit(
        &self,
        Parameters(params): Parameters<EditParams>,
    ) -> Result<Json<DeclaredPaper>, String> {
        let path = self.papers_path()?;
        let edits = paper_edits(params.add_tags, params.remove_tags, params.notes)?;
        pax_core::edit_paper(&params.reference, &self.root, &edits).map_err(|e| e.to_string())?;
        let library = Library::load(&path).map_err(|e| e.to_string())?;
        let paper = library
            .find(&params.reference)
            .ok_or_else(|| format!("{:?} disappeared after the edit", params.reference))?;
        Ok(Json(declared(paper)))
    }

    /// The review's PRISMA 2020 flow-diagram counts: records identified per
    /// provider (database searches and citation searching), duplicates
    /// removed, records screened, exclusions at each stage (full-text ones
    /// by reason, both stages by criterion), papers still awaiting a
    /// screening decision, reports sought (papers that passed screening),
    /// reports not retrieved, and papers included. Mid-round, report
    /// `awaiting_screening` as still to screen, not as passed. Computed from
    /// the library's files; nothing is written.
    #[tool]
    async fn stats(&self) -> Result<Json<PrismaStats>, String> {
        self.papers_path()?;
        let stats = pax_core::stats(&self.root).map_err(|e| e.to_string())?;
        Ok(Json(prisma_stats(stats)))
    }

    /// List the papers declared in the library, optionally filtered by author,
    /// year, and tag (all combined with AND).
    #[tool]
    async fn list(
        &self,
        Parameters(params): Parameters<ListParams>,
    ) -> Result<Json<PaperList>, String> {
        let library = pax_core::Library::load(&pax_core::nix::papers_path(&self.root))
            .map_err(|e| e.to_string())?;
        let filter = ListFilter {
            author: params.author,
            year: params.year,
            tag: params.tag,
        };
        let papers = pax_core::filter_papers(library.papers(), &filter)
            .iter()
            .map(declared)
            .collect();
        Ok(Json(PaperList { papers }))
    }

    /// Export declared papers as BibTeX text: the whole library, or only the
    /// given citation `keys` and/or the papers with a `tag`. Returns the text
    /// instead of writing a file.
    #[tool]
    async fn export(
        &self,
        Parameters(params): Parameters<ExportParams>,
    ) -> Result<Json<Exported>, String> {
        let library = Library::load(&self.papers_path()?).map_err(|e| e.to_string())?;
        let papers = select_papers(
            library.papers(),
            params.keys.as_deref(),
            params.tag.as_deref(),
        )?;
        let text = match params.format {
            ExportFormat::Bibtex => pax_core::bibtex::render(&papers),
        };
        Ok(Json(Exported {
            format: params.format,
            citation_keys: papers
                .iter()
                .map(|p| p.local.citation_key.clone())
                .collect(),
            text,
        }))
    }

    /// Verify every fetched paper's PDF still reproduces from its recorded
    /// hash (one `nix store prefetch-file` per paper, so it needs the
    /// network), without writing anything. Papers not fetched yet are
    /// reported as such; one paper's failure doesn't stop the rest.
    #[tool]
    async fn check(&self) -> Result<Json<CheckResults>, String> {
        self.papers_path()?;
        let root = self.root.clone();
        let reports = tokio::task::spawn_blocking(move || pax_core::check_library(&root))
            .await
            .map_err(|e| format!("check didn't finish: {e}"))?
            .map_err(|e| e.to_string())?;
        Ok(Json(check_results(reports)))
    }

    /// Download a declared paper's PDF through Nix: records its hash in
    /// `papers.nix` (unless already recorded), then builds it into the Nix
    /// store and returns the path. Can take minutes; it gives up after a
    /// timeout. A paper that can't be fetched (no source URL, download or
    /// build failure, hash mismatch) is reported in `status` and `reason`,
    /// not as an error.
    #[tool]
    async fn fetch(
        &self,
        Parameters(params): Parameters<CitationKeyParams>,
    ) -> Result<Json<FetchResult>, String> {
        self.papers_path()?;
        let (key, root) = (params.reference.clone(), self.root.clone());
        let attempt =
            blocking_with_timeout(FETCH_TIMEOUT, move || fetch_and_build(&key, &root)).await?;
        let result = match attempt {
            Some(attempt) => fetch_result(&params.reference, attempt)?,
            None => FetchResult {
                citation_key: params.reference,
                status: FetchState::TimedOut,
                hash: None,
                store_path: None,
                reason: Some(format!(
                    "nix didn't finish within {}s; it may still record the hash, so call \
                     `fetch` again later",
                    FETCH_TIMEOUT.as_secs()
                )),
            },
        };
        Ok(Json(result))
    }
}

impl PaxServer {
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

    /// A logged search's or citation lookup's results, flagged against the
    /// library as it is now.
    fn results<E: Display>(
        &self,
        logged: Logged<HashMap<ProviderId, Result<SearchPage, E>>>,
        providers: &[ProviderId],
    ) -> SearchResults {
        let Logged { results, log, .. } = logged;
        let index = LibraryIndex::load(&self.root);
        search_results(&results, providers, &index, log)
    }

    /// A logged multi-seed citation lookup's results, flagged against the
    /// library as it is now. When no seed was valid, no provider was asked
    /// and nothing was logged, so it's an error naming each seed's problem.
    fn citations(&self, logged: Logged<SeedResults>) -> Result<CitationResults, String> {
        let Logged { results, log, .. } = logged;
        let invalid: Vec<String> = results
            .seeds
            .iter()
            .filter_map(|seed| seed.results.as_ref().err().map(ToString::to_string))
            .collect();
        if invalid.len() == results.seeds.len() {
            return Err(invalid.join("; "));
        }
        let index = LibraryIndex::load(&self.root);
        Ok(citation_results(&results, &index, log))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for PaxServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("pax-mcp", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Tools over a pax research library: search providers for papers \
                 (search), follow papers' references and citations (refs, \
                 cited_by), show one by reference, citation key, or DOI (show), \
                 add it to the library (add) or exclude it with a reason and \
                 the protocol's criterion (exclude, and unexclude to reverse \
                 an exclusion the reviewer reconsiders), tag and annotate \
                 declared papers (edit), fetch a paper's PDF through Nix \
                 (fetch, which can take minutes), verify fetched PDFs still \
                 reproduce (check), list what's declared (list), export it as \
                 BibTeX text (export), and report PRISMA flow counts (stats). \
                 Writes go to research/ through pax-core, and searches are \
                 recorded in research/search-log.jsonl.",
            )
    }
}

/// Per-provider reports, in `providers` order, and the successful pages
/// merged into one list flagged against `index`.
fn search_results<E: Display>(
    results: &HashMap<ProviderId, Result<SearchPage, E>>,
    providers: &[ProviderId],
    index: &LibraryIndex,
    log: LogOutcome,
) -> SearchResults {
    let reports = providers
        .iter()
        .filter_map(|provider| Some(provider_report(*provider, results.get(provider)?)))
        .collect();
    let language_unknown = |work: &CandidateWork| {
        results
            .values()
            .flatten()
            .any(|page| page.is_language_unknown(work))
    };
    let merged = pax_core::merge_pages(results);
    let hits = merged
        .works
        .iter()
        .map(|merged| merged_hit(merged, index, &language_unknown))
        .collect();
    SearchResults {
        providers: reports,
        merge: MergeCounts {
            before: merged.before,
            after: merged.after,
            duplicates: merged.duplicates(),
            unmatched: merged.unmatched,
        },
        results: hits,
        log: log_status(log),
    }
}

/// Each seed's report, and pax-core's list merged across every seed and
/// provider, flagged against `index`.
fn citation_results(
    results: &SeedResults,
    index: &LibraryIndex,
    log: LogOutcome,
) -> CitationResults {
    let seed_name = |i: usize| results.seeds[i].seed.clone();
    let seeds: Vec<SeedReport> = results
        .seeds
        .iter()
        .map(|seed| {
            seed_report(
                &seed.seed,
                seed.same_as.map(seed_name),
                seed.results.as_ref(),
            )
        })
        .collect();
    let language_unknown = |work: &CandidateWork| {
        results
            .seeds
            .iter()
            .filter_map(|seed| seed.results.as_ref().ok())
            .flat_map(|lists| lists.values().flatten())
            .any(|page| page.is_language_unknown(work))
    };
    let merged = results.merge();
    let hits = merged
        .works
        .iter()
        .map(|work| {
            let mut seeds: Vec<String> = Vec::new();
            for name in work.seeds().into_iter().map(seed_name) {
                if !seeds.contains(&name) {
                    seeds.push(name);
                }
            }
            CitationHit {
                hit: merged_hit(&work.merged, index, &language_unknown),
                seeds,
            }
        })
        .collect();
    CitationResults {
        complete: seeds.iter().all(|seed| seed.complete),
        seeds,
        merge: MergeCounts {
            before: merged.before,
            after: merged.after,
            duplicates: merged.duplicates(),
            unmatched: merged.unmatched,
        },
        results: hits,
        log: log_status(log),
    }
}

/// One seed's per-provider reports and incomplete lists, or why the seed is
/// invalid.
fn seed_report(
    seed: &str,
    same_as: Option<String>,
    results: Result<&pax_core::CitationResults, &PaxError>,
) -> SeedReport {
    match results {
        Ok(lists) => {
            let incomplete: Vec<IncompleteList> = pax_core::incomplete_lists(lists)
                .into_iter()
                .map(|list| IncompleteList {
                    provider: list.provider.to_string(),
                    reason: list.reason.to_string(),
                })
                .collect();
            SeedReport {
                seed: seed.to_string(),
                same_as,
                providers: CITATION_PROVIDERS
                    .iter()
                    .filter_map(|provider| Some(provider_report(*provider, lists.get(provider)?)))
                    .collect(),
                complete: incomplete.is_empty(),
                incomplete,
                error: None,
            }
        }
        Err(e) => SeedReport {
            seed: seed.to_string(),
            same_as,
            providers: Vec::new(),
            incomplete: Vec::new(),
            complete: false,
            error: Some(e.to_string()),
        },
    }
}

/// One merged paper, flagged against `index`; `language_unknown` tells
/// whether a language filter kept a record without a language to check.
fn merged_hit(
    merged: &MergedWork,
    index: &LibraryIndex,
    language_unknown: &impl Fn(&CandidateWork) -> bool,
) -> MergedHit {
    MergedHit {
        work: candidate(
            &merged.work,
            merged.records().any(|work| index.contains(work)),
            merged.records().any(|work| index.is_excluded(work)),
        ),
        providers: merged.providers().iter().map(ToString::to_string).collect(),
        ids: merged.ids().iter().map(ToString::to_string).collect(),
        language_unknown: merged.records().all(language_unknown),
    }
}

/// Every provider's answer for a DOI, in `PROVIDER_ORDER`, and the records
/// found merged into one list flagged against `index`.
fn doi_records(
    doi: String,
    results: &HashMap<ProviderId, Result<Vec<CandidateWork>, ProviderError>>,
    index: &LibraryIndex,
) -> DoiRecords {
    let providers = PROVIDER_ORDER
        .iter()
        .filter_map(|provider| {
            let result = results.get(provider)?;
            Some(DoiLookup {
                provider: provider.to_string(),
                found: result.as_ref().is_ok_and(|works| !works.is_empty()),
                error: match result {
                    Err(ProviderError::NotFound) | Ok(_) => None,
                    Err(e) => Some(e.to_string()),
                },
            })
        })
        .collect();
    let merged = pax_core::merge_works(
        PROVIDER_ORDER
            .iter()
            .filter_map(|provider| results.get(provider)?.as_ref().ok())
            .flatten()
            .cloned(),
    );
    DoiRecords {
        doi,
        providers,
        results: merged
            .works
            .iter()
            .map(|merged| merged_hit(merged, index, &|_: &CandidateWork| false))
            .collect(),
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

fn provider_report<E: Display>(
    provider: ProviderId,
    result: &Result<SearchPage, E>,
) -> ProviderReport {
    match result {
        Ok(page) => ProviderReport {
            provider: provider.to_string(),
            retrieved: page.retrieved(),
            total: page.total,
            complete: page.is_complete(),
            max_retrievable: page.max_retrievable,
            scanned: page.scanned,
            filters: page
                .filters
                .iter()
                .map(|filter| AppliedFilter {
                    filter: filter.kind.to_string(),
                    applied: filter.application.to_string(),
                })
                .collect(),
            language_unknown: page.language_unknown.len(),
            error: None,
        },
        Err(e) => ProviderReport {
            provider: provider.to_string(),
            retrieved: 0,
            total: None,
            complete: false,
            max_retrievable: None,
            scanned: 0,
            filters: Vec::new(),
            language_unknown: 0,
            error: Some(e.to_string()),
        },
    }
}

fn log_status(log: LogOutcome) -> SearchLogStatus {
    let not_logged = |reason: String| SearchLogStatus {
        logged: false,
        path: None,
        reason: Some(reason),
    };
    match log {
        LogOutcome::Written(path) => SearchLogStatus {
            logged: true,
            path: Some(path.display().to_string()),
            reason: None,
        },
        LogOutcome::NoLibrary => {
            not_logged("no pax library at the server's root, so nothing was logged".to_string())
        }
        LogOutcome::Failed(e) => not_logged(format!(
            "writing the search log failed: {e}; the results still stand"
        )),
        other => not_logged(format!("not logged: {other:?}")),
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

fn excluded(exclusion: Exclusion) -> Excluded {
    Excluded {
        key: exclusion.key,
        title: exclusion.title,
        reason: exclusion.reason,
        stage: exclusion.stage.to_string(),
        year: exclusion.year,
        criterion: exclusion.criterion,
        round: exclusion.round,
    }
}

/// The tag and notes edits `edit` asks for. Blank tags are refused, since
/// they'd be written into `papers.nix` as empty strings, and so is an edit
/// with nothing in it (pax-core's own message for that names the CLI's
/// flags).
fn paper_edits(
    add_tags: Vec<String>,
    remove_tags: Vec<String>,
    notes: Option<String>,
) -> Result<PaperEdits, String> {
    if add_tags
        .iter()
        .chain(&remove_tags)
        .any(|tag| tag.trim().is_empty())
    {
        return Err("tags must not be blank".to_string());
    }
    if add_tags.is_empty() && remove_tags.is_empty() && notes.is_none() {
        return Err("nothing to edit: give `add_tags`, `remove_tags`, and/or `notes`".to_string());
    }
    Ok(PaperEdits {
        add_tags,
        remove_tags,
        notes,
        ..PaperEdits::default()
    })
}

/// The papers `export` renders, in library order: those named in `keys`
/// (when given) that carry `tag` (when given). A key that isn't in the
/// library is an error rather than silently missing from the export.
fn select_papers(
    papers: &[Paper],
    keys: Option<&[String]>,
    tag: Option<&str>,
) -> Result<Vec<Paper>, String> {
    if let Some(keys) = keys {
        let unknown: Vec<&str> = keys
            .iter()
            .filter(|key| !papers.iter().any(|p| &p.local.citation_key == *key))
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
        .filter(|p| keys.is_none_or(|keys| keys.contains(&p.local.citation_key)))
        .collect())
}

/// How long `fetch` waits. pax-core already bounds each `nix` call it makes
/// (prefetch, then `git add` and build), so this only caps their sum.
const FETCH_TIMEOUT: Duration = Duration::from_secs(300);

/// Runs blocking work (a `nix` call) on tokio's blocking pool, so it never
/// stalls the server's executor; `None` once `timeout` passes. The work
/// itself can't be cancelled and runs on to its end in the background.
async fn blocking_with_timeout<T: Send + 'static>(
    timeout: Duration,
    work: impl FnOnce() -> T + Send + 'static,
) -> Result<Option<T>, String> {
    match tokio::time::timeout(timeout, tokio::task::spawn_blocking(work)).await {
        Ok(joined) => joined
            .map(Some)
            .map_err(|e| format!("the task failed: {e}")),
        Err(_) => Ok(None),
    }
}

/// `fetch`'s two pax-core steps: record the hash (downloading, unless one is
/// recorded already), then build the PDF to get its store path.
type FetchAttempt = Result<(FetchOutcome, Result<ResolvedArtifact, PaxError>), PaxError>;

fn fetch_and_build(citation_key: &str, root: &Path) -> FetchAttempt {
    let outcome = pax_core::fetch_paper(citation_key, root)?;
    Ok((outcome, pax_core::resolve_artifact_path(citation_key, root)))
}

/// What a fetch attempt means for the agent. Only a bad citation key or an
/// unreadable library is an error; a paper that can't be fetched is a
/// result, with the reason.
fn fetch_result(citation_key: &str, attempt: FetchAttempt) -> Result<FetchResult, String> {
    let result = |status, hash: Option<String>, reason: String| FetchResult {
        citation_key: citation_key.to_string(),
        status,
        hash,
        store_path: None,
        reason: Some(reason),
    };
    let (outcome, built) = match attempt {
        Ok(fetched) => fetched,
        Err(PaxError::NoSourceUrl(_)) => {
            return Ok(result(
                FetchState::NoSourceUrl,
                None,
                "no PDF source is recorded (the provider found no open-access copy when it \
                 was added), so there's nothing to fetch; record one with `pax edit \
                 --source-url`"
                    .to_string(),
            ));
        }
        Err(e @ PaxError::Fetch(_)) => {
            return Ok(result(FetchState::FetchFailed, None, e.to_string()));
        }
        Err(e) => return Err(e.to_string()),
    };
    let (status, hash) = match outcome {
        FetchOutcome::Fetched { hash } => (FetchState::Fetched, hash),
        FetchOutcome::AlreadyFetched { hash } => (FetchState::AlreadyFetched, hash),
    };
    Ok(match built {
        Ok(ResolvedArtifact::Path(path)) => FetchResult {
            citation_key: citation_key.to_string(),
            status,
            hash: Some(hash),
            store_path: Some(path.display().to_string()),
            reason: None,
        },
        Err(PaxError::Build(message)) if message.contains("hash mismatch") => result(
            FetchState::HashMismatch,
            Some(hash),
            format!(
                "the source no longer matches the recorded hash; `check` reports it too: \
                 {message}"
            ),
        ),
        Err(PaxError::Build(message)) if message.contains("not tracked by Git") => result(
            FetchState::BuildFailed,
            Some(hash),
            format!(
                "research/ isn't tracked by git, and nix only sees tracked files; `git add` \
                 it in the repository that holds it, then fetch again: {message}"
            ),
        ),
        Err(e) => result(FetchState::BuildFailed, Some(hash), e.to_string()),
        Ok(_) => result(
            FetchState::BuildFailed,
            Some(hash),
            "the hash is recorded, but the PDF still couldn't be resolved".to_string(),
        ),
    })
}

fn check_results(reports: Vec<CheckReport>) -> CheckResults {
    let papers: Vec<PaperCheck> = reports
        .into_iter()
        .map(|report| {
            let check = |status| PaperCheck {
                citation_key: report.citation_key.clone(),
                status,
                expected: None,
                actual: None,
                error: None,
            };
            match report.status {
                CheckStatus::NotFetched => check(CheckState::NotFetched),
                CheckStatus::Reproducible => check(CheckState::Reproducible),
                CheckStatus::Mismatch { expected, actual } => PaperCheck {
                    expected: Some(expected),
                    actual: Some(actual),
                    ..check(CheckState::Mismatch)
                },
                CheckStatus::Error(message) => PaperCheck {
                    error: Some(message),
                    ..check(CheckState::Error)
                },
            }
        })
        .collect();
    let count = |state| papers.iter().filter(|p| p.status == state).count();
    CheckResults {
        checked: papers.len(),
        reproducible: count(CheckState::Reproducible),
        not_fetched: count(CheckState::NotFetched),
        mismatched: count(CheckState::Mismatch),
        errored: count(CheckState::Error),
        papers,
    }
}

fn candidate(work: &CandidateWork, in_library: bool, excluded: bool) -> Candidate {
    Candidate {
        id: work.id.to_string(),
        title: work.title.clone(),
        authors: work.authors.clone(),
        publish_date: work.publish_date.clone(),
        doi: work.doi.clone(),
        venue: work.venue.clone(),
        pdf_available: work.pdf_url.is_some(),
        in_library,
        excluded,
    }
}

fn prisma_stats(stats: pax_core::Stats) -> PrismaStats {
    PrismaStats {
        identified: Identified {
            databases: stats.identified.databases,
            citation_searching: stats.identified.citation_searching,
            total: stats.identified.total,
        },
        duplicates_removed: stats.duplicates_removed,
        records_screened: stats.records_screened,
        excluded_screening: stats.excluded_screening,
        awaiting_screening: stats.awaiting_screening,
        reports_sought: stats.reports_sought,
        not_retrieved: stats.not_retrieved,
        reports_assessed: stats.reports_assessed,
        excluded_full_text: FullTextExclusions {
            total: stats.excluded_full_text.total,
            reasons: stats
                .excluded_full_text
                .reasons
                .into_iter()
                .map(|r| ReasonCount {
                    reason: r.reason,
                    count: r.count,
                })
                .collect(),
        },
        excluded_by_criterion: CriterionExclusions {
            screening: criterion_counts(stats.excluded_by_criterion.screening),
            full_text: criterion_counts(stats.excluded_by_criterion.full_text),
        },
        included: stats.included,
        skipped_log_lines: stats.skipped_log_lines,
        decided_outside_log: stats.decided_outside_log,
    }
}

fn criterion_counts(counts: Vec<pax_core::CriterionCount>) -> Vec<CriterionCount> {
    counts
        .into_iter()
        .map(|c| CriterionCount {
            criterion: c.criterion,
            count: c.count,
        })
        .collect()
}

fn declared(paper: &Paper) -> DeclaredPaper {
    DeclaredPaper {
        citation_key: paper.local.citation_key.clone(),
        title: paper.identity.title.clone(),
        authors: paper.identity.authors.clone(),
        year: paper.identity.year,
        doi: paper.identity.doi.clone(),
        venue: paper.identity.venue.clone(),
        tags: paper.local.tags.clone(),
        notes: paper.local.notes.clone(),
        fetched: paper.artifact.hash.is_some(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pax_core::{Exclusion, Exclusions};
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
                "list",
                "refs",
                "search",
                "show",
                "stats",
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
        assert_eq!(all.papers.len(), 2);

        let Json(tagged) = server
            .list(Parameters(ListParams {
                tag: Some("computability".to_string()),
                ..ListParams::default()
            }))
            .await
            .unwrap();
        assert_eq!(tagged.papers.len(), 1);
        assert_eq!(tagged.papers[0].citation_key, "turing1936");
        assert!(tagged.papers[0].fetched);

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
        }))
        .unwrap();
        let options = params.paging.options().unwrap();
        assert_eq!(options.label.as_deref(), Some("R2-backward"));
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

        let out = search_results(&results, &PROVIDER_ORDER, &index, LogOutcome::NoLibrary);

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

        let out = search_results(
            &results,
            &PROVIDER_ORDER,
            &LibraryIndex::default(),
            LogOutcome::Written(PathBuf::from("research/search-log.jsonl")),
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
    fn a_failed_log_write_is_reported_without_failing_the_results() {
        let status = log_status(LogOutcome::Failed(std::io::Error::other("disk full")));
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
    fn a_withheld_list_is_reported_on_its_seed() {
        use pax_core::CitationError;

        let page = SearchPage::new(
            vec![work(ProviderId::OpenAlex, "W1", "A", Some("10.1000/a"))],
            Some(1),
            None,
        );
        let lists: pax_core::CitationResults = HashMap::from([
            (ProviderId::OpenAlex, Ok(page)),
            (
                ProviderId::SemanticScholar,
                Err(CitationError::Provider(ProviderError::Withheld {
                    message: "references withheld by the publisher".to_string(),
                })),
            ),
        ]);

        let report = seed_report("openalex:W9", None, Ok(&lists));

        assert!(!report.complete);
        assert_eq!(report.incomplete.len(), 1);
        assert_eq!(report.incomplete[0].reason, "withheld");
        assert_eq!(
            report.incomplete[0].provider,
            ProviderId::SemanticScholar.to_string()
        );
        let semantic_scholar = &report.providers[1];
        assert!(
            semantic_scholar
                .error
                .as_deref()
                .unwrap()
                .contains("withheld")
        );
        assert!(report.providers[0].complete && report.providers[0].error.is_none());
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
            &SearchOptions::default(),
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
            .add(Parameters(ReferenceParams {
                reference: "crossref:10.1112/plms/s2-42.1.230".to_string(),
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
            .add(Parameters(ReferenceParams {
                reference: "crossref:10.1000/essay".to_string(),
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
        assert_eq!(listed.papers.len(), 2, "nothing was declared");

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn add_rejects_a_reference_that_is_not_a_provider_id() {
        let server = PaxServer::new(PathBuf::from("."), Config::default());
        let result = server
            .add(Parameters(ReferenceParams {
                reference: "10.1145/359545.359563".to_string(),
            }))
            .await;
        assert!(result.is_err());
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
        let reloaded = declared(library.find("hewitt1973").unwrap());
        assert_eq!(reloaded.tags, edited.tags);
        assert_eq!(reloaded.notes, edited.notes);
        let Json(tagged) = server
            .list(Parameters(ListParams {
                tag: Some("review-1".to_string()),
                ..ListParams::default()
            }))
            .await
            .unwrap();
        assert_eq!(tagged.papers.len(), 1);

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
        })
    }

    #[tokio::test]
    async fn export_renders_the_library_or_a_selection_as_bibtex() {
        let root = library_with_two_papers("export");
        let server = PaxServer::new(root.clone(), Config::default());

        let Json(all) = server.export(export_of(None, None)).await.unwrap();
        assert_eq!(all.citation_keys, ["turing1936", "hewitt1973"]);
        assert!(all.text.contains("@article{turing1936,"), "{}", all.text);
        assert!(all.text.contains("@article{hewitt1973,"), "{}", all.text);
        assert!(
            all.text.contains("doi = {10.1112/plms/s2-42.1.230}"),
            "{}",
            all.text
        );

        let Json(by_key) = server
            .export(export_of(Some(&["hewitt1973"]), None))
            .await
            .unwrap();
        assert_eq!(by_key.citation_keys, ["hewitt1973"]);
        assert!(!by_key.text.contains("turing1936"));
        assert!(
            by_key
                .text
                .contains("title = {A Universal Modular Actor Formalism}")
        );

        let Json(by_tag) = server
            .export(export_of(None, Some("computability")))
            .await
            .unwrap();
        assert_eq!(by_tag.citation_keys, ["turing1936"]);

        let Json(none) = server
            .export(export_of(Some(&["hewitt1973"]), Some("computability")))
            .await
            .unwrap();
        assert!(none.citation_keys.is_empty() && none.text.is_empty());

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

    #[tokio::test]
    async fn check_reports_papers_without_a_source_url_as_not_fetched() {
        // Neither fixture paper has a `source_url`, so pax-core runs no `nix`
        // at all, even for turing1936's recorded hash.
        let root = library_with_two_papers("check");
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
            (checked.reproducible, checked.mismatched, checked.errored),
            (0, 0, 0)
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn check_results_carry_mismatched_hashes_and_errors() {
        let report = |key: &str, status| CheckReport {
            citation_key: key.to_string(),
            status,
        };
        let out = check_results(vec![
            report("a", CheckStatus::Reproducible),
            report(
                "b",
                CheckStatus::Mismatch {
                    expected: "sha256-old".to_string(),
                    actual: "sha256-new".to_string(),
                },
            ),
            report("c", CheckStatus::Error("fetch failed: 404".to_string())),
        ]);
        assert_eq!(
            (out.checked, out.reproducible, out.mismatched, out.errored),
            (3, 1, 1, 1)
        );
        assert_eq!(out.papers[1].status, CheckState::Mismatch);
        assert_eq!(out.papers[1].expected.as_deref(), Some("sha256-old"));
        assert_eq!(out.papers[1].actual.as_deref(), Some("sha256-new"));
        assert_eq!(out.papers[2].error.as_deref(), Some("fetch failed: 404"));
        assert!(out.papers[0].error.is_none() && out.papers[0].expected.is_none());
    }

    #[tokio::test]
    async fn fetch_reports_a_paper_without_a_source_url_without_running_nix() {
        let root = library_with_two_papers("fetch");
        let server = PaxServer::new(root.clone(), Config::default());
        let papers = pax_core::nix::papers_path(&root);
        let before = fs::read_to_string(&papers).unwrap();

        let Json(result) = server
            .fetch(Parameters(CitationKeyParams {
                reference: "hewitt1973".to_string(),
            }))
            .await
            .unwrap();
        assert_eq!(result.status, FetchState::NoSourceUrl);
        assert!(result.store_path.is_none() && result.hash.is_none());
        assert!(result.reason.unwrap().contains("source"));
        assert_eq!(fs::read_to_string(&papers).unwrap(), before);

        let unknown = server
            .fetch(Parameters(CitationKeyParams {
                reference: "dijkstra1968".to_string(),
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

        assert!(fetch_result("x", Err(PaxError::NoSuchPaper("x".to_string()))).is_err());
    }

    #[tokio::test]
    async fn blocking_work_past_its_timeout_is_reported_as_none() {
        let slow = blocking_with_timeout(Duration::from_millis(10), || {
            std::thread::sleep(Duration::from_millis(500));
        })
        .await
        .unwrap();
        assert!(slow.is_none());
        let quick = blocking_with_timeout(Duration::from_secs(5), || 42)
            .await
            .unwrap();
        assert_eq!(quick, Some(42));
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

        let out = doi_records(turing.to_string(), &results, &index);

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
        let out = doi_records(turing.to_string(), &failed, &index);
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
}
