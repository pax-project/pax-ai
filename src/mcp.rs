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
use std::path::PathBuf;

use pax_core::merge::PROVIDER_ORDER;
use pax_core::{
    CITATION_PROVIDERS, CandidateId, CandidateWork, Config, DocType, ExclusionStage, Filters,
    LibraryIndex, Limit, ListFilter, LogOutcome, Logged, Paper, PaxError, ProviderId,
    SearchOptions, SearchPage, ShowResult, YearRange,
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
        Ok(defaults.with_limit(limit).with_filters(filters))
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
    /// The seed paper: a provider reference (e.g. `openalex:W2072794470`, from
    /// any provider) or the citation key of a paper in the library.
    pub reference: String,
    #[serde(flatten)]
    pub paging: PagingParams,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReferenceParams {
    /// A provider reference from `search` (e.g. `openalex:W2072794470`), or
    /// for `show` also a citation key already in the library.
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

/// Exactly one of the two fields is set, depending on whether the reference
/// named a declared paper (citation key) or a provider candidate.
#[derive(Debug, Serialize, JsonSchema)]
pub struct ShowOutput {
    pub declared: Option<DeclaredPaper>,
    pub candidate: Option<Candidate>,
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
    /// `records_screened - excluded_screening`.
    pub reports_sought: usize,
    /// Declared papers with no fetchable artifact (no `source_url` or no
    /// `hash`).
    pub not_retrieved: usize,
    /// `reports_sought - not_retrieved`.
    pub reports_assessed: usize,
    pub excluded_full_text: FullTextExclusions,
    /// Declared papers with a fetchable artifact.
    pub included: usize,
    /// Search-log lines whose provider failed; they count nowhere else.
    pub skipped_log_lines: usize,
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

#[derive(Debug, Serialize, JsonSchema)]
pub struct PaperList {
    pub papers: Vec<DeclaredPaper>,
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
    /// The search is recorded in the library's search log.
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

    /// The papers a seed paper cites, from OpenAlex and Semantic Scholar (the
    /// providers with real citation data), in the same shape as `search`:
    /// per-provider counts, then one merged list flagged against the
    /// library. Recorded in the library's search log.
    #[tool]
    async fn refs(
        &self,
        Parameters(params): Parameters<CitationParams>,
    ) -> Result<Json<SearchResults>, String> {
        let options = params.paging.options()?;
        let logged =
            pax_core::references_logged(&params.reference, &options, &self.root, &self.config)
                .await
                .map_err(|e| e.to_string())?;
        Ok(Json(self.results(logged, &CITATION_PROVIDERS)))
    }

    /// The papers citing a seed paper; otherwise as `refs`.
    #[tool]
    async fn cited_by(
        &self,
        Parameters(params): Parameters<CitationParams>,
    ) -> Result<Json<SearchResults>, String> {
        let options = params.paging.options()?;
        let logged =
            pax_core::cited_by_logged(&params.reference, &options, &self.root, &self.config)
                .await
                .map_err(|e| e.to_string())?;
        Ok(Json(self.results(logged, &CITATION_PROVIDERS)))
    }

    /// Show one paper: a provider reference from `search` (resolved live), or
    /// the citation key of a paper already in the library.
    #[tool]
    async fn show(
        &self,
        Parameters(params): Parameters<ReferenceParams>,
    ) -> Result<Json<ShowOutput>, String> {
        let shown = pax_core::show_reference(&params.reference, &self.root, &self.config)
            .await
            .map_err(|e| e.to_string())?;
        Ok(Json(match shown {
            ShowResult::Declared(paper) => ShowOutput {
                declared: Some(declared(&paper)),
                candidate: None,
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
                }
            }
        }))
    }

    /// Declare a search result in the library (metadata only; the PDF is
    /// fetched separately). Returns the new paper's citation key. A paper
    /// already in the library (same DOI or arXiv id) is refused.
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
    /// with a reason and a stage (`screening` or `full-text`), replacing any
    /// earlier exclusion of it. Excluded papers are flagged in `search`,
    /// `refs`, and `cited_by` results. A paper declared in the library is
    /// refused; it must be removed first.
    #[tool]
    async fn exclude(
        &self,
        Parameters(params): Parameters<ExcludeParams>,
    ) -> Result<Json<Excluded>, String> {
        let exclusion = pax_core::exclude_paper(
            &params.reference,
            &params.reason,
            params.stage.into(),
            params.title.as_deref(),
            &self.root,
            &self.config,
        )
        .await
        .map_err(|e| match e {
            PaxError::ExclusionTitleUnresolved(..) => format!("{e} with `title`"),
            other => other.to_string(),
        })?;
        Ok(Json(Excluded {
            key: exclusion.key,
            title: exclusion.title,
            reason: exclusion.reason,
            stage: exclusion.stage.to_string(),
        }))
    }

    /// The review's PRISMA 2020 flow-diagram counts: records identified per
    /// provider (database searches and citation searching), duplicates
    /// removed, records screened, exclusions at each stage (full-text ones
    /// by reason), reports not retrieved, and papers included. Computed from
    /// the library's files; nothing is written.
    #[tool]
    async fn stats(&self) -> Result<Json<PrismaStats>, String> {
        if !pax_core::nix::papers_path(&self.root).exists() {
            return Err(format!(
                "no pax library at {} (research/papers.nix not found)",
                self.root.display()
            ));
        }
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
}

impl PaxServer {
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
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for PaxServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("pax-mcp", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Tools over a pax research library: search providers for papers, \
                 follow a paper's references and citations, show one, add it to \
                 the library or exclude it with a reason, list what's declared, \
                 and report PRISMA flow counts. Writes go to research/ \
                 through pax-core, and searches are recorded in \
                 research/search-log.jsonl.",
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
        .map(|merged| MergedHit {
            work: candidate(
                &merged.work,
                merged.records().any(|work| index.contains(work)),
                merged.records().any(|work| index.is_excluded(work)),
            ),
            providers: merged.providers().iter().map(ToString::to_string).collect(),
            ids: merged.ids().iter().map(ToString::to_string).collect(),
            language_unknown: merged.records().all(language_unknown),
        })
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

/// Why `add` failed; a paper already in the library says so plainly, since
/// an agent is expected to retry with another result rather than treat it
/// as a failure.
fn add_error(e: PaxError) -> String {
    match e {
        PaxError::AlreadyDeclared {
            reference,
            citation_key,
        } => format!("not added: {reference} is already in the library as {citation_key:?}"),
        other => other.to_string(),
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
        included: stats.included,
        skipped_log_lines: stats.skipped_log_lines,
    }
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
    use pax_core::{Exclusion, Exclusions, Library, ProviderError};
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
                "add", "cited_by", "exclude", "list", "refs", "search", "show", "stats"
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
            .show(Parameters(ReferenceParams {
                reference: "hewitt1973".to_string(),
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
        }))
        .unwrap();
        let options = params.paging.options().unwrap();
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

    #[tokio::test]
    async fn refs_and_cited_by_refuse_an_invalid_seed_without_logging_it() {
        let root = library_with_two_papers("citations");
        let server = PaxServer::new(root.clone(), Config::default());
        let seed = |reference: &str| {
            Parameters(CitationParams {
                reference: reference.to_string(),
                ..CitationParams::default()
            })
        };

        let unknown_key = server
            .refs(seed("dijkstra1968"))
            .await
            .err()
            .expect("an unknown citation key should be refused");
        assert!(unknown_key.contains("dijkstra1968"), "{unknown_key}");
        let bad_reference = server
            .cited_by(seed("nowhere:123"))
            .await
            .err()
            .expect("an unknown provider should be refused");
        assert!(bad_reference.contains("nowhere"), "{bad_reference}");
        assert!(!root.join("research/search-log.jsonl").exists());

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn refs_and_cited_by_reject_invalid_paging_before_any_lookup() {
        let server = PaxServer::new(PathBuf::from("."), Config::default());
        let params = || {
            Parameters(CitationParams {
                reference: "openalex:W2072794470".to_string(),
                paging: PagingParams {
                    doc_type: Some("blog-post".to_string()),
                    ..PagingParams::default()
                },
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
        })
    }

    #[tokio::test]
    async fn exclude_records_a_doi_with_a_title_without_the_network() {
        let root = library_with_two_papers("exclude");
        let server = PaxServer::new(root.clone(), Config::default());

        let Json(excluded) = server
            .exclude(exclusion(
                "https://doi.org/10.1000/Off-Topic",
                "not empirical",
                Some("An Essay"),
            ))
            .await
            .unwrap();
        assert_eq!(excluded.key, "10.1000/off-topic");
        assert_eq!(excluded.stage, "full-text");
        assert_eq!(
            (excluded.title.as_str(), excluded.reason.as_str()),
            ("An Essay", "not empirical")
        );

        let excluded_work = work(
            ProviderId::OpenAlex,
            "W9",
            "An Essay",
            Some("10.1000/off-topic"),
        );
        assert!(LibraryIndex::load(&root).is_excluded(&excluded_work));
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
            ],
            Some(2),
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
        server
            .exclude(exclusion(
                "10.1000/essay",
                "not empirical",
                Some("An Essay"),
            ))
            .await
            .unwrap();

        let Json(stats) = server.stats().await.unwrap();
        assert_eq!(stats.identified.databases.get("openalex"), Some(&2));
        assert_eq!(stats.identified.total, 2);
        assert_eq!(stats.records_screened, 2);
        assert_eq!(stats.duplicates_removed, 0);
        assert_eq!(stats.excluded_full_text.total, 1);
        assert_eq!(stats.excluded_full_text.reasons[0].reason, "not empirical");
        assert_eq!(stats.excluded_full_text.reasons[0].count, 1);

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
    async fn add_rejects_a_reference_that_is_not_a_provider_id() {
        let server = PaxServer::new(PathBuf::from("."), Config::default());
        let result = server
            .add(Parameters(ReferenceParams {
                reference: "10.1145/359545.359563".to_string(),
            }))
            .await;
        assert!(result.is_err());
    }
}
