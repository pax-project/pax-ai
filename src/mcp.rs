//! MCP server exposing a research library's `pax-core` operations as tools,
//! so an agent calls pax directly instead of going through a shell and
//! parsing CLI output.
//!
//! None of these tools need an LLM: they wrap `pax-core` calls, and every
//! write goes through `pax-core`, so an agent's decisions land in the same
//! `research/` files the `pax` CLI and lazy-pax use — searches included,
//! which are recorded in `research/search-log.jsonl`. Results are
//! `pax-core`'s own output shapes (`pax_core::output`, and `Stats`), the
//! ones the `pax` CLI prints with `--json`, so the two agree field for
//! field; only `show`'s DOI lookup, which the CLI lacks, wraps one.

use std::collections::HashMap;
use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::time::Duration;

use pax_core::output::{
    Added, CheckResults, CheckState, CitationResults, DeclaredPaper, DoiRecords, Excluded,
    ExclusionList, Exported, FetchResult, FetchState, Lists, PaperList, SearchResults, Shown,
};
use pax_core::{
    CandidateId, Config, DocType, ExclusionFilter, ExclusionOptions, ExclusionStage, FetchOutcome,
    Filters, Library, LibraryIndex, Limit, ListFilter, Logged, NameHolder, Paper, PaperEdits,
    PaxError, ProviderId, ResolvedArtifact, SearchOptions, SearchPage, SeedResults, Stats,
    YearRange,
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
    /// from any provider), the citation key or alias of a paper in the
    /// library, or an excluded paper's alias. Give exactly one of
    /// `reference` or `references`.
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

/// The text format `export` renders.
#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ExportFormat {
    Bibtex,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExportParams {
    pub format: ExportFormat,
    /// Export only these papers, by citation key or alias; every one must be
    /// in the library.
    /// Without `keys` or `tag`, the whole library is exported.
    pub keys: Option<Vec<String>>,
    /// Export only papers with this exact tag (combined with `keys` by AND).
    pub tag: Option<String>,
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
    /// `with_abstracts` adds each paper's `abstract`, making the response
    /// several times larger. The search is recorded in the library's search
    /// log, under `label` when given.
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
        Ok(Json(self.results(logged, options.abstracts)))
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
    /// `label` when given.
    #[tool]
    async fn refs(
        &self,
        Parameters(params): Parameters<CitationParams>,
    ) -> Result<Json<CitationResults>, String> {
        let options = params.paging.options()?;
        let seeds = params.seeds()?;
        let logged =
            pax_core::references_many_logged(&seeds, &options, &self.root, &self.config).await;
        self.citations(logged, Lists::Citations, options.abstracts)
            .map(Json)
    }

    /// The papers citing one or more seed papers (forward snowballing);
    /// otherwise as `refs`, `with_abstracts` included (responses several
    /// times larger). A paper's own text can't list what cites it, so a
    /// missing list here has no fallback but the other provider's.
    #[tool]
    async fn cited_by(
        &self,
        Parameters(params): Parameters<CitationParams>,
    ) -> Result<Json<CitationResults>, String> {
        let options = params.paging.options()?;
        let seeds = params.seeds()?;
        let logged =
            pax_core::cited_by_many_logged(&seeds, &options, &self.root, &self.config).await;
        self.citations(logged, Lists::Citations, options.abstracts)
            .map(Json)
    }

    /// The papers the providers themselves recommend as related to one or
    /// more seed papers: OpenAlex's related works and Semantic Scholar's
    /// recommendations. They aren't citation edges, so they reach beyond the
    /// seeds' citation neighbourhood and make a snowballing start set more
    /// varied. Input and output are as `refs`, `with_abstracts` and `label`
    /// included. Recommendations have no total, so a related list is never
    /// exhaustive: `incomplete` names only the missing lists (`failed`,
    /// `unresolved`), and `complete` means no list is missing, not that
    /// every related paper was returned. Recorded in the library's search
    /// log, and counted in `stats` as `identified.related_works`.
    #[tool]
    async fn related(
        &self,
        Parameters(params): Parameters<CitationParams>,
    ) -> Result<Json<CitationResults>, String> {
        let options = params.paging.options()?;
        let seeds = params.seeds()?;
        let logged =
            pax_core::related_many_logged(&seeds, &options, &self.root, &self.config).await;
        self.citations(logged, Lists::Related, options.abstracts)
            .map(Json)
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

    /// Export declared papers as BibTeX text: the whole library, or only the
    /// papers named in `keys` (citation keys or aliases) and/or the papers
    /// with a `tag`. Returns the text (`bibtex`) and the exported papers'
    /// `citation_keys` instead of writing a file.
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
        Ok(Json(match params.format {
            ExportFormat::Bibtex => Exported::new(&papers),
        }))
    }

    /// Verify every fetched paper's PDF still reproduces from its recorded
    /// hash (one `nix store prefetch-file` per paper, so it needs the
    /// network), without writing anything. Papers not fetched yet are
    /// reported as such, and a paper added from a local file that isn't in
    /// this machine's Nix store as `missing`; one paper's failure doesn't
    /// stop the rest.
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
    /// store) is reported in `status` and `reason`, not as an error. An excluded paper's alias is refused: it isn't in the
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

    /// A logged search's results, flagged against the library as it is now,
    /// with each paper's abstract when `abstracts` were asked for.
    fn results<E: Display>(
        &self,
        logged: Logged<HashMap<ProviderId, Result<SearchPage, E>>>,
        abstracts: bool,
    ) -> SearchResults {
        let Logged { results, log, .. } = logged;
        let index = LibraryIndex::load(&self.root);
        SearchResults::new(&results, &index, &log, abstracts)
    }

    /// A logged multi-seed citation lookup's results, flagged against the
    /// library as it is now. When no seed was valid, no provider was asked
    /// and nothing was logged, so it's an error naming each seed's problem.
    fn citations(
        &self,
        logged: Logged<SeedResults>,
        lists: Lists,
        abstracts: bool,
    ) -> Result<CitationResults, String> {
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
        Ok(CitationResults::new(
            &results, &index, &log, lists, abstracts,
        ))
    }
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
                 (fetch, which can take minutes), verify fetched PDFs still \
                 reproduce (check), list what's declared or excluded (list), export it as \
                 BibTeX text (export), and report PRISMA flow counts (stats). \
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

/// What a fetch attempt means for the agent. Only a bad citation key or an
/// unreadable library is an error; a paper that can't be fetched is a
/// result, with the reason.
fn fetch_result(citation_key: &str, attempt: FetchAttempt) -> Result<FetchResult, String> {
    let (outcome, built) = match attempt {
        Ok(fetched) => fetched,
        Err(e @ PaxError::NoSourceUrl(_)) => {
            let mut result = FetchResult::failed(citation_key, &e);
            result.reason = Some(
                "no PDF source is recorded (the provider found no open-access copy when it \
                 was added), so there's nothing to fetch; if the user has the PDF (e.g. \
                 through a university login), fetch again with `file` set to its path, or \
                 record a source with `pax edit --source-url`"
                    .to_string(),
            );
            return Ok(result);
        }
        Err(
            ref e @ PaxError::HashMismatch {
                ref recorded,
                ref actual,
                ..
            },
        ) => {
            let mut result = FetchResult::failed(citation_key, e);
            result.reason = Some(format!(
                "a different file is already recorded (hash {recorded}; this file's is \
                 {actual}), so nothing was written; fetch again with `force: true` to \
                 replace it"
            ));
            return Ok(result);
        }
        Err(e @ PaxError::LocalArtifactMissing(_)) => {
            let mut result = FetchResult::failed(citation_key, &e);
            result.reason = Some(file_needed_reason());
            return Ok(result);
        }
        Err(e @ PaxError::Fetch(_)) => return Ok(FetchResult::failed(citation_key, &e)),
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

/// `fetch`'s reason for `file-needed`, and `check`'s error for `missing`:
/// pax-core's names the CLI's commands (`--file`, `sync --from`).
fn file_needed_reason() -> String {
    "this paper was added from a local file, which isn't in this Nix store; fetch again \
     with `file` set to the PDF's path to provide it again"
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
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
                "list",
                "refs",
                "related",
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
        })
    }

    #[tokio::test]
    async fn export_renders_the_library_or_a_selection_as_bibtex() {
        let root = library_with_two_papers("export");
        let server = PaxServer::new(root.clone(), Config::default());

        let Json(all) = server.export(export_of(None, None)).await.unwrap();
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

        let Json(by_key) = server
            .export(export_of(Some(&["hewitt1973"]), None))
            .await
            .unwrap();
        assert_eq!(by_key.citation_keys, ["hewitt1973"]);
        assert!(!by_key.bibtex.contains("turing1936"));
        assert!(
            by_key
                .bibtex
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
        assert!(file_needed.reason.unwrap().contains("`file`"));
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

    #[tokio::test]
    async fn fetch_with_a_file_records_its_hash_and_force_replaces_one() {
        if !nix_available() {
            eprintln!("skipped: nix isn't available");
            return;
        }
        let root = library_with_two_papers("fetch-file");
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
        assert_eq!(forced.status, FetchState::Fetched, "{:?}", forced.reason);
        assert_eq!(forced.hash.as_deref(), Some(hash.as_str()));
        assert!(!fs::read_to_string(&papers).unwrap().contains("sha256-AAAA"));

        fs::remove_dir_all(root).unwrap();
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
        }
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

        let Json(exported) = server
            .export(Parameters(ExportParams {
                format: ExportFormat::Bibtex,
                keys: Some(vec!["P17".to_string()]),
                tag: None,
            }))
            .await
            .unwrap();
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
