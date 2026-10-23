//! MCP server exposing a research library's `pax-core` operations as tools,
//! so an agent calls pax directly instead of going through a shell and
//! parsing CLI output.
//!
//! None of these tools need an LLM: they wrap `pax-core` calls, and every
//! write goes through `pax-core`, so an agent's decisions land in the same
//! `research/` files the `pax` CLI and lazy-pax use. Results are this
//! module's own serializable types, since `pax-core`'s don't derive
//! `Serialize`.

use std::future::Future;
use std::path::PathBuf;

use pax_core::{
    CandidateId, CandidateWork, Config, LibraryIndex, ListFilter, Paper, ProviderId, ShowResult,
};
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{Implementation, ServerCapabilities, ServerConfig};
use rmcp::{Json, ServerHandler, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Providers in the order results are reported, matching the `pax` CLI.
const PROVIDERS: [ProviderId; 4] = [
    ProviderId::OpenAlex,
    ProviderId::Crossref,
    ProviderId::SemanticScholar,
    ProviderId::ArXiv,
];

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchParams {
    /// Free-text query. Give exactly one of `query` or `author`.
    pub query: Option<String>,
    /// Author name, for an author-scoped search instead of free text.
    pub author: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReferenceParams {
    /// A provider reference from `search` (e.g. `openalex:W2072794470`), or
    /// for `show` also a citation key already in the library.
    pub reference: String,
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

/// A search hit from one provider, not yet declared in the library.
#[derive(Debug, Serialize, JsonSchema)]
pub struct Hit {
    /// The reference `show` and `add` accept, e.g. `openalex:W2072794470`.
    pub id: String,
    pub title: String,
    pub authors: Vec<String>,
    pub publish_date: String,
    pub doi: Option<String>,
    pub venue: Option<String>,
    pub pdf_available: bool,
    /// Whether a paper with this DOI is already declared in the library.
    pub in_library: bool,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct ProviderResults {
    pub provider: String,
    pub hits: Vec<Hit>,
    /// Set when this provider failed; the other providers' results still stand.
    pub error: Option<String>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct SearchResults {
    pub providers: Vec<ProviderResults>,
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
    pub candidate: Option<Hit>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct Added {
    pub citation_key: String,
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
    /// papers, by free text or by author. Each hit's `id` is what `show` and
    /// `add` accept.
    #[tool]
    async fn search(
        &self,
        Parameters(params): Parameters<SearchParams>,
    ) -> Result<Json<SearchResults>, String> {
        let config = self.config.clone();
        let results = match (params.query, params.author) {
            (Some(query), None) => {
                off_thread(move || async move { pax_core::search_all(&query, &config).await })
                    .await?
            }
            (None, Some(author)) => {
                off_thread(
                    move || async move { pax_core::search_by_author(&author, &config).await },
                )
                .await?
            }
            _ => return Err("give exactly one of `query` or `author`".to_string()),
        };
        let index = LibraryIndex::load(&self.root);
        let providers = PROVIDERS
            .iter()
            .filter_map(|provider| {
                let result = results.get(provider)?;
                Some(match result {
                    Ok(works) => ProviderResults {
                        provider: provider.to_string(),
                        hits: works.iter().map(|w| hit(w, &index)).collect(),
                        error: None,
                    },
                    Err(e) => ProviderResults {
                        provider: provider.to_string(),
                        hits: Vec::new(),
                        error: Some(e.to_string()),
                    },
                })
            })
            .collect();
        Ok(Json(SearchResults { providers }))
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
            ShowResult::Candidate(work) => ShowOutput {
                declared: None,
                candidate: Some(hit(&work, &LibraryIndex::load(&self.root))),
            },
        }))
    }

    /// Declare a search result in the library (metadata only; the PDF is
    /// fetched separately). Returns the new paper's citation key.
    #[tool]
    async fn add(
        &self,
        Parameters(params): Parameters<ReferenceParams>,
    ) -> Result<Json<Added>, String> {
        let id: CandidateId = params.reference.parse().map_err(|e| format!("{e}"))?;
        let key = pax_core::add_candidate(&id, &self.root, &self.config)
            .await
            .map_err(|e| e.to_string())?;
        Ok(Json(Added {
            citation_key: key.0,
        }))
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

#[tool_handler(router = self.tool_router)]
impl ServerHandler for PaxServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("pax-mcp", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Tools over a pax research library: search providers for papers, \
                 show one, add it to the library, and list what's declared. \
                 Writes go to research/papers.nix through pax-core.",
            )
    }
}

/// Runs a `pax-core` call on a blocking-pool thread, off the request path.
/// Needed for paged searches: their futures aren't `Send` (Crossref's
/// language filter keeps its state in a `RefCell`), so they can't be awaited
/// in a tool handler directly. The future is built and driven on that one
/// thread, by the server's runtime.
async fn off_thread<F, Fut, T>(make: F) -> Result<T, String>
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = T>,
    T: Send + 'static,
{
    let runtime = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || runtime.block_on(make()))
        .await
        .map_err(|e| format!("the request failed to run: {e}"))
}

fn hit(work: &CandidateWork, index: &LibraryIndex) -> Hit {
    Hit {
        id: work.id.to_string(),
        title: work.title.clone(),
        authors: work.authors.clone(),
        publish_date: work.publish_date.clone(),
        doi: work.doi.clone(),
        venue: work.venue.clone(),
        pdf_available: work.pdf_url.is_some(),
        in_library: index.contains(work),
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
        assert_eq!(names, ["add", "list", "search", "show"]);
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
            SearchParams {
                query: None,
                author: None,
            },
            SearchParams {
                query: Some("actor model".to_string()),
                author: Some("Hewitt".to_string()),
            },
        ] {
            assert!(server.search(Parameters(params)).await.is_err());
        }
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
