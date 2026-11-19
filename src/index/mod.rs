//! The embedding index shared by related-paper recommendations, full-text
//! semantic search and RAG retrieval: each paper's full text, extracted from
//! its PDF and chunked, plus a title + venue vector (see #52).
//!
//! [`PaperIndex::build`] fills the [`EmbeddingStore`] from `papers.nix`;
//! [`PaperIndex::vectors`] reads one paper's vector set back. There is no
//! manifest: both load `papers.nix` through pax-core and work out each
//! paper's store keys, so `papers.nix` alone decides what's indexed.

pub mod chunk;
pub mod extract;
pub mod similarity;
pub mod store;

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use pax_core::{Exclusions, Library, Paper, PaxError, ResolvedArtifact};
use serde::Serialize;

use crate::blocking::blocking_with_timeout;
use crate::port::{EmbedKind, Embedder};
pub use store::{EmbeddingStore, Entry, EntryKey, StoredChunk};

/// How long resolving a paper's PDF may take. A paper downloaded by URL is
/// built through `nix`, which may download it again if this machine's store
/// lost it.
const RESOLVE_TIMEOUT: Duration = Duration::from_secs(300);

/// How long `pdftotext` may take on one PDF.
const EXTRACT_TIMEOUT: Duration = Duration::from_secs(120);

/// The most chunks sent to the embedder in one call.
const EMBED_BATCH: usize = 32;

/// The embedding index of one library.
pub struct PaperIndex {
    root: PathBuf,
    store: EmbeddingStore,
}

/// What [`PaperIndex::build`] did, one report per declared paper, in
/// library order.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[cfg_attr(feature = "mcp", derive(schemars::JsonSchema))]
pub struct BuildReport {
    pub papers: Vec<PaperReport>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[cfg_attr(feature = "mcp", derive(schemars::JsonSchema))]
pub struct PaperReport {
    pub citation_key: String,
    #[serde(flatten)]
    pub status: PaperStatus,
}

/// Where a paper stands in the index after a build.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[cfg_attr(feature = "mcp", derive(schemars::JsonSchema))]
#[serde(tag = "status", rename_all = "kebab-case")]
pub enum PaperStatus {
    /// Its title + venue vector and `chunks` full-text chunks are stored.
    Indexed { chunks: usize },
    /// Only its title + venue vector is stored: no PDF on this machine, or
    /// its text couldn't be extracted or embedded.
    Fallback { reason: String },
    /// Not even its title + venue vector was stored (an embedding error).
    Failed { reason: String },
}

/// A paper's vector set: title + venue first, then full-text chunks if
/// indexed. `fallback` is set when it has no full-text chunks; `chunks` is
/// empty when the paper isn't indexed at all (for this model).
#[derive(Debug, Clone, PartialEq)]
pub struct PaperVectors {
    pub citation_key: String,
    pub fallback: bool,
    pub chunks: Vec<StoredChunk>,
}

impl PaperIndex {
    /// The index of the library at `root` (the directory containing
    /// `research/`). Touches nothing on disk itself.
    pub fn open(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
            store: EmbeddingStore::for_library(root),
        }
    }

    /// Indexes every declared paper (excluded papers never are): embeds its
    /// title + venue as a document, then its PDF's full text in chunks,
    /// reusing whatever the store already has, so a rebuild only embeds
    /// what changed. One paper's failure is that paper's report, never the
    /// run's; only a `papers.nix` that can't be read is an error. Ends by
    /// pruning the store of every entry the current `papers.nix` doesn't
    /// imply for this embedder's model, another model's entries included.
    ///
    /// The `nix` call that resolves a PDF and `pdftotext` run on tokio's
    /// blocking pool, each with a timeout, so this must run on a tokio
    /// runtime.
    pub async fn build(&self, embedder: &impl Embedder) -> Result<BuildReport, PaxError> {
        let library = Library::load(&pax_core::nix::papers_path(&self.root))?;
        let model_id = embedder.model_id();
        let mut keep = HashSet::new();
        let mut papers = Vec::with_capacity(library.papers().len());
        for paper in library.papers() {
            let keys = PaperKeys::of(paper, model_id);
            let status = self.build_paper(embedder, paper, &keys).await;
            keep.insert(keys.title_venue);
            keep.extend(keys.full_text);
            papers.push(PaperReport {
                citation_key: paper.local.citation_key.clone(),
                status,
            });
        }
        self.store.prune(&keep);
        Ok(BuildReport { papers })
    }

    /// A paper's vector set: title + venue first, then full-text chunks if indexed.
    ///
    /// `name` is a citation key or an alias, as for pax-core; an excluded
    /// paper's alias is [`PaxError::ExcludedNotDeclared`], an unknown name
    /// [`PaxError::NoSuchPaper`].
    pub fn vectors(&self, model_id: &str, name: &str) -> Result<PaperVectors, PaxError> {
        let library = Library::load(&pax_core::nix::papers_path(&self.root))?;
        let Some(paper) = library.find(name).or_else(|| library.find_alias(name)) else {
            let exclusions = Exclusions::load(&pax_core::nix::excluded_path(&self.root))?;
            return Err(match exclusions.find_alias(name) {
                Some(exclusion) => PaxError::ExcludedNotDeclared {
                    name: name.to_string(),
                    key: exclusion.key.clone(),
                },
                None => PaxError::NoSuchPaper(name.to_string()),
            });
        };
        Ok(self.paper_vectors(paper, model_id))
    }

    /// Every declared paper's vector set, in `papers.nix` order: what
    /// `similar` and `search` rank over. A paper with no entries for
    /// `model_id` (not indexed yet) has empty `chunks`.
    pub fn all_vectors(&self, model_id: &str) -> Result<Vec<PaperVectors>, PaxError> {
        let library = Library::load(&pax_core::nix::papers_path(&self.root))?;
        Ok(library
            .papers()
            .iter()
            .map(|paper| self.paper_vectors(paper, model_id))
            .collect())
    }

    fn paper_vectors(&self, paper: &Paper, model_id: &str) -> PaperVectors {
        let keys = PaperKeys::of(paper, model_id);
        let mut chunks = self
            .store
            .get(&keys.title_venue)
            .map(|entry| entry.chunks)
            .unwrap_or_default();
        let full_text = keys.full_text.and_then(|key| self.store.get(&key));
        let fallback = full_text.is_none();
        chunks.extend(full_text.into_iter().flat_map(|entry| entry.chunks));
        PaperVectors {
            citation_key: paper.local.citation_key.clone(),
            fallback,
            chunks,
        }
    }

    async fn build_paper(
        &self,
        embedder: &impl Embedder,
        paper: &Paper,
        keys: &PaperKeys,
    ) -> PaperStatus {
        if self.store.get(&keys.title_venue).is_none() {
            match embed_documents(embedder, &[keys.title_venue_text.as_str()]).await {
                Ok(chunks) => self.store.put(&keys.title_venue, &Entry { chunks }),
                Err(e) => {
                    return PaperStatus::Failed {
                        reason: format!("embedding the title and venue failed: {e}"),
                    };
                }
            }
        }

        let Some(full_text_key) = &keys.full_text else {
            return PaperStatus::Fallback {
                reason: no_pdf_reason(paper),
            };
        };
        if let Some(entry) = self.store.get(full_text_key) {
            return PaperStatus::Indexed {
                chunks: entry.chunks.len(),
            };
        }
        let chunks = match self.full_text_chunks(&paper.local.citation_key).await {
            Ok(chunks) => chunks,
            Err(reason) => return PaperStatus::Fallback { reason },
        };
        let mut stored = Vec::with_capacity(chunks.len());
        for batch in chunks.chunks(EMBED_BATCH) {
            let texts: Vec<&str> = batch.iter().map(|c| c.text.as_str()).collect();
            match embed_documents(embedder, &texts).await {
                Ok(embedded) => stored.extend(embedded),
                Err(e) => {
                    return PaperStatus::Fallback {
                        reason: format!("embedding the full text failed: {e}"),
                    };
                }
            }
        }
        let count = stored.len();
        self.store.put(full_text_key, &Entry { chunks: stored });
        PaperStatus::Indexed { chunks: count }
    }

    /// The paper's PDF, resolved through pax-core, extracted and chunked;
    /// the fallback reason when there's no text to embed.
    async fn full_text_chunks(&self, citation_key: &str) -> Result<Vec<chunk::Chunk>, String> {
        let (key, root) = (citation_key.to_string(), self.root.clone());
        let resolved = blocking_with_timeout(RESOLVE_TIMEOUT, move || {
            pax_core::resolve_artifact_path(&key, &root)
        })
        .await
        .map_err(|e| format!("resolving the PDF failed: {e}"))?
        .ok_or_else(|| {
            format!(
                "nix didn't resolve the PDF within {}s",
                RESOLVE_TIMEOUT.as_secs()
            )
        })?;
        let pdf = match resolved {
            Ok(ResolvedArtifact::Path(pdf)) => pdf,
            Ok(ResolvedArtifact::NotFetched) => return Err(NOT_FETCHED.to_string()),
            Ok(ResolvedArtifact::NoSourceUrl) => return Err(NO_SOURCE_URL.to_string()),
            Err(PaxError::LocalArtifactMissing(_)) => return Err(LOCAL_MISSING.to_string()),
            Err(e) => return Err(format!("the PDF couldn't be resolved: {e}")),
        };
        let text = blocking_with_timeout(EXTRACT_TIMEOUT, move || extract::extract_text(&pdf))
            .await
            .map_err(|e| format!("extracting the text failed: {e}"))?
            .ok_or_else(|| {
                format!(
                    "pdftotext didn't finish within {}s",
                    EXTRACT_TIMEOUT.as_secs()
                )
            })?
            .map_err(|e| e.to_string())?;
        let chunks = chunk::chunk(&text);
        if chunks.is_empty() {
            return Err(extract::ExtractError::Empty.to_string());
        }
        Ok(chunks)
    }
}

/// One paper [`PaperIndex::similar`] recommends.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[cfg_attr(feature = "mcp", derive(schemars::JsonSchema))]
pub struct SimilarPaper {
    pub citation_key: String,
    /// The paper's chunk-set similarity to the seed.
    pub score: f32,
    /// Only its title + venue vector is indexed, so its score compares less.
    pub fallback: bool,
}

/// The papers most similar to a seed paper, by their indexed vectors.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[cfg_attr(feature = "mcp", derive(schemars::JsonSchema))]
pub struct Similar {
    /// The seed's citation key.
    pub seed: String,
    pub seed_fallback: bool,
    /// Best first, at most `limit`, never the seed.
    pub papers: Vec<SimilarPaper>,
    /// Citation keys of the declared papers not indexed for this model,
    /// the seed included when it isn't.
    pub unindexed: Vec<String>,
}

impl PaperIndex {
    /// The `limit` declared papers most similar to the one `name` names.
    ///
    /// `name` is a citation key or alias, resolved as `vectors` does. Scores are
    /// `chunk_set_similarity`. An unindexed seed gives empty `papers`.
    /// Equal scores keep `papers.nix` order. Reads only the store: nothing
    /// is embedded.
    pub fn similar(&self, model_id: &str, name: &str, limit: usize) -> Result<Similar, PaxError> {
        let seed = self.vectors(model_id, name)?;
        let all = self.all_vectors(model_id)?;
        let unindexed = all
            .iter()
            .filter(|paper| paper.chunks.is_empty())
            .map(|paper| paper.citation_key.clone())
            .collect();
        let mut papers = Vec::new();
        if !seed.chunks.is_empty() {
            let seed_set = vector_set(&seed);
            papers = all
                .iter()
                .filter(|paper| paper.citation_key != seed.citation_key && !paper.chunks.is_empty())
                .map(|paper| SimilarPaper {
                    citation_key: paper.citation_key.clone(),
                    score: similarity::chunk_set_similarity(&seed_set, &vector_set(paper)),
                    fallback: paper.fallback,
                })
                .collect();
            papers.sort_by(|a, b| b.score.total_cmp(&a.score));
            papers.truncate(limit);
        }
        Ok(Similar {
            seed: seed.citation_key,
            seed_fallback: seed.fallback,
            papers,
            unindexed,
        })
    }
}

/// A paper's vectors, as the similarity functions take them.
fn vector_set(paper: &PaperVectors) -> Vec<&[f32]> {
    paper.chunks.iter().map(|c| c.vector.as_slice()).collect()
}

const NOT_FETCHED: &str = "the PDF hasn't been fetched yet";
const NO_SOURCE_URL: &str = "no PDF source is recorded, so there's no PDF to index";
const LOCAL_MISSING: &str =
    "the PDF was added from a local file, which isn't in this machine's Nix store";

/// Why a paper with no recorded hash has no PDF, as pax-core's
/// `resolve_artifact_path` tells the two cases apart, without calling it.
fn no_pdf_reason(paper: &Paper) -> String {
    if paper.artifact.source_url.is_some() {
        NOT_FETCHED
    } else {
        NO_SOURCE_URL
    }
    .to_string()
}

/// The store keys `papers.nix` implies for one paper under one model: a
/// title + venue key always, and a full-text key once its PDF's hash is
/// recorded.
struct PaperKeys {
    /// The text the title + venue vector embeds: the title, then the venue
    /// on its own line when there is one.
    title_venue_text: String,
    title_venue: EntryKey,
    full_text: Option<EntryKey>,
}

impl PaperKeys {
    fn of(paper: &Paper, model_id: &str) -> Self {
        let identity = &paper.identity;
        let text = match identity.venue.as_deref().map(str::trim) {
            Some(venue) if !venue.is_empty() => format!("{}\n{venue}", identity.title.trim()),
            _ => identity.title.trim().to_string(),
        };
        PaperKeys {
            title_venue: EntryKey::TitleVenue {
                model_id: model_id.to_string(),
                text: text.clone(),
            },
            title_venue_text: text,
            full_text: paper.artifact.hash.as_ref().map(|hash| EntryKey::FullText {
                model_id: model_id.to_string(),
                artifact_hash: hash.clone(),
                extractor_version: extract::EXTRACTOR_VERSION,
                chunker_version: chunk::CHUNKER_VERSION,
            }),
        }
    }
}

/// `texts` embedded as documents and L2-normalized, paired with their text.
async fn embed_documents(
    embedder: &impl Embedder,
    texts: &[&str],
) -> Result<Vec<StoredChunk>, String> {
    let vectors = embedder
        .embed(texts, EmbedKind::Document)
        .await
        .map_err(|e| e.to_string())?;
    if vectors.len() != texts.len() {
        return Err(format!(
            "the embedder returned {} vectors for {} texts",
            vectors.len(),
            texts.len()
        ));
    }
    Ok(texts
        .iter()
        .zip(vectors)
        .map(|(text, mut vector)| {
            similarity::normalize(&mut vector);
            StoredChunk {
                text: text.to_string(),
                vector,
            }
        })
        .collect())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::error::PortError;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// An offline [`Embedder`]: a text's vector is derived from its bytes
    /// (so equal texts get equal vectors), and any text containing
    /// `fail_on` fails the whole call. Counts the texts it embeds.
    pub(crate) struct FakeEmbedder {
        pub(crate) fail_on: Option<&'static str>,
        pub(crate) embedded: AtomicUsize,
    }

    impl FakeEmbedder {
        pub(crate) fn new() -> Self {
            FakeEmbedder {
                fail_on: None,
                embedded: AtomicUsize::new(0),
            }
        }

        pub(crate) fn failing_on(marker: &'static str) -> Self {
            FakeEmbedder {
                fail_on: Some(marker),
                ..Self::new()
            }
        }

        pub(crate) fn embedded(&self) -> usize {
            self.embedded.load(Ordering::Relaxed)
        }
    }

    impl Embedder for FakeEmbedder {
        fn model_id(&self) -> &str {
            "fake:model"
        }

        async fn embed(&self, texts: &[&str], kind: EmbedKind) -> Result<Vec<Vec<f32>>, PortError> {
            assert_eq!(kind, EmbedKind::Document);
            if let Some(marker) = self.fail_on
                && texts.iter().any(|t| t.contains(marker))
            {
                return Err(PortError::Request("the fake backend is down".to_string()));
            }
            self.embedded.fetch_add(texts.len(), Ordering::Relaxed);
            Ok(texts
                .iter()
                .map(|t| {
                    let bytes = t.as_bytes();
                    let sum: u32 = bytes.iter().map(|&b| u32::from(b)).sum();
                    vec![bytes.len() as f32 + 1.0, (sum % 97) as f32, 3.0]
                })
                .collect())
        }
    }

    struct Scratch(PathBuf);

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn paper(key: &str, title: &str, venue: Option<&str>, source_url: Option<&str>) -> String {
        let quoted = |v: Option<&str>| v.map_or("null".to_string(), |v| format!("\"{v}\""));
        format!(
            "  {key} = {{\n    doi = null;\n    title = \"{title}\";\n    authors = [ ];\n    \
             year = 2020;\n    venue = {};\n    source_url = {};\n    hash = null;\n    \
             tags = [ ];\n    notes = null;\n  }};\n",
            quoted(venue),
            quoted(source_url),
        )
    }

    /// A library whose papers have no hash, so nothing runs `nix`.
    fn library(name: &str, papers: &[String]) -> Scratch {
        let root = std::env::temp_dir().join(format!("pax-ai-index-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("research")).unwrap();
        write_papers(&root, papers);
        Scratch(root)
    }

    fn write_papers(root: &Path, papers: &[String]) {
        fs::write(
            root.join("research/papers.nix"),
            format!("{{\n{}}}\n", papers.concat()),
        )
        .unwrap();
    }

    fn statuses(report: &BuildReport) -> Vec<(&str, &PaperStatus)> {
        report
            .papers
            .iter()
            .map(|p| (p.citation_key.as_str(), &p.status))
            .collect()
    }

    fn entries(root: &Path) -> usize {
        fs::read_dir(root.join("research/.pax-ai/embeddings"))
            .map(|dir| dir.count())
            .unwrap_or(0)
    }

    #[tokio::test]
    async fn papers_without_a_pdf_get_a_normalized_title_venue_vector() {
        let scratch = library(
            "fallback",
            &[
                paper(
                    "a2020",
                    "Attention",
                    Some("NeurIPS"),
                    Some("https://x/a.pdf"),
                ),
                paper("b2020", "Bert", None, None),
            ],
        );
        let index = PaperIndex::open(&scratch.0);
        let embedder = FakeEmbedder::new();

        let report = index.build(&embedder).await.unwrap();
        assert_eq!(
            statuses(&report),
            [
                (
                    "a2020",
                    &PaperStatus::Fallback {
                        reason: NOT_FETCHED.to_string()
                    }
                ),
                (
                    "b2020",
                    &PaperStatus::Fallback {
                        reason: NO_SOURCE_URL.to_string()
                    }
                ),
            ]
        );
        assert_eq!(embedder.embedded(), 2);

        let vectors = index.vectors("fake:model", "a2020").unwrap();
        assert_eq!(vectors.citation_key, "a2020");
        assert!(vectors.fallback);
        assert_eq!(vectors.chunks.len(), 1);
        assert_eq!(vectors.chunks[0].text, "Attention\nNeurIPS");
        let norm: f32 = vectors.chunks[0].vector.iter().map(|x| x * x).sum();
        assert!((norm - 1.0).abs() < 1e-5, "{norm}");
        assert!(scratch.0.join("research/.pax-ai/.gitignore").exists());
    }

    #[tokio::test]
    async fn a_rebuild_reuses_stored_entries_without_the_embedder() {
        let scratch = library("reuse", &[paper("a2020", "Attention", None, None)]);
        let index = PaperIndex::open(&scratch.0);
        let first = index.build(&FakeEmbedder::new()).await.unwrap();

        let embedder = FakeEmbedder::new();
        let second = index.build(&embedder).await.unwrap();
        assert_eq!(second, first);
        assert_eq!(embedder.embedded(), 0);
    }

    #[tokio::test]
    async fn an_embedding_failure_fails_only_that_paper() {
        let scratch = library(
            "failed",
            &[
                paper("a2020", "Attention", None, None),
                paper("b2020", "Broken", None, None),
            ],
        );
        let index = PaperIndex::open(&scratch.0);

        let report = index
            .build(&FakeEmbedder::failing_on("Broken"))
            .await
            .unwrap();
        let [a, b] = &report.papers[..] else {
            panic!("{report:?}");
        };
        assert!(matches!(a.status, PaperStatus::Fallback { .. }), "{a:?}");
        let PaperStatus::Failed { reason } = &b.status else {
            panic!("{b:?}");
        };
        assert!(reason.contains("the fake backend is down"), "{reason}");

        let vectors = index.vectors("fake:model", "b2020").unwrap();
        assert!(vectors.chunks.is_empty());
        assert!(vectors.fallback);
    }

    #[tokio::test]
    async fn a_stored_full_text_entry_is_reused_without_resolving_the_pdf() {
        // A local artifact's hash with no entry would run `nix`; with its
        // entry stored, nothing does.
        let scratch = library("full-text", &[]);
        let mut with_hash = paper("a2020", "Attention", None, None);
        with_hash = with_hash.replace("hash = null", "hash = \"sha256-AAAA\"");
        write_papers(&scratch.0, &[with_hash]);
        let index = PaperIndex::open(&scratch.0);
        let full_text = EntryKey::FullText {
            model_id: "fake:model".to_string(),
            artifact_hash: "sha256-AAAA".to_string(),
            extractor_version: extract::EXTRACTOR_VERSION,
            chunker_version: chunk::CHUNKER_VERSION,
        };
        let chunk = |text: &str, vector: Vec<f32>| StoredChunk {
            text: text.to_string(),
            vector,
        };
        index.store.put(
            &full_text,
            &Entry {
                chunks: vec![chunk("one", vec![1.0, 0.0]), chunk("two", vec![0.0, 1.0])],
            },
        );

        let report = index.build(&FakeEmbedder::new()).await.unwrap();
        assert_eq!(
            statuses(&report),
            [("a2020", &PaperStatus::Indexed { chunks: 2 })]
        );
        let vectors = index.vectors("fake:model", "a2020").unwrap();
        assert!(!vectors.fallback);
        let texts: Vec<&str> = vectors.chunks.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, ["Attention", "one", "two"]);
    }

    #[tokio::test]
    async fn a_build_prunes_entries_papers_nix_no_longer_implies() {
        let scratch = library(
            "prune",
            &[
                paper("a2020", "Attention", None, None),
                paper("b2020", "Bert", None, None),
            ],
        );
        let index = PaperIndex::open(&scratch.0);
        index.build(&FakeEmbedder::new()).await.unwrap();
        assert_eq!(entries(&scratch.0), 2);

        // A retitled paper's old vector, and a removed paper's, are orphans.
        write_papers(&scratch.0, &[paper("a2020", "Attention Again", None, None)]);
        index.build(&FakeEmbedder::new()).await.unwrap();
        assert_eq!(entries(&scratch.0), 1);
        assert_eq!(
            index.vectors("fake:model", "a2020").unwrap().chunks[0].text,
            "Attention Again"
        );
    }

    #[test]
    fn vectors_resolve_aliases_and_refuse_unknown_or_excluded_names() {
        let scratch = library("names", &[]);
        let aliased = paper("a2020", "Attention", None, None)
            .replace("notes = null;", "notes = null;\n    alias = \"P1\";");
        write_papers(&scratch.0, &[aliased]);
        fs::write(
            scratch.0.join("research/excluded.nix"),
            "{\n  \"10.1/x\" = {\n    title = \"Gone\";\n    reason = \"off topic\";\n    \
             stage = \"screening\";\n    alias = \"P2\";\n  };\n}\n",
        )
        .unwrap();
        let index = PaperIndex::open(&scratch.0);

        let vectors = index.vectors("fake:model", "P1").unwrap();
        assert_eq!(vectors.citation_key, "a2020");
        assert!(vectors.chunks.is_empty());
        assert!(matches!(
            index.vectors("fake:model", "P2"),
            Err(PaxError::ExcludedNotDeclared { .. })
        ));
        assert!(matches!(
            index.vectors("fake:model", "nobody"),
            Err(PaxError::NoSuchPaper(_))
        ));
    }

    #[tokio::test]
    async fn all_vectors_lists_every_paper_in_order_indexed_or_not() {
        let scratch = library("all", &[paper("a2020", "Attention", None, None)]);
        let index = PaperIndex::open(&scratch.0);
        index.build(&FakeEmbedder::new()).await.unwrap();
        write_papers(
            &scratch.0,
            &[
                paper("a2020", "Attention", None, None),
                paper("b2020", "Bert", None, None),
            ],
        );

        let all = index.all_vectors("fake:model").unwrap();
        let keys: Vec<&str> = all.iter().map(|v| v.citation_key.as_str()).collect();
        assert_eq!(keys, ["a2020", "b2020"]);
        assert_eq!(all[0], index.vectors("fake:model", "a2020").unwrap());
        assert!(all[0].fallback && all[0].chunks.len() == 1);
        assert!(all[1].chunks.is_empty(), "b2020 isn't indexed yet");
        assert!(
            index.all_vectors("other:model").unwrap()[0]
                .chunks
                .is_empty()
        );
    }

    #[tokio::test]
    async fn an_unreadable_library_is_an_error() {
        let index = PaperIndex::open(Path::new("/nonexistent/pax-ai-index"));
        assert!(index.build(&FakeEmbedder::new()).await.is_err());
    }

    /// Stores `title_venue` as the paper's title + venue vector (for a
    /// paper titled `title`, with no venue) and, when given, `full_text`'s
    /// vectors as the full-text chunks of its hash.
    fn store_vectors(
        index: &PaperIndex,
        title: &str,
        title_venue: Vec<f32>,
        full_text: Option<(&str, Vec<Vec<f32>>)>,
    ) {
        let stored = |vectors: Vec<Vec<f32>>| Entry {
            chunks: vectors
                .into_iter()
                .map(|vector| StoredChunk {
                    text: title.to_string(),
                    vector,
                })
                .collect(),
        };
        let key = EntryKey::TitleVenue {
            model_id: "fake:model".to_string(),
            text: title.to_string(),
        };
        index.store.put(&key, &stored(vec![title_venue]));
        if let Some((hash, chunks)) = full_text {
            let key = EntryKey::FullText {
                model_id: "fake:model".to_string(),
                artifact_hash: hash.to_string(),
                extractor_version: extract::EXTRACTOR_VERSION,
                chunker_version: chunk::CHUNKER_VERSION,
            };
            index.store.put(&key, &stored(chunks));
        }
    }

    /// A seed with full text (aliased `P1`), a full-text match, a fallback
    /// paper and an unindexed one, stored without the embedder or `nix`.
    fn similar_library(name: &str) -> (Scratch, PaperIndex) {
        let with_hash = |key: &str, title: &str, hash: &str| {
            paper(key, title, None, None).replace("hash = null", &format!("hash = \"{hash}\""))
        };
        let seed = with_hash("a2020", "Attention", "sha256-AAAA")
            .replace("notes = null;", "notes = null;\n    alias = \"P1\";");
        let scratch = library(name, &[]);
        write_papers(
            &scratch.0,
            &[
                paper("c2020", "Capsules", None, None),
                seed,
                paper("d2020", "Dropout", None, None),
                with_hash("b2020", "Bert", "sha256-BBBB"),
            ],
        );
        let index = PaperIndex::open(&scratch.0);
        store_vectors(
            &index,
            "Attention",
            vec![1.0, 0.0],
            Some(("sha256-AAAA", vec![vec![1.0, 0.0], vec![0.0, 1.0]])),
        );
        store_vectors(
            &index,
            "Bert",
            vec![1.0, 0.0],
            Some(("sha256-BBBB", vec![vec![0.0, 1.0]])),
        );
        store_vectors(&index, "Capsules", vec![0.0, 1.0], None);
        (scratch, index)
    }

    #[test]
    fn similar_ranks_by_chunk_set_similarity_and_lists_the_unindexed() {
        let (_scratch, index) = similar_library("similar");

        let similar = index.similar("fake:model", "P1", 10).unwrap();
        assert_eq!(similar.seed, "a2020");
        assert!(!similar.seed_fallback);
        // Bert matches every seed vector exactly both ways; Capsules matches
        // one of the seed's three, and the seed matches it: (1/3 + 1) / 2.
        let ranked: Vec<(&str, bool)> = similar
            .papers
            .iter()
            .map(|p| (p.citation_key.as_str(), p.fallback))
            .collect();
        assert_eq!(ranked, [("b2020", false), ("c2020", true)]);
        assert!((similar.papers[0].score - 1.0).abs() < 1e-5);
        assert!((similar.papers[1].score - 2.0 / 3.0).abs() < 1e-5);
        assert_eq!(similar.unindexed, ["d2020"]);

        let top = index.similar("fake:model", "a2020", 1).unwrap();
        assert_eq!(top.papers.len(), 1);
        assert_eq!(top.papers[0].citation_key, "b2020");
    }

    #[test]
    fn a_fallback_seed_still_finds_papers() {
        let (_scratch, index) = similar_library("similar-fallback");

        let similar = index.similar("fake:model", "c2020", 10).unwrap();
        assert!(similar.seed_fallback);
        let keys: Vec<&str> = similar
            .papers
            .iter()
            .map(|p| p.citation_key.as_str())
            .collect();
        assert_eq!(keys, ["b2020", "a2020"]);
    }

    #[test]
    fn an_unindexed_seed_gives_no_papers() {
        let (_scratch, index) = similar_library("similar-unindexed");

        let similar = index.similar("fake:model", "d2020", 10).unwrap();
        assert_eq!(similar.seed, "d2020");
        assert!(similar.papers.is_empty());
        assert_eq!(similar.unindexed, ["d2020"]);

        let other_model = index.similar("other:model", "a2020", 10).unwrap();
        assert!(other_model.papers.is_empty());
        assert_eq!(other_model.unindexed, ["c2020", "a2020", "d2020", "b2020"]);
    }

    #[test]
    fn similar_refuses_unknown_names() {
        let (_scratch, index) = similar_library("similar-names");
        assert!(matches!(
            index.similar("fake:model", "nobody", 10),
            Err(PaxError::NoSuchPaper(_))
        ));
    }

    #[test]
    fn the_report_serializes_with_a_flat_status() {
        let report = BuildReport {
            papers: vec![
                PaperReport {
                    citation_key: "a".to_string(),
                    status: PaperStatus::Indexed { chunks: 3 },
                },
                PaperReport {
                    citation_key: "b".to_string(),
                    status: PaperStatus::Fallback {
                        reason: "no PDF".to_string(),
                    },
                },
            ],
        };
        assert_eq!(
            serde_json::to_value(&report).unwrap(),
            serde_json::json!({ "papers": [
                { "citation_key": "a", "status": "indexed", "chunks": 3 },
                { "citation_key": "b", "status": "fallback", "reason": "no PDF" },
            ]})
        );
    }
}
