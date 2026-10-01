//! Caches an LLM-generated summary of a paper's abstract or full text.
//!
//! Per `CLAUDE.md`'s non-negotiables, a cached summary is a rebuildable
//! derived cache, never a second source of truth alongside `papers.nix`.
//! Entries are keyed by a hash of the summarized text itself — mirroring how
//! `pax-core::nix` content-addresses a fetched artifact by its own hash,
//! rather than by citation key — so identical abstract/full text always hits
//! the same entry and changed text naturally misses instead of needing
//! explicit invalidation, and the cache lives entirely outside `papers.nix`.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::error::PortError;
use crate::port::Summarizer;

/// An on-disk, content-addressed cache of paper summaries, backed by a
/// [`Summarizer`].
pub struct SummaryCache {
    dir: PathBuf,
}

impl SummaryCache {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// Summarizes `text` via `summarizer`, reusing an on-disk cache entry for
    /// identical input text when one already exists instead of making a new
    /// LLM call.
    pub async fn summarize(
        &self,
        summarizer: &impl Summarizer,
        text: &str,
    ) -> Result<String, PortError> {
        let hash = content_hash(text);
        let path = self.entry_path(&hash);

        if let Ok(cached) = std::fs::read_to_string(&path) {
            return Ok(cached);
        }

        let summary = summarizer.summarize(text).await?;
        self.write_entry(&hash, &path, &summary);
        Ok(summary)
    }

    fn entry_path(&self, hash: &str) -> PathBuf {
        self.dir.join(format!("{hash}.txt"))
    }

    /// Written to a sibling temp file and renamed into place, same rationale
    /// as `pax_core::library::Library::save`: a plain write truncates before
    /// the new content is flushed, so a concurrent reader of this same entry
    /// could otherwise observe a partial file.
    ///
    /// Best-effort: the summary was already generated successfully, so an
    /// incidental caching failure (no permission, disk full, ...) shouldn't
    /// fail the call that asked for it — it just costs a repeat LLM call next
    /// time.
    fn write_entry(&self, hash: &str, path: &Path, summary: &str) {
        if std::fs::create_dir_all(&self.dir).is_err() {
            return;
        }
        let tmp_path = self.dir.join(format!("{hash}.tmp.{}", std::process::id()));
        if std::fs::write(&tmp_path, summary).is_ok() {
            let _ = std::fs::rename(&tmp_path, path);
        }
    }
}

fn content_hash(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    struct CountingSummarizer {
        calls: Rc<Cell<u32>>,
        response: String,
    }

    impl Summarizer for CountingSummarizer {
        async fn summarize(&self, _text: &str) -> Result<String, PortError> {
            self.calls.set(self.calls.get() + 1);
            Ok(self.response.clone())
        }
    }

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "pax-ai-summary-cache-test-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[tokio::test]
    async fn identical_text_is_served_from_cache_without_a_second_summarizer_call() {
        let dir = scratch_dir("hit");
        let cache = SummaryCache::new(&dir);
        let calls = Rc::new(Cell::new(0));
        let summarizer = CountingSummarizer {
            calls: calls.clone(),
            response: "A concise summary.".to_string(),
        };

        let first = cache.summarize(&summarizer, "abstract text").await.unwrap();
        let second = cache.summarize(&summarizer, "abstract text").await.unwrap();

        assert_eq!(first, "A concise summary.");
        assert_eq!(second, "A concise summary.");
        assert_eq!(
            calls.get(),
            1,
            "second call should be served from the cache"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn different_text_gets_its_own_cache_entry_and_calls_the_summarizer_again() {
        let dir = scratch_dir("miss");
        let cache = SummaryCache::new(&dir);
        let calls = Rc::new(Cell::new(0));
        let summarizer = CountingSummarizer {
            calls: calls.clone(),
            response: "Summary.".to_string(),
        };

        cache.summarize(&summarizer, "text one").await.unwrap();
        cache.summarize(&summarizer, "text two").await.unwrap();

        assert_eq!(calls.get(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn a_fresh_summary_persists_to_disk_for_a_new_cache_instance() {
        let dir = scratch_dir("persist");
        let calls = Rc::new(Cell::new(0));
        let summarizer = CountingSummarizer {
            calls: calls.clone(),
            response: "Persisted summary.".to_string(),
        };

        SummaryCache::new(&dir)
            .summarize(&summarizer, "abstract text")
            .await
            .unwrap();

        let second_cache = SummaryCache::new(&dir);
        let result = second_cache
            .summarize(&summarizer, "abstract text")
            .await
            .unwrap();

        assert_eq!(result, "Persisted summary.");
        assert_eq!(
            calls.get(),
            1,
            "a fresh SummaryCache should still hit the on-disk entry"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn content_hash_is_stable_and_distinguishes_different_text() {
        assert_eq!(content_hash("a"), content_hash("a"));
        assert_ne!(content_hash("a"), content_hash("b"));
    }
}
