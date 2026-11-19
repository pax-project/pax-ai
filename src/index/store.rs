//! The on-disk embedding store: one content-addressed file per [`EntryKey`]
//! under `research/.pax-ai/embeddings/`.
//!
//! Per `CLAUDE.md`'s non-negotiables, this is a rebuildable derived cache,
//! never a second source of truth: there is no manifest, so `papers.nix`
//! alone decides which entries matter, and anything else can be [pruned]. As
//! in [`SummaryCache`](crate::SummaryCache), an entry's file name is a hash of
//! everything its vectors depend on (the model, and the artifact hash plus
//! the extractor/chunker versions, or the title + venue text), so a changed
//! input or a new model is simply a miss and nothing is ever invalidated.
//!
//! # On-disk format
//!
//! `<sha256 of the key>.json`, a JSON object:
//!
//! ```json
//! { "format": 1,
//!   "key": { "kind": "full-text", "model_id": "...", "artifact_hash": "...",
//!            "extractor_version": 1, "chunker_version": 1 },
//!   "chunks": [ { "text": "...", "vector": [0.1, ...] } ] }
//! ```
//!
//! A title + venue key is stored as `{ "kind": "title-venue", "model_id",
//! "text_sha256" }`: the text itself never lands in the key. The file name is
//! the SHA-256 of that key object's JSON. `get` checks both `format` and the
//! stored key, so a file in an unknown format, or one whose key doesn't
//! match, reads as a miss. `f32`s round-trip exactly through JSON.
//!
//! [pruned]: EmbeddingStore::prune

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The version of the on-disk entry format. A file carrying any other
/// version reads as `None`.
const FORMAT_VERSION: u32 = 1;

/// What an entry's vectors were derived from, and so where it lives.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EntryKey {
    /// A paper's full-text chunks, from the artifact with this hash.
    FullText {
        model_id: String,
        artifact_hash: String,
        extractor_version: u32,
        chunker_version: u32,
    },
    /// A paper's title + venue vector. The store hashes `text`.
    TitleVenue { model_id: String, text: String },
}

/// One embedded chunk. `vector` is L2-normalized by the caller before `put`.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredChunk {
    pub text: String,
    pub vector: Vec<f32>,
}

/// Everything stored under one [`EntryKey`].
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub chunks: Vec<StoredChunk>,
}

/// The embedding store of one library.
pub struct EmbeddingStore {
    dir: PathBuf,
}

impl EmbeddingStore {
    /// `<root>/research/.pax-ai/embeddings/`; writes `research/.pax-ai/.gitignore`
    /// (`*`) on first put. Touches nothing on disk itself.
    pub fn for_library(root: &Path) -> Self {
        Self {
            dir: root.join("research").join(".pax-ai").join("embeddings"),
        }
    }

    /// The entry stored under `key`. Missing, unreadable, in an unknown
    /// format, or stored under a different key: `None`.
    pub fn get(&self, key: &EntryKey) -> Option<Entry> {
        let repr = KeyRepr::from(key);
        let bytes = std::fs::read(self.entry_path(&repr.file_stem())).ok()?;
        let file: EntryFile = serde_json::from_slice(&bytes).ok()?;
        if file.format != FORMAT_VERSION || file.key != repr {
            return None;
        }
        Some(Entry {
            chunks: file
                .chunks
                .into_iter()
                .map(|c| StoredChunk {
                    text: c.text,
                    vector: c.vector,
                })
                .collect(),
        })
    }

    /// Stores `entry` under `key`, replacing any previous one.
    ///
    /// Written to a sibling temp file and renamed into place, as in
    /// `SummaryCache::write_entry`, so a concurrent reader never sees a
    /// partial file. Best effort: the vectors were already computed, so a
    /// caching failure (no permission, disk full, ...) only costs a repeat
    /// embedding call next time.
    pub fn put(&self, key: &EntryKey, entry: &Entry) {
        if std::fs::create_dir_all(&self.dir).is_err() {
            return;
        }
        self.write_gitignore();

        let repr = KeyRepr::from(key);
        let file = EntryFileRef {
            format: FORMAT_VERSION,
            key: &repr,
            chunks: entry
                .chunks
                .iter()
                .map(|c| ChunkRef {
                    text: &c.text,
                    vector: &c.vector,
                })
                .collect(),
        };
        let Ok(bytes) = serde_json::to_vec(&file) else {
            return;
        };
        let stem = repr.file_stem();
        let tmp_path = self.dir.join(format!(
            "{stem}.tmp.{}.{}",
            std::process::id(),
            TMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        if std::fs::write(&tmp_path, bytes).is_ok()
            && std::fs::rename(&tmp_path, self.entry_path(&stem)).is_err()
        {
            let _ = std::fs::remove_file(&tmp_path);
        }
    }

    /// Deletes entries whose key isn't in `keep`; returns how many.
    ///
    /// Only entry files (`<sha256>.json`) are considered, so anything else in
    /// the directory, an in-flight temp file included, is left alone. A file
    /// that can't be deleted isn't counted.
    pub fn prune(&self, keep: &HashSet<EntryKey>) -> usize {
        let keep: HashSet<String> = keep.iter().map(|k| KeyRepr::from(k).file_stem()).collect();
        let Ok(dir) = std::fs::read_dir(&self.dir) else {
            return 0;
        };
        dir.filter_map(Result::ok)
            .filter(|e| {
                let name = e.file_name();
                let Some(stem) = name.to_str().and_then(|n| n.strip_suffix(".json")) else {
                    return false;
                };
                is_sha256_hex(stem)
                    && !keep.contains(stem)
                    && std::fs::remove_file(e.path()).is_ok()
            })
            .count()
    }

    fn entry_path(&self, stem: &str) -> PathBuf {
        self.dir.join(format!("{stem}.json"))
    }

    /// `research/.pax-ai/.gitignore` with `*`, so the whole cache directory
    /// (the `.gitignore` included) stays out of git without pax-core knowing
    /// about it. Written once, when missing; best effort.
    fn write_gitignore(&self) {
        let Some(pax_ai_dir) = self.dir.parent() else {
            return;
        };
        let path = pax_ai_dir.join(".gitignore");
        if !path.exists() {
            let _ = std::fs::write(path, "*\n");
        }
    }
}

/// Makes concurrent `put`s from one process use distinct temp files.
static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// An [`EntryKey`] as stored on disk and hashed into a file name.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
enum KeyRepr {
    FullText {
        model_id: String,
        artifact_hash: String,
        extractor_version: u32,
        chunker_version: u32,
    },
    TitleVenue {
        model_id: String,
        text_sha256: String,
    },
}

impl From<&EntryKey> for KeyRepr {
    fn from(key: &EntryKey) -> Self {
        match key {
            EntryKey::FullText {
                model_id,
                artifact_hash,
                extractor_version,
                chunker_version,
            } => KeyRepr::FullText {
                model_id: model_id.clone(),
                artifact_hash: artifact_hash.clone(),
                extractor_version: *extractor_version,
                chunker_version: *chunker_version,
            },
            EntryKey::TitleVenue { model_id, text } => KeyRepr::TitleVenue {
                model_id: model_id.clone(),
                text_sha256: sha256_hex(text.as_bytes()),
            },
        }
    }
}

impl KeyRepr {
    /// The SHA-256 of this key's JSON: stable, since serde writes a struct's
    /// fields in declaration order.
    fn file_stem(&self) -> String {
        let json = serde_json::to_vec(self).expect("a key of strings and integers serializes");
        sha256_hex(&json)
    }
}

#[derive(Deserialize)]
struct EntryFile {
    format: u32,
    key: KeyRepr,
    chunks: Vec<ChunkFile>,
}

#[derive(Deserialize)]
struct ChunkFile {
    text: String,
    vector: Vec<f32>,
}

#[derive(Serialize)]
struct EntryFileRef<'a> {
    format: u32,
    key: &'a KeyRepr,
    chunks: Vec<ChunkRef<'a>>,
}

#[derive(Serialize)]
struct ChunkRef<'a> {
    text: &'a str,
    vector: &'a [f32],
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh library root in the temp dir, removed on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "pax-ai-embedding-store-test-{name}-{}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            Self(dir)
        }

        fn embeddings_dir(&self) -> PathBuf {
            self.0.join("research/.pax-ai/embeddings")
        }

        fn entry_files(&self) -> Vec<PathBuf> {
            let mut files: Vec<PathBuf> = std::fs::read_dir(self.embeddings_dir())
                .map(|d| d.filter_map(Result::ok).map(|e| e.path()).collect())
                .unwrap_or_default();
            files.sort();
            files
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn full_text(hash: &str) -> EntryKey {
        EntryKey::FullText {
            model_id: "ollama:nomic-embed-text".to_string(),
            artifact_hash: hash.to_string(),
            extractor_version: 1,
            chunker_version: 1,
        }
    }

    fn title_venue(text: &str) -> EntryKey {
        EntryKey::TitleVenue {
            model_id: "ollama:nomic-embed-text".to_string(),
            text: text.to_string(),
        }
    }

    fn entry(texts: &[&str]) -> Entry {
        Entry {
            chunks: texts
                .iter()
                .enumerate()
                .map(|(i, t)| StoredChunk {
                    text: t.to_string(),
                    vector: vec![0.6, -0.8, i as f32 * 0.1, 1.0e-7, f32::MIN_POSITIVE],
                })
                .collect(),
        }
    }

    #[test]
    fn a_put_entry_reads_back_exactly_from_a_new_store() {
        let scratch = Scratch::new("roundtrip");
        let key = full_text("sha256-abc");
        let stored = entry(&["first chunk", "second \"chunk\"\n"]);

        EmbeddingStore::for_library(&scratch.0).put(&key, &stored);

        assert_eq!(
            EmbeddingStore::for_library(&scratch.0).get(&key),
            Some(stored)
        );
    }

    #[test]
    fn a_missing_entry_is_none() {
        let scratch = Scratch::new("missing");
        let store = EmbeddingStore::for_library(&scratch.0);
        assert_eq!(store.get(&full_text("sha256-abc")), None);
        assert!(!scratch.0.exists(), "get must not create anything");
    }

    #[test]
    fn put_replaces_the_previous_entry() {
        let scratch = Scratch::new("replace");
        let store = EmbeddingStore::for_library(&scratch.0);
        let key = title_venue("A Title. Venue");

        store.put(&key, &entry(&["old"]));
        store.put(&key, &entry(&["new"]));

        assert_eq!(store.get(&key), Some(entry(&["new"])));
        assert_eq!(scratch.entry_files().len(), 1, "no temp files left behind");
    }

    #[test]
    fn any_change_in_a_key_is_a_miss() {
        let scratch = Scratch::new("keys");
        let store = EmbeddingStore::for_library(&scratch.0);
        store.put(&full_text("sha256-abc"), &entry(&["full"]));
        store.put(&title_venue("A Title. Venue"), &entry(&["title"]));

        let misses = [
            full_text("sha256-other"),
            EntryKey::FullText {
                model_id: "other-model".to_string(),
                artifact_hash: "sha256-abc".to_string(),
                extractor_version: 1,
                chunker_version: 1,
            },
            EntryKey::FullText {
                model_id: "ollama:nomic-embed-text".to_string(),
                artifact_hash: "sha256-abc".to_string(),
                extractor_version: 2,
                chunker_version: 1,
            },
            EntryKey::FullText {
                model_id: "ollama:nomic-embed-text".to_string(),
                artifact_hash: "sha256-abc".to_string(),
                extractor_version: 1,
                chunker_version: 2,
            },
            title_venue("A Title. Other Venue"),
            EntryKey::TitleVenue {
                model_id: "other-model".to_string(),
                text: "A Title. Venue".to_string(),
            },
        ];
        for key in &misses {
            assert_eq!(store.get(key), None, "{key:?}");
        }
        assert_eq!(scratch.entry_files().len(), 2);
    }

    #[test]
    fn title_venue_text_is_hashed_not_stored_in_the_key() {
        let scratch = Scratch::new("hashed-text");
        let store = EmbeddingStore::for_library(&scratch.0);
        store.put(
            &title_venue("A Distinctive Title"),
            &Entry { chunks: vec![] },
        );

        let files = scratch.entry_files();
        assert_eq!(files.len(), 1);
        let contents = std::fs::read_to_string(&files[0]).unwrap();
        assert!(!contents.contains("Distinctive"), "{contents}");
        assert!(contents.contains("text_sha256"), "{contents}");
    }

    #[test]
    fn the_first_put_writes_a_self_ignoring_gitignore() {
        let scratch = Scratch::new("gitignore");
        EmbeddingStore::for_library(&scratch.0).put(&full_text("h"), &entry(&["c"]));

        let gitignore = scratch.0.join("research/.pax-ai/.gitignore");
        assert_eq!(std::fs::read_to_string(gitignore).unwrap(), "*\n");
    }

    #[test]
    fn an_unknown_format_version_reads_as_none() {
        let scratch = Scratch::new("format");
        let store = EmbeddingStore::for_library(&scratch.0);
        let key = full_text("sha256-abc");
        store.put(&key, &entry(&["chunk"]));

        let path = scratch.entry_files().remove(0);
        let contents = std::fs::read_to_string(&path).unwrap();
        let bumped = contents.replacen("\"format\":1", "\"format\":2", 1);
        assert_ne!(contents, bumped);
        std::fs::write(&path, bumped).unwrap();

        assert_eq!(store.get(&key), None);
    }

    #[test]
    fn an_unreadable_or_mismatched_file_reads_as_none() {
        let scratch = Scratch::new("corrupt");
        let store = EmbeddingStore::for_library(&scratch.0);
        let key = full_text("sha256-abc");
        let other = full_text("sha256-other");
        store.put(&key, &entry(&["chunk"]));
        store.put(&other, &entry(&["other"]));
        let key_path = store.entry_path(&KeyRepr::from(&key).file_stem());
        let other_path = store.entry_path(&KeyRepr::from(&other).file_stem());

        // A file under `key`'s name that holds another key's entry.
        std::fs::copy(&other_path, &key_path).unwrap();
        assert_eq!(store.get(&key), None);

        std::fs::write(&key_path, b"{\"format\":1,").unwrap();
        assert_eq!(store.get(&key), None);
    }

    #[test]
    fn put_is_best_effort_when_the_directory_cant_be_created() {
        let scratch = Scratch::new("unwritable");
        std::fs::create_dir_all(&scratch.0).unwrap();
        // `research` is a file, so `research/.pax-ai/embeddings` can't exist.
        std::fs::write(scratch.0.join("research"), b"").unwrap();
        let store = EmbeddingStore::for_library(&scratch.0);
        let key = full_text("h");

        store.put(&key, &entry(&["c"]));

        assert_eq!(store.get(&key), None);
    }

    #[test]
    fn prune_deletes_only_entries_not_kept() {
        let scratch = Scratch::new("prune");
        let store = EmbeddingStore::for_library(&scratch.0);
        let kept = [full_text("keep"), title_venue("Kept. Venue")];
        let dropped = [full_text("drop"), title_venue("Dropped. Venue")];
        for key in kept.iter().chain(&dropped) {
            store.put(key, &entry(&["c"]));
        }
        let stray = scratch.embeddings_dir().join("notes.txt");
        std::fs::write(&stray, b"not an entry").unwrap();

        let keep: HashSet<EntryKey> = kept.iter().cloned().collect();
        assert_eq!(store.prune(&keep), 2);

        for key in &kept {
            assert!(store.get(key).is_some(), "{key:?}");
        }
        for key in &dropped {
            assert_eq!(store.get(key), None, "{key:?}");
        }
        assert!(stray.exists(), "non-entry files are left alone");
        assert_eq!(store.prune(&keep), 0);
    }

    #[test]
    fn prune_of_a_store_never_written_is_zero() {
        let scratch = Scratch::new("prune-empty");
        assert_eq!(
            EmbeddingStore::for_library(&scratch.0).prune(&HashSet::new()),
            0
        );
    }
}
