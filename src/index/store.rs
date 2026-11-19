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
//! # Keys
//!
//! A key is stored as a JSON object, e.g. `{ "kind": "full-text",
//! "model_id": "...", "artifact_hash": "...", "extractor_version": 1,
//! "chunker_version": 1 }`. A title + venue key is `{ "kind": "title-venue",
//! "model_id", "text_sha256" }`: the text itself never lands in the key. The
//! file name's stem is the SHA-256 of that key JSON, the same in both
//! formats. `get` checks the stored key, so a file whose key doesn't match
//! reads as a miss.
//!
//! # On-disk format, version 2
//!
//! `<sha256 of the key>.bin`. Every integer is little-endian; `u32` lengths
//! and counts, so no field exceeds 4 GiB:
//!
//! | bytes                | field                                        |
//! |----------------------|----------------------------------------------|
//! | 4                    | magic `PAXE`                                 |
//! | 4                    | `u32` format version, `2`                    |
//! | 4                    | `u32` key length `k`                         |
//! | `k`                  | the key JSON (UTF-8)                         |
//! | 4                    | `u32` chunk count `n`                        |
//! | 4                    | `u32` dimension `d` (`0` when `n` is `0`)    |
//! | `n` × (4 + len)      | each chunk's `u32` text length, then its UTF-8 text, in order |
//! | `n` × `d` × 4        | every vector's `d` `f32`s, chunk by chunk    |
//!
//! The vectors are raw `f32` bits, so they round-trip exactly and scores
//! don't change. Every chunk of an entry has the same dimension; `put` skips
//! (writes nothing for) an entry whose vectors differ in length. A file with
//! another magic or version, one that ends early, one with bytes left over,
//! or one whose text isn't UTF-8 reads as `None`.
//!
//! # Version 1 (read only)
//!
//! `<sha256 of the key>.json`, written before version 2: `{ "format": 1,
//! "key": { ... }, "chunks": [ { "text": "...", "vector": [0.1, ...] } ] }`.
//! `get` falls back to it when no `.bin` exists, so an upgrade never forces a
//! re-embed; `put` only writes version 2, and `prune` removes a `.json` once
//! a `.bin` for the same key exists.
//!
//! [pruned]: EmbeddingStore::prune

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The magic tag every version 2 entry starts with.
const MAGIC: [u8; 4] = *b"PAXE";

/// The version `put` writes. A `.bin` file carrying any other version reads
/// as `None`.
const FORMAT_VERSION: u32 = 2;

/// The version of the legacy JSON entries, still read by `get`.
const JSON_FORMAT_VERSION: u32 = 1;

/// The extension of a version 2 entry.
const BIN_EXT: &str = "bin";

/// The extension of a version 1 entry.
const JSON_EXT: &str = "json";

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

    /// The entry stored under `key`: the version 2 file, or else the version
    /// 1 one. Missing, unreadable, malformed, in an unknown format, or stored
    /// under a different key: `None`.
    pub fn get(&self, key: &EntryKey) -> Option<Entry> {
        let repr = KeyRepr::from(key);
        let stem = repr.file_stem();
        if let Ok(bytes) = std::fs::read(self.entry_path(&stem, BIN_EXT)) {
            return decode(&bytes, &repr);
        }
        let bytes = std::fs::read(self.entry_path(&stem, JSON_EXT)).ok()?;
        decode_json(&bytes, &repr)
    }

    /// Stores `entry` under `key` in version 2, replacing any previous one.
    ///
    /// Written to a sibling temp file and renamed into place, as in
    /// `SummaryCache::write_entry`, so a concurrent reader never sees a
    /// partial file. Best effort: the vectors were already computed, so a
    /// caching failure (no permission, disk full, ...) only costs a repeat
    /// embedding call next time.
    pub fn put(&self, key: &EntryKey, entry: &Entry) {
        let repr = KeyRepr::from(key);
        let Some(bytes) = encode(&repr, entry) else {
            return;
        };
        if std::fs::create_dir_all(&self.dir).is_err() {
            return;
        }
        self.write_gitignore();

        let stem = repr.file_stem();
        let tmp_path = self.dir.join(format!(
            "{stem}.tmp.{}.{}",
            std::process::id(),
            TMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        if std::fs::write(&tmp_path, bytes).is_ok()
            && std::fs::rename(&tmp_path, self.entry_path(&stem, BIN_EXT)).is_err()
        {
            let _ = std::fs::remove_file(&tmp_path);
        }
    }

    /// Deletes entries whose key isn't in `keep`, and version 1 files whose
    /// key also has a version 2 file; returns how many files it deleted.
    ///
    /// Only entry files (`<sha256>.bin` and `<sha256>.json`) are considered,
    /// so anything else in the directory, an in-flight temp file included, is
    /// left alone. A file that can't be deleted isn't counted.
    pub fn prune(&self, keep: &HashSet<EntryKey>) -> usize {
        let keep: HashSet<String> = keep.iter().map(|k| KeyRepr::from(k).file_stem()).collect();
        let Ok(dir) = std::fs::read_dir(&self.dir) else {
            return 0;
        };
        dir.filter_map(Result::ok)
            .filter(|e| {
                let name = e.file_name();
                let Some((stem, ext)) = name.to_str().and_then(|n| n.split_once('.')) else {
                    return false;
                };
                if !is_sha256_hex(stem) {
                    return false;
                }
                let doomed = match ext {
                    BIN_EXT => !keep.contains(stem),
                    JSON_EXT => !keep.contains(stem) || self.entry_path(stem, BIN_EXT).exists(),
                    _ => false,
                };
                doomed && std::fs::remove_file(e.path()).is_ok()
            })
            .count()
    }

    fn entry_path(&self, stem: &str, ext: &str) -> PathBuf {
        self.dir.join(format!("{stem}.{ext}"))
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
    /// This key's JSON: stable, since serde writes a struct's fields in
    /// declaration order.
    fn json(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("a key of strings and integers serializes")
    }

    /// The SHA-256 of this key's JSON.
    fn file_stem(&self) -> String {
        sha256_hex(&self.json())
    }
}

/// `entry` under `key` in version 2; `None` when its vectors differ in
/// length or a field doesn't fit a `u32`.
fn encode(key: &KeyRepr, entry: &Entry) -> Option<Vec<u8>> {
    let dim = entry.chunks.first().map_or(0, |c| c.vector.len());
    if entry.chunks.iter().any(|c| c.vector.len() != dim) {
        return None;
    }
    let key_json = key.json();
    let texts: usize = entry.chunks.iter().map(|c| 4 + c.text.len()).sum();
    let mut out = Vec::with_capacity(4 * 5 + key_json.len() + texts + entry.chunks.len() * dim * 4);
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    put_bytes(&mut out, &key_json)?;
    put_u32(&mut out, entry.chunks.len())?;
    put_u32(&mut out, dim)?;
    for chunk in &entry.chunks {
        put_bytes(&mut out, chunk.text.as_bytes())?;
    }
    for value in entry.chunks.iter().flat_map(|c| &c.vector) {
        out.extend_from_slice(&value.to_le_bytes());
    }
    Some(out)
}

fn put_u32(out: &mut Vec<u8>, n: usize) -> Option<()> {
    out.extend_from_slice(&u32::try_from(n).ok()?.to_le_bytes());
    Some(())
}

fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) -> Option<()> {
    put_u32(out, bytes.len())?;
    out.extend_from_slice(bytes);
    Some(())
}

/// A version 2 file's entry, if it is well formed and stored under `key`.
fn decode(bytes: &[u8], key: &KeyRepr) -> Option<Entry> {
    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC || r.u32()? != FORMAT_VERSION {
        return None;
    }
    let key_len = r.len()?;
    let stored: KeyRepr = serde_json::from_slice(r.take(key_len)?).ok()?;
    if stored != *key {
        return None;
    }
    let count = r.len()?;
    let dim = r.len()?;
    // Each chunk takes at least its 4-byte text length, which bounds the
    // allocation by the file's size, whatever `count` claims.
    if count > r.0.len() / 4 {
        return None;
    }
    let mut texts = Vec::with_capacity(count);
    for _ in 0..count {
        let len = r.len()?;
        texts.push(std::str::from_utf8(r.take(len)?).ok()?.to_owned());
    }
    let floats = r.take(count.checked_mul(dim)?.checked_mul(4)?)?;
    if !r.0.is_empty() {
        return None;
    }
    let stride = dim * 4;
    let chunks = texts
        .into_iter()
        .enumerate()
        .map(|(i, text)| StoredChunk {
            text,
            vector: floats[i * stride..(i + 1) * stride]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
        })
        .collect();
    Some(Entry { chunks })
}

/// A cursor over a version 2 file; every read past the end is `None`.
struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        if n > self.0.len() {
            return None;
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Some(head)
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn len(&mut self) -> Option<usize> {
        usize::try_from(self.u32()?).ok()
    }
}

/// A version 1 file's entry, if it parses and is stored under `key`.
fn decode_json(bytes: &[u8], key: &KeyRepr) -> Option<Entry> {
    let file: JsonEntryFile = serde_json::from_slice(bytes).ok()?;
    if file.format != JSON_FORMAT_VERSION || file.key != *key {
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

#[derive(Deserialize)]
struct JsonEntryFile {
    format: u32,
    key: KeyRepr,
    chunks: Vec<JsonChunk>,
}

#[derive(Deserialize)]
struct JsonChunk {
    text: String,
    vector: Vec<f32>,
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

    /// `n` chunks of 768 dimensions, as `nomic-embed-text`'s, each with about
    /// 1,200 bytes of text and pseudo-random values in `[-1, 1]`.
    fn realistic_entry(n: usize) -> Entry {
        let mut state: u32 = 0x9e37_79b9;
        Entry {
            chunks: (0..n)
                .map(|i| StoredChunk {
                    text: format!("chunk {i}: {}", "lorem ipsum dolor sit amet ".repeat(44)),
                    vector: (0..768)
                        .map(|_| {
                            state ^= state << 13;
                            state ^= state >> 17;
                            state ^= state << 5;
                            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
                        })
                        .collect(),
                })
                .collect(),
        }
    }

    /// Writes `entry` under `key` as a version 1 `.json` file, as the store
    /// did before version 2.
    fn put_v1(scratch: &Scratch, key: &EntryKey, entry: &Entry) -> PathBuf {
        let repr = KeyRepr::from(key);
        let chunks: Vec<_> = entry
            .chunks
            .iter()
            .map(|c| serde_json::json!({ "text": c.text, "vector": c.vector }))
            .collect();
        let json = serde_json::json!({ "format": 1, "key": repr, "chunks": chunks });
        std::fs::create_dir_all(scratch.embeddings_dir()).unwrap();
        let path = scratch
            .embeddings_dir()
            .join(format!("{}.json", repr.file_stem()));
        std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
        path
    }

    fn bin_path(scratch: &Scratch, key: &EntryKey) -> PathBuf {
        scratch
            .embeddings_dir()
            .join(format!("{}.bin", KeyRepr::from(key).file_stem()))
    }

    #[test]
    fn a_put_entry_reads_back_exactly_from_a_new_store() {
        let scratch = Scratch::new("roundtrip");
        let key = full_text("sha256-abc");
        let stored = entry(&["first chunk", "second \"chunk\"\n", "", "ünïcødé"]);

        EmbeddingStore::for_library(&scratch.0).put(&key, &stored);

        assert_eq!(
            scratch.entry_files(),
            vec![bin_path(&scratch, &key)],
            "put writes only a version 2 file"
        );
        assert_eq!(
            EmbeddingStore::for_library(&scratch.0).get(&key),
            Some(stored)
        );
    }

    #[test]
    fn a_version_2_round_trip_is_bit_exact() {
        let scratch = Scratch::new("bit-exact");
        let store = EmbeddingStore::for_library(&scratch.0);
        let key = full_text("sha256-abc");
        let mut stored = realistic_entry(3);
        // Values `==` can't tell apart, or can't compare at all.
        stored.chunks[0].vector[..6].copy_from_slice(&[
            -0.0,
            f32::from_bits(1),
            f32::NAN,
            f32::INFINITY,
            f32::MAX,
            f32::EPSILON,
        ]);

        store.put(&key, &stored);
        let read = store.get(&key).unwrap();

        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(read.chunks.len(), stored.chunks.len());
        for (r, s) in read.chunks.iter().zip(&stored.chunks) {
            assert_eq!(r.text, s.text);
            assert_eq!(bits(&r.vector), bits(&s.vector));
        }
    }

    #[test]
    fn an_entry_with_no_chunks_round_trips() {
        let scratch = Scratch::new("empty");
        let store = EmbeddingStore::for_library(&scratch.0);
        let key = title_venue("A Title. Venue");
        store.put(&key, &Entry { chunks: vec![] });
        assert_eq!(store.get(&key), Some(Entry { chunks: vec![] }));
    }

    #[test]
    fn an_entry_whose_vectors_differ_in_length_is_not_stored() {
        let scratch = Scratch::new("ragged");
        let store = EmbeddingStore::for_library(&scratch.0);
        let key = full_text("sha256-abc");
        let mut ragged = entry(&["a", "b"]);
        ragged.chunks[1].vector.pop();

        store.put(&key, &ragged);

        assert_eq!(store.get(&key), None);
        assert!(scratch.entry_files().is_empty());
    }

    /// Prints the on-disk size of one 768-dimension chunk in both versions;
    /// `cargo test smaller -- --nocapture` shows it.
    #[test]
    fn version_2_is_much_smaller_than_version_1() {
        let scratch = Scratch::new("sizes");
        let key = full_text("sha256-abc");
        let stored = realistic_entry(1);
        let v1 = put_v1(&scratch, &key, &stored);
        EmbeddingStore::for_library(&scratch.0).put(&key, &stored);

        let v1_len = std::fs::metadata(v1).unwrap().len();
        let v2_len = std::fs::metadata(bin_path(&scratch, &key)).unwrap().len();
        let text_len = stored.chunks[0].text.len();
        println!(
            "one 768-dim chunk, {text_len} B of text: v1 {v1_len} B, v2 {v2_len} B ({:.2}x smaller)",
            v1_len as f64 / v2_len as f64
        );
        assert!(v2_len < v1_len / 2, "v1 {v1_len} B, v2 {v2_len} B");
    }

    #[test]
    fn a_version_1_entry_is_still_read() {
        let scratch = Scratch::new("v1");
        let store = EmbeddingStore::for_library(&scratch.0);
        let full = full_text("sha256-abc");
        let title = title_venue("A Title. Venue");
        put_v1(&scratch, &full, &entry(&["full", "text"]));
        put_v1(&scratch, &title, &entry(&["title"]));

        assert_eq!(store.get(&full), Some(entry(&["full", "text"])));
        assert_eq!(store.get(&title), Some(entry(&["title"])));
    }

    #[test]
    fn a_version_2_entry_wins_over_a_version_1_one() {
        let scratch = Scratch::new("v2-wins");
        let store = EmbeddingStore::for_library(&scratch.0);
        let key = full_text("sha256-abc");
        put_v1(&scratch, &key, &entry(&["old"]));
        store.put(&key, &entry(&["new"]));

        assert_eq!(store.get(&key), Some(entry(&["new"])));
    }

    #[test]
    fn a_mismatched_malformed_or_unknown_version_1_file_reads_as_none() {
        let scratch = Scratch::new("v1-bad");
        let store = EmbeddingStore::for_library(&scratch.0);
        let key = full_text("sha256-abc");
        let other = full_text("sha256-other");
        let path = put_v1(&scratch, &key, &entry(&["chunk"]));
        let other_path = put_v1(&scratch, &other, &entry(&["other"]));

        // A file under `key`'s name that holds another key's entry.
        std::fs::copy(&other_path, &path).unwrap();
        assert_eq!(store.get(&key), None);

        put_v1(&scratch, &key, &entry(&["chunk"]));
        let contents = std::fs::read_to_string(&path).unwrap();
        let bumped = contents.replacen("\"format\":1", "\"format\":3", 1);
        assert_ne!(contents, bumped);
        std::fs::write(&path, bumped).unwrap();
        assert_eq!(store.get(&key), None);

        std::fs::write(&path, b"{\"format\":1,").unwrap();
        assert_eq!(store.get(&key), None);
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
    fn get_checks_the_stored_key() {
        let scratch = Scratch::new("key-check");
        let store = EmbeddingStore::for_library(&scratch.0);
        let key = full_text("sha256-abc");
        let other = full_text("sha256-other");
        store.put(&other, &entry(&["other"]));

        // A well-formed file under `key`'s name that holds another key's entry.
        std::fs::copy(bin_path(&scratch, &other), bin_path(&scratch, &key)).unwrap();

        assert_eq!(store.get(&key), None);
        assert_eq!(store.get(&other), Some(entry(&["other"])));
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
        let contents = String::from_utf8_lossy(&std::fs::read(&files[0]).unwrap()).into_owned();
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
    fn an_unknown_magic_or_version_reads_as_none() {
        let scratch = Scratch::new("format");
        let store = EmbeddingStore::for_library(&scratch.0);
        let key = full_text("sha256-abc");
        store.put(&key, &entry(&["chunk"]));
        let path = bin_path(&scratch, &key);
        let good = std::fs::read(&path).unwrap();

        let mut version = good.clone();
        version[4..8].copy_from_slice(&3u32.to_le_bytes());
        let mut magic = good.clone();
        magic[0] = b'X';
        for (what, bad) in [("version 3", version), ("another magic", magic)] {
            std::fs::write(&path, bad).unwrap();
            assert_eq!(store.get(&key), None, "{what}");
        }
    }

    #[test]
    fn a_truncated_or_padded_file_reads_as_none() {
        let scratch = Scratch::new("truncated");
        let store = EmbeddingStore::for_library(&scratch.0);
        let key = full_text("sha256-abc");
        store.put(&key, &entry(&["first", "second"]));
        let path = bin_path(&scratch, &key);
        let good = std::fs::read(&path).unwrap();

        for len in 0..good.len() {
            std::fs::write(&path, &good[..len]).unwrap();
            assert_eq!(store.get(&key), None, "truncated to {len} bytes");
        }
        let mut padded = good.clone();
        padded.push(0);
        std::fs::write(&path, padded).unwrap();
        assert_eq!(store.get(&key), None, "a trailing byte");
    }

    #[test]
    fn a_malformed_file_reads_as_none() {
        let scratch = Scratch::new("malformed");
        let store = EmbeddingStore::for_library(&scratch.0);
        let key = full_text("sha256-abc");
        store.put(&key, &entry(&["chunk"]));
        let path = bin_path(&scratch, &key);
        let good = std::fs::read(&path).unwrap();
        let key_len = u32::from_le_bytes(good[8..12].try_into().unwrap()) as usize;
        let counts = 12 + key_len;
        let text = counts + 8 + 4;

        let set_u32 = |at: usize, n: u32| {
            let mut bytes = good.clone();
            bytes[at..at + 4].copy_from_slice(&n.to_le_bytes());
            bytes
        };
        let mut bad_key_json = good.clone();
        bad_key_json[12] = b'[';
        let mut bad_utf8 = good.clone();
        bad_utf8[text] = 0xff;
        for (what, bad) in [
            ("key length past the end", set_u32(8, u32::MAX)),
            ("key not JSON", bad_key_json),
            ("chunk count past the end", set_u32(counts, u32::MAX)),
            ("dimension past the end", set_u32(counts + 4, u32::MAX)),
            ("text length past the end", set_u32(text - 4, u32::MAX)),
            ("text not UTF-8", bad_utf8),
            ("not an entry at all", b"{\"format\":1,".to_vec()),
        ] {
            std::fs::write(&path, bad).unwrap();
            assert_eq!(store.get(&key), None, "{what}");
        }
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
    fn prune_handles_both_versions() {
        let scratch = Scratch::new("prune-versions");
        let store = EmbeddingStore::for_library(&scratch.0);
        // Kept, with only a v1 file: it stays, since `get` still reads it.
        let v1_only = full_text("v1-only");
        // Kept, with both: the v1 file is superseded.
        let upgraded = full_text("upgraded");
        // Not kept: every file goes, whichever version.
        let dropped_v1 = full_text("dropped-v1");
        let dropped_both = title_venue("Dropped. Venue");

        let v1_only_path = put_v1(&scratch, &v1_only, &entry(&["v1"]));
        let upgraded_v1 = put_v1(&scratch, &upgraded, &entry(&["old"]));
        store.put(&upgraded, &entry(&["new"]));
        put_v1(&scratch, &dropped_v1, &entry(&["gone"]));
        put_v1(&scratch, &dropped_both, &entry(&["gone"]));
        store.put(&dropped_both, &entry(&["gone"]));
        let tmp = scratch.embeddings_dir().join(format!(
            "{}.tmp.1.2",
            KeyRepr::from(&dropped_v1).file_stem()
        ));
        std::fs::write(&tmp, b"in flight").unwrap();

        let keep: HashSet<EntryKey> = [v1_only.clone(), upgraded.clone()].into();
        assert_eq!(store.prune(&keep), 4);

        let mut expected = vec![v1_only_path, bin_path(&scratch, &upgraded), tmp];
        expected.sort();
        assert_eq!(scratch.entry_files(), expected);
        assert!(!upgraded_v1.exists());
        assert_eq!(store.get(&v1_only), Some(entry(&["v1"])));
        assert_eq!(store.get(&upgraded), Some(entry(&["new"])));
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
