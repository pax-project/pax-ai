//! Splitting a paper's full text into chunks to embed.
//!
//! Chunks are measured in characters, not tokens, so chunking needs no
//! tokenizer and doesn't depend on the embedding model. At `MAX_CHARS`
//! (about 375 tokens of English prose at ~4 characters a token) a chunk sits
//! far inside `nomic-embed-text`'s 8192-token context, and stays inside a
//! 2048-token one even for text that tokenizes badly (formulas, symbols).
//! Chunks that small also keep a search hit (#8) or a RAG passage (#9)
//! focused on one passage of the paper.

/// Part of every full-text store key: bump it when a change here alters the
/// chunks cut from the same text, so old entries read as misses.
pub const CHUNKER_VERSION: u32 = 1;

/// The most characters in a chunk.
const MAX_CHARS: usize = 1500;

/// About how many characters of the end of one chunk start the next, so a
/// passage cut by a chunk boundary is whole in one of the two.
const OVERLAP_CHARS: usize = 200;

/// A chunk ends at the last sentence end past this many characters, rather
/// than mid-sentence at `MAX_CHARS`.
const MIN_SENTENCE_CUT: usize = MAX_CHARS / 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub text: String,
}

/// Deterministic: the same text always gives the same chunks. Each chunk fits the
/// embedding model's context. An empty text gives no chunks.
///
/// Whitespace is collapsed to single spaces, so line breaks from the PDF
/// layout don't matter. A chunk is a run of whole words of at most
/// `MAX_CHARS`, ending at a sentence end when there's one past
/// `MIN_SENTENCE_CUT`; the next chunk starts up to `OVERLAP_CHARS` back. A
/// word longer than `MAX_CHARS` (e.g. a long URL) is cut into pieces.
pub fn chunk(text: &str) -> Vec<Chunk> {
    let words = words(text);
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < words.len() {
        let end = chunk_end(&words, start);
        chunks.push(Chunk {
            text: words[start..end].join(" "),
        });
        if end == words.len() {
            break;
        }
        start = overlap_start(&words, start, end);
    }
    chunks
}

/// The text's words, each at most `MAX_CHARS` long.
fn words(text: &str) -> Vec<&str> {
    let mut words = Vec::new();
    for word in text.split_whitespace() {
        let mut rest = word;
        while rest.chars().count() > MAX_CHARS {
            let (at, _) = rest.char_indices().nth(MAX_CHARS).unwrap();
            words.push(&rest[..at]);
            rest = &rest[at..];
        }
        words.push(rest);
    }
    words
}

/// One past the last word of the chunk starting at `start`.
fn chunk_end(words: &[&str], start: usize) -> usize {
    let mut len = 0;
    let mut end = start;
    let mut sentence_end = None;
    while end < words.len() {
        let added = words[end].chars().count() + usize::from(end > start);
        if len + added > MAX_CHARS {
            break;
        }
        len += added;
        end += 1;
        if len >= MIN_SENTENCE_CUT && ends_sentence(words[end - 1]) {
            sentence_end = Some(end);
        }
    }
    if end == words.len() {
        end
    } else {
        sentence_end.unwrap_or(end)
    }
}

/// The first word of the chunk after `[start, end)`: as far back from `end`
/// as fits in `OVERLAP_CHARS`, but always past `start`, so chunking advances.
fn overlap_start(words: &[&str], start: usize, end: usize) -> usize {
    let mut next = end;
    let mut len = 0;
    while next > start + 1 {
        let added = words[next - 1].chars().count() + usize::from(next < end);
        if len + added > OVERLAP_CHARS {
            break;
        }
        len += added;
        next -= 1;
    }
    next
}

fn ends_sentence(word: &str) -> bool {
    word.trim_end_matches(['"', '\'', ')', ']', '”', '’'])
        .ends_with(['.', '?', '!'])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `n` sentences of distinct numbered words, about 60 characters each.
    fn prose(n: usize) -> String {
        (0..n)
            .map(|i| format!("Sentence {i} says something about word{i} and more{i} here."))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The chunks' words with each overlap removed, i.e. the text's words if
    /// chunking lost and reordered nothing.
    fn rejoined(chunks: &[Chunk]) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for chunk in chunks {
            let words: Vec<&str> = chunk.text.split(' ').collect();
            let overlap = (0..=words.len().min(out.len()))
                .rev()
                .find(|&k| {
                    out[out.len() - k..]
                        .iter()
                        .zip(&words[..k])
                        .all(|(a, b)| a == b)
                })
                .unwrap();
            out.extend(words[overlap..].iter().map(|w| w.to_string()));
        }
        out
    }

    #[test]
    fn empty_text_gives_no_chunks() {
        assert!(chunk("").is_empty());
        assert!(chunk(" \n\t\x0c ").is_empty());
    }

    #[test]
    fn short_text_is_one_chunk_with_whitespace_collapsed() {
        assert_eq!(
            chunk("  A short\n\nabstract.\t "),
            vec![Chunk {
                text: "A short abstract.".to_string()
            }]
        );
    }

    #[test]
    fn is_deterministic() {
        let text = prose(200);
        assert_eq!(chunk(&text), chunk(&text));
    }

    #[test]
    fn every_chunk_fits() {
        let text = prose(200);
        let chunks = chunk(&text);
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(chunk.text.chars().count() <= MAX_CHARS);
            assert!(!chunk.text.is_empty());
        }
    }

    #[test]
    fn loses_and_reorders_nothing() {
        let text = prose(200);
        let words: Vec<String> = text.split_whitespace().map(str::to_string).collect();
        assert_eq!(rejoined(&chunk(&text)), words);
    }

    #[test]
    fn consecutive_chunks_overlap() {
        let chunks = chunk(&prose(200));
        for pair in chunks.windows(2) {
            let last_word = pair[0].text.rsplit(' ').next().unwrap();
            assert!(pair[1].text.contains(last_word), "{pair:?}");
            let first_word = pair[1].text.split(' ').next().unwrap();
            let tail = &pair[0].text[pair[0].text.len() - OVERLAP_CHARS..];
            assert!(tail.contains(first_word), "{pair:?}");
        }
    }

    #[test]
    fn chunks_end_at_sentence_ends() {
        let chunks = chunk(&prose(200));
        for chunk in &chunks[..chunks.len() - 1] {
            assert!(chunk.text.ends_with("here."), "{chunk:?}");
        }
    }

    #[test]
    fn text_without_sentence_ends_is_cut_at_the_limit() {
        let text = (0..1000)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let chunks = chunk(&text);
        assert!(chunks.len() > 1);
        assert!(chunks[0].text.chars().count() > MAX_CHARS - 10);
        assert_eq!(
            rejoined(&chunks),
            text.split(' ').map(str::to_string).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_word_longer_than_a_chunk_is_cut() {
        let long = "x".repeat(MAX_CHARS * 2 + 10);
        let chunks = chunk(&format!("before {long} after"));
        for chunk in &chunks {
            assert!(chunk.text.chars().count() <= MAX_CHARS);
        }
        let all: String = chunks.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(all.matches('x').count(), long.len());
        assert!(chunks[0].text.starts_with("before"));
        assert!(chunks.last().unwrap().text.ends_with("after"));
    }

    #[test]
    fn counts_characters_not_bytes() {
        let text = "é".repeat(MAX_CHARS);
        assert_eq!(chunk(&text).len(), 1);
    }
}
