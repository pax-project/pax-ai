//! Scoring provider candidates, papers that aren't in the library, against
//! one indexed library paper (#65).
//!
//! A candidate is embedded on the spot from its title and abstract (its
//! title alone when it has none) and never stored: it isn't in
//! `papers.nix`, so it doesn't belong in the store. It scores by its best
//! cosine match among the seed's vector set, with the candidate in the query
//! position (`best_chunk`), one direction only. These scores rank
//! candidates against this seed; they aren't comparable with
//! [`PaperIndex::similar`](super::PaperIndex::similar)'s chunk-set scores.

use pax_core::CandidateWork;

use super::{EMBED_BATCH, PaperVectors, similarity};
use crate::error::PortError;
use crate::port::{EmbedKind, Embedder};

/// How one candidate scored against the seed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CandidateScore {
    /// The candidate's best cosine among the seed's vectors; higher is
    /// closer, at most 1.
    pub score: f32,
    /// Whether its abstract was embedded with its title.
    pub had_abstract: bool,
}

/// The text a candidate is embedded from: its title, then a blank line and
/// its abstract when it has a non-blank one; and whether it had one.
pub fn candidate_text(work: &CandidateWork) -> (String, bool) {
    let title = work.title.trim();
    match work.abstract_text.as_deref().map(str::trim) {
        Some(text) if !text.is_empty() => (format!("{title}\n\n{text}"), true),
        _ => (title.to_string(), false),
    }
}

/// Each of `works` scored against `seed`, in the order given.
///
/// The candidates are embedded as documents, in batches; any embedding
/// error fails the whole call. `seed` must be indexed (non-empty `chunks`):
/// against an empty vector set every candidate scores 0.
pub async fn score_candidates(
    embedder: &impl Embedder,
    seed: &PaperVectors,
    works: &[CandidateWork],
) -> Result<Vec<CandidateScore>, PortError> {
    let texts: Vec<(String, bool)> = works.iter().map(candidate_text).collect();
    let seed_set: Vec<&[f32]> = seed.chunks.iter().map(|c| c.vector.as_slice()).collect();
    let mut scores = Vec::with_capacity(works.len());
    for batch in texts.chunks(EMBED_BATCH) {
        let inputs: Vec<&str> = batch.iter().map(|(text, _)| text.as_str()).collect();
        let vectors = embedder.embed(&inputs, EmbedKind::Document).await?;
        if vectors.len() != inputs.len() {
            return Err(PortError::UnexpectedResponse(format!(
                "the embedder returned {} vectors for {} texts",
                vectors.len(),
                inputs.len()
            )));
        }
        for ((_, had_abstract), mut vector) in batch.iter().zip(vectors) {
            similarity::normalize(&mut vector);
            let score = similarity::best_chunk(&vector, &seed_set).map_or(0.0, |(_, score)| score);
            scores.push(CandidateScore {
                score,
                had_abstract: *had_abstract,
            });
        }
    }
    Ok(scores)
}

/// The indices of `scores`, best first; equal scores keep the order given.
pub fn best_first(scores: &[CandidateScore]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[b].score.total_cmp(&scores[a].score));
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::StoredChunk;
    use crate::index::tests::FakeEmbedder;
    use pax_core::{CandidateId, ProviderId};

    fn work(native_id: &str, title: &str, abstract_text: Option<&str>) -> CandidateWork {
        CandidateWork {
            id: CandidateId {
                provider: ProviderId::OpenAlex,
                native_id: native_id.to_string(),
            },
            title: title.to_string(),
            authors: Vec::new(),
            publish_date: String::new(),
            doi: None,
            pdf_url: None,
            venue: None,
            abstract_text: abstract_text.map(str::to_string),
        }
    }

    /// `text`'s normalized vector under [`FakeEmbedder`].
    async fn vector(text: &str) -> Vec<f32> {
        let mut v = FakeEmbedder::new()
            .embed(&[text], EmbedKind::Document)
            .await
            .unwrap()
            .remove(0);
        similarity::normalize(&mut v);
        v
    }

    async fn seed(texts: &[&str]) -> PaperVectors {
        let mut chunks = Vec::new();
        for text in texts {
            chunks.push(StoredChunk {
                text: text.to_string(),
                vector: vector(text).await,
            });
        }
        PaperVectors {
            citation_key: "a2020".to_string(),
            fallback: false,
            chunks,
        }
    }

    #[test]
    fn the_text_is_the_title_and_a_non_blank_abstract() {
        assert_eq!(
            candidate_text(&work("W1", " Actors ", Some(" Message passing. "))),
            ("Actors\n\nMessage passing.".to_string(), true)
        );
        assert_eq!(
            candidate_text(&work("W1", "Actors", Some("  "))),
            ("Actors".to_string(), false)
        );
        assert_eq!(
            candidate_text(&work("W1", "Actors", None)),
            ("Actors".to_string(), false)
        );
    }

    #[tokio::test]
    async fn a_candidate_scores_by_its_best_match_among_the_seeds_vectors() {
        let seed = seed(&["Attention", "Actors\n\nMessage passing."]).await;
        let works = [
            work("W1", "Unrelated title of some length", None),
            work("W2", "Actors", Some("Message passing.")),
            work("W3", "Attention", None),
        ];
        let embedder = FakeEmbedder::new();

        let scores = score_candidates(&embedder, &seed, &works).await.unwrap();

        assert_eq!(*embedder.kinds.lock().unwrap(), [EmbedKind::Document]);
        assert_eq!(embedder.embedded(), 3);
        let had: Vec<bool> = scores.iter().map(|s| s.had_abstract).collect();
        assert_eq!(had, [false, true, false]);
        assert!((scores[1].score - 1.0).abs() < 1e-5, "{scores:?}");
        assert!((scores[2].score - 1.0).abs() < 1e-5, "{scores:?}");
        assert!(scores[0].score < 1.0 - 1e-3, "{scores:?}");
        assert_eq!(best_first(&scores)[2], 0);
    }

    #[test]
    fn best_first_keeps_the_given_order_for_equal_scores() {
        let score = |score| CandidateScore {
            score,
            had_abstract: false,
        };
        assert_eq!(
            best_first(&[score(0.2), score(0.9), score(0.2), score(0.5)]),
            [1, 3, 0, 2]
        );
    }

    #[tokio::test]
    async fn an_embedding_error_fails_the_call() {
        let seed = seed(&["Attention"]).await;
        let works = [work("W1", "Attention", None), work("W2", "Down", None)];
        let embedder = FakeEmbedder::failing_on("Down");
        assert!(score_candidates(&embedder, &seed, &works).await.is_err());
    }

    #[tokio::test]
    async fn no_candidates_embed_nothing() {
        let seed = seed(&["Attention"]).await;
        let embedder = FakeEmbedder::new();
        let scores = score_candidates(&embedder, &seed, &[]).await.unwrap();
        assert!(scores.is_empty());
        assert!(embedder.kinds.lock().unwrap().is_empty());
    }
}
