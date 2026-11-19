//! Similarity over sets of embedding vectors.
//!
//! Every paper is a set of vectors: its title + venue vector plus its
//! full-text chunks (a fallback paper, with no PDF, is a set of one). These
//! are pure functions over L2-normalized vectors, so a cosine is a dot
//! product, and search is brute force in memory.

/// L2-normalizes `v` in place. A zero vector (or one with no elements) is
/// left as it is, rather than turned into NaNs.
pub fn normalize(v: &mut [f32]) {
    let norm = dot(v, v).sqrt();
    if norm > 0.0 && norm.is_finite() {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Inputs normalized. Mean over A of each chunk's best cosine in B, averaged with the
/// reverse direction. Either side empty → 0.0.
pub fn chunk_set_similarity(a: &[&[f32]], b: &[&[f32]]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    (mean_best_match(a, b) + mean_best_match(b, a)) / 2.0
}

/// Index and cosine of the chunk closest to `query`; `None` if `chunks` is empty.
///
/// Ties go to the earliest chunk, so the result is deterministic.
pub fn best_chunk(query: &[f32], chunks: &[&[f32]]) -> Option<(usize, f32)> {
    let mut best: Option<(usize, f32)> = None;
    for (i, chunk) in chunks.iter().enumerate() {
        let score = dot(query, chunk);
        match best {
            Some((_, top)) if score <= top => {}
            _ => best = Some((i, score)),
        }
    }
    best
}

/// Mean over `from` of each vector's best cosine in `to`. Both non-empty.
fn mean_best_match(from: &[&[f32]], to: &[&[f32]]) -> f32 {
    let total: f32 = from
        .iter()
        .filter_map(|v| best_chunk(v, to).map(|(_, score)| score))
        .sum();
    total / from.len() as f32
}

/// Dot product, which is the cosine for normalized vectors.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors of different dimensions");
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    fn unit(v: &[f32]) -> Vec<f32> {
        let mut v = v.to_vec();
        normalize(&mut v);
        v
    }

    fn refs(vs: &[Vec<f32>]) -> Vec<&[f32]> {
        vs.iter().map(Vec::as_slice).collect()
    }

    fn assert_close(a: f32, b: f32) {
        assert!((a - b).abs() < EPS, "{a} != {b}");
    }

    #[test]
    fn normalize_gives_unit_length() {
        let v = unit(&[3.0, 4.0]);
        assert_close(v[0], 0.6);
        assert_close(v[1], 0.8);
        assert_close(dot(&v, &v), 1.0);
    }

    #[test]
    fn normalize_leaves_zero_and_empty_vectors_alone() {
        let mut zero = [0.0, 0.0, 0.0];
        normalize(&mut zero);
        assert_eq!(zero, [0.0, 0.0, 0.0]);

        let mut empty: [f32; 0] = [];
        normalize(&mut empty);
    }

    #[test]
    fn normalize_is_idempotent() {
        let once = unit(&[1.0, 2.0, 2.0]);
        let twice = unit(&once);
        for (x, y) in once.iter().zip(&twice) {
            assert_close(*x, *y);
        }
    }

    #[test]
    fn empty_sets_score_zero() {
        let a = [unit(&[1.0, 0.0])];
        assert_eq!(chunk_set_similarity(&refs(&a), &[]), 0.0);
        assert_eq!(chunk_set_similarity(&[], &refs(&a)), 0.0);
        assert_eq!(chunk_set_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn identical_sets_score_one() {
        let a = [unit(&[1.0, 0.0, 0.0]), unit(&[0.0, 1.0, 1.0])];
        assert_close(chunk_set_similarity(&refs(&a), &refs(&a)), 1.0);
    }

    #[test]
    fn orthogonal_sets_score_zero() {
        let a = [unit(&[1.0, 0.0])];
        let b = [unit(&[0.0, 1.0])];
        assert_close(chunk_set_similarity(&refs(&a), &refs(&b)), 0.0);
    }

    #[test]
    fn similarity_is_symmetric() {
        let a = [unit(&[1.0, 0.2, 0.0]), unit(&[0.0, 1.0, 0.3])];
        let b = [
            unit(&[0.9, 0.1, 0.4]),
            unit(&[0.1, 0.1, 1.0]),
            unit(&[0.5, 0.5, 0.5]),
        ];
        let ab = chunk_set_similarity(&refs(&a), &refs(&b));
        let ba = chunk_set_similarity(&refs(&b), &refs(&a));
        assert_close(ab, ba);
    }

    #[test]
    fn averages_best_matches_in_both_directions() {
        // A: x. B: x, y. A→B: best of x is 1. B→A: x→1, y→0, mean 0.5.
        // Symmetric mean: (1 + 0.5) / 2 = 0.75.
        let a = [unit(&[1.0, 0.0])];
        let b = [unit(&[1.0, 0.0]), unit(&[0.0, 1.0])];
        assert_close(chunk_set_similarity(&refs(&a), &refs(&b)), 0.75);
    }

    #[test]
    fn set_of_one_matches_the_other_papers_title_venue_vector() {
        // A fallback paper is only its title + venue vector; a full-text paper
        // leads with its own, then its chunks. The fallback still finds its
        // like-for-like match in the other paper's title + venue vector.
        let fallback = [unit(&[1.0, 1.0, 0.0, 0.0])];
        let full = [
            unit(&[1.0, 0.9, 0.0, 0.0]),
            unit(&[0.0, 0.0, 1.0, 0.0]),
            unit(&[0.0, 0.0, 0.0, 1.0]),
        ];
        let score = chunk_set_similarity(&refs(&fallback), &refs(&full));
        let forward = dot(&fallback[0], &full[0]);
        let backward = forward / 3.0;
        assert_close(score, (forward + backward) / 2.0);
        assert_close(score, chunk_set_similarity(&refs(&full), &refs(&fallback)));
    }

    #[test]
    fn best_chunk_of_nothing_is_none() {
        assert_eq!(best_chunk(&unit(&[1.0, 0.0]), &[]), None);
    }

    #[test]
    fn best_chunk_picks_the_closest() {
        let chunks = [unit(&[0.0, 1.0]), unit(&[1.0, 0.1]), unit(&[1.0, 1.0])];
        let (i, score) = best_chunk(&unit(&[1.0, 0.0]), &refs(&chunks)).unwrap();
        assert_eq!(i, 1);
        assert_close(score, chunks[1][0]);
    }

    #[test]
    fn best_chunk_of_one_is_that_chunk() {
        let chunks = [unit(&[0.0, 1.0])];
        let (i, score) = best_chunk(&unit(&[1.0, 0.0]), &refs(&chunks)).unwrap();
        assert_eq!(i, 0);
        assert_close(score, 0.0);
    }

    #[test]
    fn best_chunk_breaks_ties_by_earliest() {
        let chunks = [unit(&[0.0, 1.0]), unit(&[1.0, 0.0]), unit(&[1.0, 0.0])];
        let (i, _) = best_chunk(&unit(&[1.0, 0.0]), &refs(&chunks)).unwrap();
        assert_eq!(i, 1);
    }

    #[test]
    fn best_chunk_keeps_negative_scores() {
        let chunks = [unit(&[-1.0, 0.0]), unit(&[-1.0, -1.0])];
        let (i, score) = best_chunk(&unit(&[1.0, 0.0]), &refs(&chunks)).unwrap();
        assert_eq!(i, 1);
        assert!(score < 0.0);
    }
}
