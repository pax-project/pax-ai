use crate::error::PortError;

/// Produces a short prose summary of a paper's abstract or full text via an
/// LLM.
///
/// This is deliberately its own trait rather than one method on a combined
/// `LlmPort` alongside [`Embedder`]: a deployment's summarizer and embedder
/// don't have to be the same backend. Anthropic, the first adapter, has no
/// embeddings endpoint at all (see `adapter::anthropic`), so a single fat
/// trait would force one adapter to fake half of it.
#[allow(async_fn_in_trait)]
pub trait Summarizer {
    async fn summarize(&self, text: &str) -> Result<String, PortError>;
}

/// Embeds text into a dense vector for similarity search — related-paper
/// recommendations, full-text semantic search, and RAG retrieval all reduce
/// to nearest-neighbor lookups over vectors this produces.
///
/// See [`Summarizer`] for why this is a separate trait rather than folded
/// into one.
#[allow(async_fn_in_trait)]
pub trait Embedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, PortError>;
}
