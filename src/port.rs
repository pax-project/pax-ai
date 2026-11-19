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

/// Whether text is embedded as a document to be searched or as a query to
/// search with. Some models (e.g. `nomic-embed-text`) embed the two
/// differently; the adapter handles that, so callers only say which it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedKind {
    Document,
    Query,
}

/// Embeds text into dense vectors for similarity search — related-paper
/// recommendations, full-text semantic search, and RAG retrieval all reduce
/// to nearest-neighbor lookups over vectors this produces.
///
/// See [`Summarizer`] for why this is a separate trait rather than folded
/// into one.
pub trait Embedder {
    /// Stable id of the backend + model, e.g. "ollama:nomic-embed-text". Part of every store key.
    fn model_id(&self) -> &str;
    /// One vector per input, in order; a count mismatch is `PortError::UnexpectedResponse`.
    /// The future is `Send`, so MCP tool handlers can await it.
    ///
    /// Vectors come back as the backend returns them, not normalized.
    fn embed(
        &self,
        texts: &[&str],
        kind: EmbedKind,
    ) -> impl Future<Output = Result<Vec<Vec<f32>>, PortError>> + Send;
}
