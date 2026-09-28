use thiserror::Error;

/// Failure modes shared by every [`crate::port::Summarizer`] /
/// [`crate::port::Embedder`] adapter. Kept small and backend-agnostic on
/// purpose — a caller handling a `PortError` should never need to know
/// whether it came from a hosted API or a local model server.
#[derive(Debug, Error)]
pub enum PortError {
    #[error("request to LLM backend failed: {0}")]
    Request(String),
    #[error("LLM backend returned an unexpected response: {0}")]
    UnexpectedResponse(String),
    #[error("LLM backend declined the request: {0}")]
    Refused(String),
}
