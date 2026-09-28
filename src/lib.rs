//! `pax-ai` is the shared AI substrate for the `pax` project: summaries,
//! semantic search, and RAG over a `pax-core` research library, plus an
//! MCP server exposing the same capabilities to external clients.
//!
//! The LLM/embedding capability is exposed as a port (see [`port`]) with
//! swappable adapters (see [`adapter`]) — see `CLAUDE.md` for the full
//! architecture rationale.

pub mod adapter;
pub mod error;
pub mod port;
pub mod summary;

pub use error::PortError;
pub use port::{Embedder, Summarizer};
pub use summary::SummaryCache;
