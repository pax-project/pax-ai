//! The embedding index shared by related-paper recommendations, full-text
//! semantic search and RAG retrieval: each paper's full text, extracted from
//! its PDF and chunked, plus a title + venue vector (see #52).

pub mod chunk;
pub mod extract;
pub mod store;
