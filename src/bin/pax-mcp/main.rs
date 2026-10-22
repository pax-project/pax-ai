//! `pax-mcp [ROOT]`: serves the research library under ROOT (default: the
//! current directory) over MCP on stdin/stdout. Stdout carries the protocol,
//! so nothing else may print to it.

use std::path::PathBuf;

use pax_ai::mcp::PaxServer;
use pax_core::Config;
use rmcp::ServiceExt;
use rmcp::transport::stdio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    // Only the binary reads the environment; the library is handed a Config,
    // the same split pax-core's `pax` binary uses.
    let config = Config {
        semantic_scholar_api_key: std::env::var("SEMANTIC_SCHOLAR_API_KEY").ok(),
        arxiv_contact: std::env::var("ARXIV_CONTACT").ok(),
    };

    let service = PaxServer::new(root, config).serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}
