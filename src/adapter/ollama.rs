use serde::Deserialize;
use serde_json::json;

use crate::error::PortError;
use crate::port::Embedder;

const DEFAULT_BASE_URL: &str = "http://localhost:11434";
/// A small (~137M param), CPU-friendly embedding model — no GPU needed and
/// light enough to run alongside a normal dev machine's other work.
const DEFAULT_MODEL: &str = "nomic-embed-text";

/// [`Embedder`] backed by a locally running Ollama server.
///
/// The natural complement to [`crate::adapter::anthropic::AnthropicAdapter`]:
/// Anthropic has no embeddings endpoint, and a local model is a genuinely
/// low-resource way to fill that gap without a second paid vendor/API key.
/// Per `CLAUDE.md`'s port/adapter split, nothing above [`Embedder`] needs to
/// know this is what's providing it.
pub struct OllamaAdapter {
    client: reqwest::Client,
    base_url: String,
    model: String,
}

impl OllamaAdapter {
    /// Points at the default local Ollama server (`localhost:11434`) using
    /// `nomic-embed-text`. See `with_model`/`with_base_url` to override
    /// either.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: DEFAULT_BASE_URL.to_string(),
            model: DEFAULT_MODEL.to_string(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}

impl Default for OllamaAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl Embedder for OllamaAdapter {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, PortError> {
        let url = format!("{}/api/embed", self.base_url);

        let response = self
            .client
            .post(url)
            .json(&json!({
                "model": self.model,
                "input": text,
            }))
            .send()
            .await
            .map_err(|e| PortError::Request(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(PortError::Request(format!("HTTP {status}: {body}")));
        }

        let body: EmbedResponse = response
            .json()
            .await
            .map_err(|e| PortError::UnexpectedResponse(e.to_string()))?;

        extract_embedding(body)
    }
}

#[derive(Debug, Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

fn extract_embedding(response: EmbedResponse) -> Result<Vec<f32>, PortError> {
    response
        .embeddings
        .into_iter()
        .next()
        .ok_or_else(|| PortError::UnexpectedResponse("response contained no embedding".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_the_single_embedding_for_a_single_input() {
        let response: EmbedResponse = serde_json::from_str(
            r#"{"model": "nomic-embed-text", "embeddings": [[0.1, 0.2, 0.3]]}"#,
        )
        .unwrap();
        assert_eq!(extract_embedding(response).unwrap(), vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn reports_unexpected_response_when_embeddings_is_empty() {
        let response: EmbedResponse =
            serde_json::from_str(r#"{"model": "nomic-embed-text", "embeddings": []}"#).unwrap();
        assert!(matches!(
            extract_embedding(response),
            Err(PortError::UnexpectedResponse(_))
        ));
    }
}
