use serde::Deserialize;
use serde_json::json;

use crate::error::PortError;
use crate::port::{EmbedKind, Embedder};

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
///
/// Its [`Embedder::model_id`] is `ollama:<model>`. For `nomic-embed-text`
/// models it adds the `search_document: `/`search_query: ` prefix the model
/// expects, per [`EmbedKind`], and sends a whole batch in one `/api/embed`
/// request.
pub struct OllamaAdapter {
    client: reqwest::Client,
    base_url: String,
    model: String,
    model_id: String,
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
            model_id: model_id(DEFAULT_MODEL),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self.model_id = model_id(&self.model);
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
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn embed(&self, texts: &[&str], kind: EmbedKind) -> Result<Vec<Vec<f32>>, PortError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let input = prefixed(&self.model, texts, kind);
        let url = format!("{}/api/embed", self.base_url);

        let response = self
            .client
            .post(url)
            .json(&json!({
                "model": self.model,
                "input": input,
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

        extract_embeddings(body, texts.len())
    }
}

fn model_id(model: &str) -> String {
    format!("ollama:{model}")
}

/// `nomic-embed-text` (any tag, e.g. `nomic-embed-text:v1.5`) is trained with
/// task prefixes, and embeds documents and queries poorly without them.
fn is_nomic(model: &str) -> bool {
    model.starts_with("nomic-embed-text")
}

fn prefixed(model: &str, texts: &[&str], kind: EmbedKind) -> Vec<String> {
    let prefix = if is_nomic(model) {
        match kind {
            EmbedKind::Document => "search_document: ",
            EmbedKind::Query => "search_query: ",
        }
    } else {
        ""
    };
    texts.iter().map(|t| format!("{prefix}{t}")).collect()
}

#[derive(Debug, Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

fn extract_embeddings(
    response: EmbedResponse,
    expected: usize,
) -> Result<Vec<Vec<f32>>, PortError> {
    let got = response.embeddings.len();
    if got != expected {
        return Err(PortError::UnexpectedResponse(format!(
            "expected {expected} embeddings, got {got}"
        )));
    }
    Ok(response.embeddings)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response(json: &str) -> EmbedResponse {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn returns_one_embedding_per_input_in_order() {
        let body =
            response(r#"{"model": "nomic-embed-text", "embeddings": [[0.1, 0.2], [0.3, 0.4]]}"#);
        assert_eq!(
            extract_embeddings(body, 2).unwrap(),
            vec![vec![0.1, 0.2], vec![0.3, 0.4]]
        );
    }

    #[test]
    fn reports_unexpected_response_on_a_count_mismatch() {
        let empty = response(r#"{"model": "nomic-embed-text", "embeddings": []}"#);
        assert!(matches!(
            extract_embeddings(empty, 1),
            Err(PortError::UnexpectedResponse(_))
        ));
        let extra = response(r#"{"model": "nomic-embed-text", "embeddings": [[0.1], [0.2]]}"#);
        assert!(matches!(
            extract_embeddings(extra, 1),
            Err(PortError::UnexpectedResponse(_))
        ));
    }

    #[test]
    fn model_id_names_the_backend_and_model() {
        assert_eq!(OllamaAdapter::new().model_id(), "ollama:nomic-embed-text");
        assert_eq!(
            OllamaAdapter::new()
                .with_model("mxbai-embed-large")
                .model_id(),
            "ollama:mxbai-embed-large"
        );
    }

    #[test]
    fn nomic_models_get_the_task_prefix() {
        assert_eq!(
            prefixed("nomic-embed-text", &["a", "b"], EmbedKind::Document),
            vec!["search_document: a", "search_document: b"]
        );
        assert_eq!(
            prefixed("nomic-embed-text:v1.5", &["q"], EmbedKind::Query),
            vec!["search_query: q"]
        );
    }

    #[test]
    fn other_models_get_the_text_unchanged() {
        assert_eq!(
            prefixed("mxbai-embed-large", &["a"], EmbedKind::Query),
            vec!["a"]
        );
    }

    #[tokio::test]
    async fn an_empty_batch_makes_no_request() {
        // Unroutable base URL: any request would fail.
        let adapter = OllamaAdapter::new().with_base_url("http://127.0.0.1:0");
        let result = adapter.embed(&[], EmbedKind::Document).await;
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn embed_future_is_send() {
        fn assert_send<T: Send>(_: T) {}
        let adapter = OllamaAdapter::new();
        assert_send(adapter.embed(&["x"], EmbedKind::Document));
    }
}
