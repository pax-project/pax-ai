use serde::Deserialize;
use serde_json::json;

use crate::error::PortError;
use crate::port::Summarizer;

const MESSAGES_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MODEL: &str = "claude-opus-5";
/// Non-streaming default recommended for Claude Opus 5 and later — low
/// enough to stay under SDK/HTTP timeouts, high enough that adaptive
/// thinking (on by default for this model) doesn't consume the whole
/// budget before any summary text is produced.
const MAX_TOKENS: u32 = 16_000;

/// [`Summarizer`] backed by Anthropic's hosted Messages API.
///
/// Anthropic has no embeddings endpoint of its own — see the org's
/// port/adapter notes in `CLAUDE.md` — so this adapter implements
/// [`Summarizer`] only. Embeddings are served by a separate, independently
/// chosen adapter (e.g. `adapter::ollama`).
pub struct AnthropicAdapter {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

impl AnthropicAdapter {
    /// Uses Anthropic's current flagship model. See `with_model` to pin a
    /// different one.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_model(api_key, DEFAULT_MODEL)
    }

    pub fn with_model(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model: model.into(),
        }
    }
}

impl Summarizer for AnthropicAdapter {
    async fn summarize(&self, text: &str) -> Result<String, PortError> {
        let prompt =
            format!("Summarize the following paper text in a few concise sentences:\n\n{text}");

        let response = self
            .client
            .post(MESSAGES_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            // Re-runs a policy-declined request on Anthropic's recommended
            // fallback model server-side instead of surfacing a refusal for
            // what is, from this adapter's caller's perspective, an
            // ordinary summarization request.
            .header("anthropic-beta", "server-side-fallback-2026-07-01")
            .json(&json!({
                "model": self.model,
                "max_tokens": MAX_TOKENS,
                "fallbacks": "default",
                "messages": [{"role": "user", "content": prompt}],
            }))
            .send()
            .await
            .map_err(|e| PortError::Request(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(PortError::Request(format!("HTTP {status}: {body}")));
        }

        let body: MessagesResponse = response
            .json()
            .await
            .map_err(|e| PortError::UnexpectedResponse(e.to_string()))?;

        extract_summary(body)
    }
}

#[derive(Debug, Deserialize)]
struct MessagesResponse {
    content: Vec<ContentBlock>,
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(other)]
    Other,
}

fn extract_summary(response: MessagesResponse) -> Result<String, PortError> {
    if response.stop_reason.as_deref() == Some("refusal") {
        return Err(PortError::Refused(
            "Anthropic declined the summarization request".to_string(),
        ));
    }

    response
        .content
        .into_iter()
        .find_map(|block| match block {
            ContentBlock::Text { text } => Some(text),
            ContentBlock::Other => None,
        })
        .ok_or_else(|| {
            PortError::UnexpectedResponse("response contained no text block".to_string())
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_text_from_a_normal_response() {
        let response: MessagesResponse = serde_json::from_str(
            r#"{
                "content": [{"type": "text", "text": "A concise summary."}],
                "stop_reason": "end_turn"
            }"#,
        )
        .unwrap();
        assert_eq!(extract_summary(response).unwrap(), "A concise summary.");
    }

    #[test]
    fn skips_non_text_blocks_such_as_a_fallback_marker() {
        let response: MessagesResponse = serde_json::from_str(
            r#"{
                "content": [
                    {"type": "fallback", "from": {"model": "claude-opus-5"}, "to": {"model": "claude-opus-4-8"}},
                    {"type": "text", "text": "Summary after fallback."}
                ],
                "stop_reason": "end_turn"
            }"#,
        )
        .unwrap();
        assert_eq!(
            extract_summary(response).unwrap(),
            "Summary after fallback."
        );
    }

    #[test]
    fn reports_a_refusal_distinctly_from_a_missing_text_block() {
        let response: MessagesResponse =
            serde_json::from_str(r#"{"content": [], "stop_reason": "refusal"}"#).unwrap();
        assert!(matches!(
            extract_summary(response),
            Err(PortError::Refused(_))
        ));
    }

    #[test]
    fn reports_unexpected_response_when_no_text_block_and_not_refused() {
        let response: MessagesResponse =
            serde_json::from_str(r#"{"content": [], "stop_reason": "end_turn"}"#).unwrap();
        assert!(matches!(
            extract_summary(response),
            Err(PortError::UnexpectedResponse(_))
        ));
    }
}
