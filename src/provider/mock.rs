//! Mock LLM provider for testing.
//!
//! Returns configurable canned responses without making any API calls.
//! Activate via `provider = "mock"` in config.
//!
//! Config options:
//! - `model`: ignored (accepted for compatibility)
//! - `mock_response`: canned text response (default: "Mock response")
//! - `mock_tool_call`: if set, returns a tool_use block instead of text
//!   Format: "tool_name:arg_json" (e.g., "shell_execute:{\"command\":\"echo hello\"}")

use async_trait::async_trait;

use super::{
    ChatResponse, ContentBlock, LlmTool, Message, Provider, ProviderCapabilities, StopReason,
    Usage,
};
use crate::config::LlmConfig;
use crate::error::Result;

pub struct MockProvider {
    response: String,
    tool_call: Option<(String, serde_json::Value)>,
}

impl MockProvider {
    pub fn new(config: &LlmConfig) -> Result<Self> {
        let response = config
            .base_url
            .as_deref()
            .unwrap_or("Mock response")
            .to_string();

        // Parse tool call from model field if it contains ":"
        // Format: "tool_name:{\"arg\":\"value\"}"
        let tool_call = if config.model.contains(':') {
            let mut parts = config.model.splitn(2, ':');
            let name = parts.next().unwrap_or("").to_string();
            let args_str = parts.next().unwrap_or("{}");
            let args: serde_json::Value =
                serde_json::from_str(args_str).unwrap_or(serde_json::json!({}));
            Some((name, args))
        } else {
            None
        };

        Ok(Self {
            response,
            tool_call,
        })
    }
}

#[async_trait]
impl Provider for MockProvider {
    fn name(&self) -> &str {
        "mock"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supports_tools: true,
            supports_vision: false,
            supports_streaming: false,
            max_tokens: Some(4096),
        }
    }

    async fn chat(
        &self,
        _system: &str,
        _messages: &[Message],
        _tools: &[LlmTool],
    ) -> Result<ChatResponse> {
        if let Some((ref name, ref args)) = self.tool_call {
            Ok(ChatResponse {
                content: vec![ContentBlock::tool_use("mock_call_1", name, args.clone())],
                stop_reason: StopReason::ToolUse,
                usage: Usage {
                    input_tokens: 50,
                    output_tokens: 25,
                    ..Default::default()
                },
            })
        } else {
            Ok(ChatResponse {
                content: vec![ContentBlock::Text {
                    text: self.response.clone(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: Usage {
                    input_tokens: 50,
                    output_tokens: 25,
                    ..Default::default()
                },
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_config(response: &str) -> LlmConfig {
        LlmConfig {
            provider: "mock".to_string(),
            model: "mock".to_string(),
            api_key_env: None,
            base_url: Some(response.to_string()),
            max_tokens: None,
            temperature: None,
            max_retries: None,
            initial_backoff_ms: None,
            max_backoff_ms: None,
        }
    }

    #[tokio::test]
    async fn returns_canned_text() {
        let provider = MockProvider::new(&mock_config("Hello from mock")).unwrap();
        let response = provider
            .chat("system", &[Message::user("hi")], &[])
            .await
            .unwrap();
        assert_eq!(response.text(), "Hello from mock");
        assert_eq!(response.usage.input_tokens, 50);
    }

    #[tokio::test]
    async fn returns_tool_call() {
        let config = LlmConfig {
            provider: "mock".to_string(),
            model: "echo:{\"text\":\"hello\"}".to_string(),
            api_key_env: None,
            base_url: None,
            max_tokens: None,
            temperature: None,
            max_retries: None,
            initial_backoff_ms: None,
            max_backoff_ms: None,
        };
        let provider = MockProvider::new(&config).unwrap();
        let response = provider
            .chat("system", &[Message::user("test")], &[])
            .await
            .unwrap();
        assert_eq!(response.stop_reason, StopReason::ToolUse);
        let tool_uses = response.tool_uses();
        assert_eq!(tool_uses.len(), 1);
        assert_eq!(tool_uses[0].0, "mock_call_1");
        assert_eq!(tool_uses[0].1, "echo");
    }
}
