//! Mock LLM provider for testing.
//!
//! Returns configurable canned responses without making any API calls.
//! Activate via `provider = "mock"` in config.
//!
//! Config options:
//! - `model`: ignored, unless it contains ":" — then parsed as a one-shot
//!   tool call, format "tool_name:arg_json" (e.g.,
//!   "shell_execute:{\"command\":\"echo hello\"}"). The tool call is
//!   returned on the FIRST chat() only; subsequent calls return the canned
//!   text, so agent loops terminate.
//! - `base_url`: canned text response (default: "Mock response")
//!
//! Pass a [`CallRecorder`] via [`MockProvider::with_recorder`] to capture
//! every chat() invocation (system prompt, messages, tool names) for
//! assertion in integration tests.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::{
    ChatResponse, ContentBlock, LlmTool, Message, Provider, ProviderCapabilities, StopReason, Usage,
};
use crate::config::LlmConfig;
use crate::error::Result;

/// A single recorded chat() invocation.
#[derive(Debug, Clone)]
pub struct RecordedCall {
    pub system: String,
    pub messages: Vec<Message>,
    pub tool_names: Vec<String>,
}

/// Shared recorder for MockProvider calls. Clone it before handing the
/// provider to the system under test; read back with [`CallRecorder::calls`].
#[derive(Debug, Clone, Default)]
pub struct CallRecorder {
    calls: Arc<Mutex<Vec<RecordedCall>>>,
}

impl CallRecorder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot of all recorded calls so far.
    pub fn calls(&self) -> Vec<RecordedCall> {
        self.calls.lock().unwrap().clone()
    }

    fn record(&self, call: RecordedCall) {
        self.calls.lock().unwrap().push(call);
    }
}

pub struct MockProvider {
    response: String,
    tool_call: Option<(String, serde_json::Value)>,
    call_count: AtomicUsize,
    recorder: Option<CallRecorder>,
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
            call_count: AtomicUsize::new(0),
            recorder: None,
        })
    }

    /// Like [`MockProvider::new`], but records every chat() call into the
    /// given recorder for later assertion.
    pub fn with_recorder(config: &LlmConfig, recorder: CallRecorder) -> Result<Self> {
        let mut provider = Self::new(config)?;
        provider.recorder = Some(recorder);
        Ok(provider)
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
        system: &str,
        messages: &[Message],
        tools: &[LlmTool],
    ) -> Result<ChatResponse> {
        if let Some(ref recorder) = self.recorder {
            recorder.record(RecordedCall {
                system: system.to_string(),
                messages: messages.to_vec(),
                tool_names: tools.iter().map(|t| t.name.clone()).collect(),
            });
        }

        let call_index = self.call_count.fetch_add(1, Ordering::SeqCst);
        if let Some((ref name, ref args)) = self.tool_call {
            // One-shot: tool call on the first invocation only, so agent
            // loops (call tool -> feed result back) terminate with text.
            if call_index == 0 {
                return Ok(ChatResponse {
                    content: vec![ContentBlock::tool_use("mock_call_1", name, args.clone())],
                    stop_reason: StopReason::ToolUse,
                    usage: Usage {
                        input_tokens: 50,
                        output_tokens: 25,
                        ..Default::default()
                    },
                });
            }
        }

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
    async fn returns_tool_call_once_then_text() {
        let config = LlmConfig {
            provider: "mock".to_string(),
            model: "echo:{\"text\":\"hello\"}".to_string(),
            api_key_env: None,
            base_url: Some("done".to_string()),
            max_tokens: None,
            temperature: None,
            max_retries: None,
            initial_backoff_ms: None,
            max_backoff_ms: None,
        };
        let provider = MockProvider::new(&config).unwrap();

        let first = provider
            .chat("system", &[Message::user("test")], &[])
            .await
            .unwrap();
        assert_eq!(first.stop_reason, StopReason::ToolUse);
        let tool_uses = first.tool_uses();
        assert_eq!(tool_uses.len(), 1);
        assert_eq!(tool_uses[0].0, "mock_call_1");
        assert_eq!(tool_uses[0].1, "echo");

        let second = provider
            .chat("system", &[Message::user("test")], &[])
            .await
            .unwrap();
        assert_eq!(second.stop_reason, StopReason::EndTurn);
        assert_eq!(second.text(), "done");
    }

    #[tokio::test]
    async fn recorder_captures_calls() {
        let recorder = CallRecorder::new();
        let provider = MockProvider::with_recorder(&mock_config("hi"), recorder.clone()).unwrap();

        provider
            .chat("the system prompt", &[Message::user("hello")], &[])
            .await
            .unwrap();

        let calls = recorder.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].system, "the system prompt");
        assert_eq!(calls[0].messages.len(), 1);
    }
}
