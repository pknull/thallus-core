//! LLM Provider trait and implementations.
//!
//! All providers are compiled in; runtime selection via config.

mod anthropic;
mod claude_cli;
pub mod mock;
mod openai;
pub mod cache;
pub mod pricing;
pub mod retry;

pub use anthropic::AnthropicProvider;
pub use claude_cli::ClaudeCodeProvider;
pub use mock::MockProvider;
pub use openai::OpenAiCompatProvider;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::config::LlmConfig;
use crate::error::{CoreError, Result};
use crate::mcp::LlmTool;

/// Maximum length for API error response bodies in error messages.
/// Prevents leaking large response bodies that may contain sensitive information.
pub(crate) const MAX_ERROR_BODY_LENGTH: usize = 256;

/// Truncate an error response body to prevent leaking large/sensitive content.
pub(crate) fn truncate_error_body(body: &str) -> String {
    if body.len() <= MAX_ERROR_BODY_LENGTH {
        body.to_string()
    } else {
        // Find safe UTF-8 boundary for truncation
        let mut end = MAX_ERROR_BODY_LENGTH;
        while end > 0 && !body.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}... [truncated]", &body[..end])
    }
}

/// Provider capabilities.
#[derive(Debug, Clone, Default)]
pub struct ProviderCapabilities {
    pub supports_tools: bool,
    pub supports_vision: bool,
    pub supports_streaming: bool,
    pub max_tokens: Option<u32>,
}

/// Message role.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    System,
}

/// Message content block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        is_error: bool,
    },
}

impl ContentBlock {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    pub fn tool_use(
        id: impl Into<String>,
        name: impl Into<String>,
        input: serde_json::Value,
    ) -> Self {
        Self::ToolUse {
            id: id.into(),
            name: name.into(),
            input,
        }
    }

    pub fn tool_result(
        tool_use_id: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self::ToolResult {
            tool_use_id: tool_use_id.into(),
            content: content.into(),
            is_error,
        }
    }
}

/// Conversation message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

impl Message {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::text(text)],
        }
    }

    pub fn assistant(content: Vec<ContentBlock>) -> Self {
        Self {
            role: Role::Assistant,
            content,
        }
    }

    pub fn tool_results(results: Vec<ContentBlock>) -> Self {
        Self {
            role: Role::User,
            content: results,
        }
    }
}

/// Stop reason from LLM response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
}

impl StopReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::EndTurn => "end_turn",
            Self::ToolUse => "tool_use",
            Self::MaxTokens => "max_tokens",
            Self::StopSequence => "stop_sequence",
        }
    }
}

/// Chat response from provider.
#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    pub usage: Usage,
}

impl ChatResponse {
    /// Extract tool_use blocks from the response.
    pub fn tool_uses(&self) -> Vec<(&str, &str, &serde_json::Value)> {
        self.content
            .iter()
            .filter_map(|block| {
                if let ContentBlock::ToolUse { id, name, input } = block {
                    Some((id.as_str(), name.as_str(), input))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get text content from the response.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| {
                if let ContentBlock::Text { text } = block {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("")
    }
}

/// Token usage statistics.
///
/// Canonical representation across all providers. Each provider maps its
/// response format into these fields:
///
/// - **Anthropic**: `input_tokens` = uncached only, `cache_write_tokens` =
///   `cache_creation_input_tokens`, `cache_read_tokens` = `cache_read_input_tokens`
/// - **OpenAI**: `input_tokens` = `prompt_tokens - cached_tokens`,
///   `cache_read_tokens` = `prompt_tokens_details.cached_tokens`,
///   `reasoning_tokens` = `completion_tokens_details.reasoning_tokens`
/// - **OpenAI-compat (Ollama, vLLM, Groq)**: only `input_tokens` + `output_tokens`,
///   all other fields default to 0
#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    /// Input tokens NOT served from cache.
    pub input_tokens: u32,
    /// Generated output tokens.
    pub output_tokens: u32,
    /// Input tokens served from cache (discounted rate).
    pub cache_read_tokens: u32,
    /// Input tokens written to cache (may cost more than regular input).
    pub cache_write_tokens: u32,
    /// Internal chain-of-thought tokens (OpenAI o1/o3; billed as output).
    pub reasoning_tokens: u32,
}

impl TokenUsage {
    /// Total input tokens across all sources.
    pub fn total_input(&self) -> u32 {
        self.input_tokens
            .saturating_add(self.cache_read_tokens)
            .saturating_add(self.cache_write_tokens)
    }

    /// Total output tokens (reasoning tokens are a subset, not additive).
    pub fn total_output(&self) -> u32 {
        self.output_tokens
    }
}

/// Backward-compatible alias.
pub type Usage = TokenUsage;

/// Event emitted during streaming response generation.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Incremental text content.
    TextDelta(String),
    /// A tool_use block is starting.
    ToolUseStart { id: String, name: String },
    /// Incremental JSON input for an in-progress tool_use block.
    ToolInputDelta { id: String, json_chunk: String },
    /// Stream complete — contains the full accumulated response.
    Done(ChatResponse),
}

/// Callback type for streaming events.
pub type StreamCallback = dyn Fn(StreamEvent) + Send + Sync;

/// LLM Provider trait.
#[async_trait]
pub trait Provider: Send + Sync {
    /// Get provider name for metrics.
    fn name(&self) -> &str;

    /// Get provider capabilities.
    fn capabilities(&self) -> ProviderCapabilities;

    /// Send a chat completion request (non-streaming).
    async fn chat(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[LlmTool],
    ) -> Result<ChatResponse>;

    /// Stream a chat completion, calling `on_event` for each token/tool delta.
    ///
    /// Returns the final accumulated `ChatResponse` when the stream ends.
    /// Default implementation falls back to non-streaming `chat()`.
    async fn chat_stream(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[LlmTool],
        on_event: &StreamCallback,
    ) -> Result<ChatResponse> {
        let response = self.chat(system, messages, tools).await?;
        let text = response.text();
        if !text.is_empty() {
            on_event(StreamEvent::TextDelta(text));
        }
        on_event(StreamEvent::Done(response.clone()));
        Ok(response)
    }
}

/// Create a provider from configuration.
pub fn create_provider(config: &LlmConfig) -> Result<Box<dyn Provider>> {
    match config.provider.as_str() {
        "anthropic" => Ok(Box::new(AnthropicProvider::new(config)?)),
        "openai" | "ollama" | "openai-compat" => Ok(Box::new(OpenAiCompatProvider::new(config)?)),
        "claude-code" => Ok(Box::new(ClaudeCodeProvider::new(config)?)),
        "mock" => Ok(Box::new(MockProvider::new(config)?)),
        other => Err(CoreError::Provider {
            reason: format!("unknown provider: {}", other),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_construction() {
        let msg = Message::user("Hello");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content.len(), 1);
    }

    #[test]
    fn tool_use_extraction() {
        let response = ChatResponse {
            content: vec![
                ContentBlock::text("Let me check that"),
                ContentBlock::tool_use(
                    "call_1",
                    "shell_execute",
                    serde_json::json!({"command": "ls"}),
                ),
            ],
            stop_reason: StopReason::ToolUse,
            usage: Usage::default(),
        };

        let uses = response.tool_uses();
        assert_eq!(uses.len(), 1);
        assert_eq!(uses[0].1, "shell_execute");
    }
}
