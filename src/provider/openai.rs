//! OpenAI-compatible provider (works with OpenAI, Ollama, vLLM, etc.)

use async_trait::async_trait;

use super::{
    truncate_error_body, ChatResponse, ContentBlock, Message, Provider, ProviderCapabilities, Role,
    StopReason, Usage,
};
use crate::config::LlmConfig;
use crate::error::{CoreError, Result};
use crate::mcp::LlmTool;

/// OpenAI-compatible provider (works with OpenAI, Ollama, vLLM, etc.)
pub struct OpenAiCompatProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    model: String,
    max_tokens: u32,
}

impl OpenAiCompatProvider {
    pub fn new(config: &LlmConfig) -> Result<Self> {
        let base_url = match config.provider.as_str() {
            "ollama" => config
                .base_url
                .clone()
                .unwrap_or_else(|| "http://localhost:11434/v1".to_string()),
            "openai" => "https://api.openai.com/v1".to_string(),
            "openai-compat" => config
                .base_url
                .clone()
                .ok_or_else(|| CoreError::Config {
                    reason: "openai-compat requires base_url".into(),
                })?,
            _ => {
                return Err(CoreError::Config {
                    reason: format!("unsupported provider: {}", config.provider),
                })
            }
        };

        let api_key = config
            .api_key_env
            .as_ref()
            .and_then(|env| std::env::var(env).ok());

        Ok(Self {
            client: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            model: config.model.clone(),
            max_tokens: config.max_tokens.unwrap_or(4096),
        })
    }
}

#[async_trait]
impl Provider for OpenAiCompatProvider {
    fn name(&self) -> &str {
        "openai-compat"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supports_tools: true,
            supports_vision: false,
            supports_streaming: true,
            max_tokens: Some(self.max_tokens),
        }
    }

    async fn chat(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[LlmTool],
    ) -> Result<ChatResponse> {
        let url = format!("{}/chat/completions", self.base_url);

        // Convert messages to OpenAI format
        let mut openai_messages: Vec<serde_json::Value> = vec![serde_json::json!({
            "role": "system",
            "content": system
        })];

        for msg in messages {
            let converted = convert_message_to_openai(msg);
            openai_messages.extend(converted);
        }

        let mut body = serde_json::json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": openai_messages,
        });

        if !tools.is_empty() {
            let openai_tools: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.input_schema,
                        }
                    })
                })
                .collect();
            body["tools"] = serde_json::Value::Array(openai_tools);
        }

        let mut request = self.client.post(&url).json(&body);

        if let Some(ref api_key) = self.api_key {
            request = request.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = request.send().await.map_err(|e| CoreError::Provider {
            reason: format!("request failed: {}", e),
        })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(CoreError::Provider {
                reason: format!("API error {}: {}", status, truncate_error_body(&body)),
            });
        }

        let response_body: serde_json::Value =
            response.json().await.map_err(|e| CoreError::Provider {
                reason: format!("failed to parse response: {}", e),
            })?;

        // Parse OpenAI response
        let choice = &response_body["choices"][0];
        let message = &choice["message"];

        let mut content = Vec::new();

        // Text content
        if let Some(text) = message["content"].as_str() {
            if !text.is_empty() {
                content.push(ContentBlock::text(text));
            }
        }

        // Tool calls
        if let Some(tool_calls) = message["tool_calls"].as_array() {
            for call in tool_calls {
                let id = call["id"].as_str().unwrap_or("").to_string();
                let name = call["function"]["name"].as_str().unwrap_or("").to_string();
                let arguments: serde_json::Value = call["function"]["arguments"]
                    .as_str()
                    .and_then(|s| serde_json::from_str(s).ok())
                    .unwrap_or(serde_json::json!({}));
                content.push(ContentBlock::tool_use(id, name, arguments));
            }
        }

        let stop_reason = match choice["finish_reason"].as_str() {
            Some("stop") => StopReason::EndTurn,
            Some("tool_calls") => StopReason::ToolUse,
            Some("length") => StopReason::MaxTokens,
            _ => StopReason::EndTurn,
        };

        let usage = Usage {
            input_tokens: response_body["usage"]["prompt_tokens"]
                .as_u64()
                .unwrap_or(0) as u32,
            output_tokens: response_body["usage"]["completion_tokens"]
                .as_u64()
                .unwrap_or(0) as u32,
        };

        Ok(ChatResponse {
            content,
            stop_reason,
            usage,
        })
    }
}

/// Convert a Message to one or more OpenAI Chat Completions API messages.
///
/// OpenAI format differences from Anthropic:
/// - Assistant tool calls go in `tool_calls` array on the message
/// - Tool results are separate messages with `role: "tool"` and `tool_call_id`
fn convert_message_to_openai(msg: &Message) -> Vec<serde_json::Value> {
    let role = match msg.role {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::System => "system",
    };

    // Separate content types
    let mut texts: Vec<&str> = Vec::new();
    let mut tool_calls: Vec<serde_json::Value> = Vec::new();
    let mut tool_results: Vec<(String, String, bool)> = Vec::new();

    for block in &msg.content {
        match block {
            ContentBlock::Text { text } => {
                texts.push(text.as_str());
            }
            ContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(serde_json::json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": serde_json::to_string(input).unwrap_or_default()
                    }
                }));
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                tool_results.push((tool_use_id.clone(), content.clone(), *is_error));
            }
        }
    }

    let mut messages = Vec::new();

    // Build the main message (if it has text or tool_calls)
    if !texts.is_empty() || !tool_calls.is_empty() {
        let text_content = if texts.is_empty() {
            serde_json::Value::Null
        } else if texts.len() == 1 {
            serde_json::Value::String(texts[0].to_string())
        } else {
            serde_json::Value::String(texts.join("\n"))
        };

        let mut message = serde_json::json!({
            "role": role,
            "content": text_content,
        });

        // Add tool_calls for assistant messages
        if !tool_calls.is_empty() && role == "assistant" {
            message["tool_calls"] = serde_json::Value::Array(tool_calls);
        }

        messages.push(message);
    }

    // Tool results become separate "tool" role messages
    for (tool_call_id, content, is_error) in tool_results {
        let result_content = if is_error {
            format!("Error: {}", content)
        } else {
            content
        };
        messages.push(serde_json::json!({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result_content,
        }));
    }

    // If no messages were generated (empty content), create a placeholder
    if messages.is_empty() {
        messages.push(serde_json::json!({
            "role": role,
            "content": "",
        }));
    }

    messages
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert_message_text_only() {
        let msg = Message::user("Hello, world!");
        let converted = convert_message_to_openai(&msg);

        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0]["role"], "user");
        assert_eq!(converted[0]["content"], "Hello, world!");
    }

    #[test]
    fn convert_message_assistant_with_tool_call() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::text("Let me check that for you."),
                ContentBlock::tool_use(
                    "call_123",
                    "shell_execute",
                    serde_json::json!({"command": "ls -la"}),
                ),
            ],
        };
        let converted = convert_message_to_openai(&msg);

        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0]["role"], "assistant");
        assert_eq!(converted[0]["content"], "Let me check that for you.");

        let tool_calls = converted[0]["tool_calls"].as_array().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0]["id"], "call_123");
        assert_eq!(tool_calls[0]["type"], "function");
        assert_eq!(tool_calls[0]["function"]["name"], "shell_execute");
    }

    #[test]
    fn convert_message_tool_result() {
        let msg = Message {
            role: Role::User,
            content: vec![ContentBlock::tool_result(
                "call_123",
                "file1.txt\nfile2.txt",
                false,
            )],
        };
        let converted = convert_message_to_openai(&msg);

        // Tool results become separate "tool" role messages
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0]["role"], "tool");
        assert_eq!(converted[0]["tool_call_id"], "call_123");
        assert_eq!(converted[0]["content"], "file1.txt\nfile2.txt");
    }

    #[test]
    fn convert_message_tool_result_error() {
        let msg = Message {
            role: Role::User,
            content: vec![ContentBlock::tool_result(
                "call_456",
                "Command not found",
                true,
            )],
        };
        let converted = convert_message_to_openai(&msg);

        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0]["role"], "tool");
        assert_eq!(converted[0]["content"], "Error: Command not found");
    }

    #[test]
    fn convert_message_mixed_content() {
        // Simulates a multi-turn conversation turn with text and tool results
        let msg = Message {
            role: Role::User,
            content: vec![
                ContentBlock::text("Here are the results:"),
                ContentBlock::tool_result("call_1", "output 1", false),
                ContentBlock::tool_result("call_2", "output 2", false),
            ],
        };
        let converted = convert_message_to_openai(&msg);

        // Should produce: 1 user message with text + 2 tool messages
        assert_eq!(converted.len(), 3);
        assert_eq!(converted[0]["role"], "user");
        assert_eq!(converted[0]["content"], "Here are the results:");
        assert_eq!(converted[1]["role"], "tool");
        assert_eq!(converted[1]["tool_call_id"], "call_1");
        assert_eq!(converted[2]["role"], "tool");
        assert_eq!(converted[2]["tool_call_id"], "call_2");
    }
}
