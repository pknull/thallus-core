//! OpenAI-compatible provider (works with OpenAI, Ollama, vLLM, etc.)

use async_trait::async_trait;
use futures::StreamExt;
use reqwest_eventsource::{Event, RequestBuilderExt};

use super::{
    retry::{self, RetryPolicy},
    ChatResponse, ContentBlock, Message, Provider, ProviderCapabilities, Role, StopReason,
    StreamCallback, StreamEvent, Usage,
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
    retry_policy: RetryPolicy,
}

impl OpenAiCompatProvider {
    pub fn new(config: &LlmConfig) -> Result<Self> {
        let base_url = match config.provider.as_str() {
            "ollama" => config
                .base_url
                .clone()
                .unwrap_or_else(|| "http://localhost:11434/v1".to_string()),
            "openai" => "https://api.openai.com/v1".to_string(),
            "openai-compat" => config.base_url.clone().ok_or_else(|| CoreError::Config {
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
            retry_policy: RetryPolicy::from_config(config),
        })
    }
}

impl OpenAiCompatProvider {
    /// Send a request with retry on transient failures.
    async fn send_with_retry(
        &self,
        url: &str,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        let auth_header = self.api_key.as_ref().map(|k| format!("Bearer {}", k));
        retry::send_json_with_retry(&self.retry_policy, "openai-compat", &self.client, || {
            let mut req = self.client.post(url).json(body);
            if let Some(ref auth) = auth_header {
                req = req.header("Authorization", auth);
            }
            req.build().map_err(|e| CoreError::Provider {
                reason: format!("failed to build request: {}", e),
            })
        })
        .await
    }

    /// Build the OpenAI-format request body.
    fn build_body(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[LlmTool],
        stream: bool,
    ) -> serde_json::Value {
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

        if stream {
            body["stream"] = serde_json::Value::Bool(true);
            // Required to receive usage in streaming responses
            body["stream_options"] = serde_json::json!({"include_usage": true});
        }

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

        body
    }

    /// Process an SSE stream from an OpenAI-compatible API.
    async fn process_stream(
        &self,
        body: serde_json::Value,
        on_event: &StreamCallback,
    ) -> Result<ChatResponse> {
        let url = format!("{}/chat/completions", self.base_url);

        let mut request = self.client.post(&url).json(&body);
        if let Some(ref api_key) = self.api_key {
            request = request.header("Authorization", format!("Bearer {}", api_key));
        }

        let mut es = request.eventsource().map_err(|e| CoreError::Provider {
            reason: format!("failed to create event source: {}", e),
        })?;

        let mut text_content = String::new();
        // tool_calls: indexed by position, each holds (id, name, arguments_json)
        let mut tool_calls: Vec<(String, String, String)> = Vec::new();
        let mut stop_reason = StopReason::EndTurn;
        let mut usage = Usage::default();

        while let Some(event) = es.next().await {
            match event {
                Ok(Event::Open) => {}
                Ok(Event::Message(msg)) => {
                    if msg.data == "[DONE]" {
                        es.close();
                        break;
                    }

                    let data: serde_json::Value = match serde_json::from_str(&msg.data) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    // Usage appears in the final chunk (with stream_options.include_usage)
                    if !data["usage"].is_null() {
                        usage = parse_openai_usage(&data["usage"]);
                    }

                    let choice = &data["choices"][0];
                    let delta = &choice["delta"];

                    // Text delta
                    if let Some(text) = delta["content"].as_str() {
                        if !text.is_empty() {
                            text_content.push_str(text);
                            on_event(StreamEvent::TextDelta(text.to_string()));
                        }
                    }

                    // Tool call deltas
                    if let Some(tc_array) = delta["tool_calls"].as_array() {
                        for tc in tc_array {
                            let idx = tc["index"].as_u64().unwrap_or(0) as usize;

                            // Extend vector if needed
                            while tool_calls.len() <= idx {
                                tool_calls.push((String::new(), String::new(), String::new()));
                            }

                            // Initial chunk contains id and function name
                            if let Some(id) = tc["id"].as_str() {
                                tool_calls[idx].0 = id.to_string();
                            }
                            if let Some(name) = tc["function"]["name"].as_str() {
                                tool_calls[idx].1 = name.to_string();
                                on_event(StreamEvent::ToolUseStart {
                                    id: tool_calls[idx].0.clone(),
                                    name: name.to_string(),
                                });
                            }

                            // Argument fragments
                            if let Some(args) = tc["function"]["arguments"].as_str() {
                                tool_calls[idx].2.push_str(args);
                                on_event(StreamEvent::ToolInputDelta {
                                    id: tool_calls[idx].0.clone(),
                                    json_chunk: args.to_string(),
                                });
                            }
                        }
                    }

                    // Finish reason
                    if let Some(fr) = choice["finish_reason"].as_str() {
                        stop_reason = match fr {
                            "stop" => StopReason::EndTurn,
                            "tool_calls" => StopReason::ToolUse,
                            "length" => StopReason::MaxTokens,
                            _ => StopReason::EndTurn,
                        };
                    }
                }
                Err(reqwest_eventsource::Error::StreamEnded) => break,
                Err(e) => {
                    es.close();
                    return Err(CoreError::Provider {
                        reason: format!("SSE stream error: {}", e),
                    });
                }
            }
        }

        // Build final content blocks
        let mut content = Vec::new();
        if !text_content.is_empty() {
            content.push(ContentBlock::text(&text_content));
        }
        for (id, name, args_json) in &tool_calls {
            if !name.is_empty() {
                let arguments: serde_json::Value =
                    serde_json::from_str(args_json).unwrap_or(serde_json::json!({}));
                content.push(ContentBlock::tool_use(id, name, arguments));
            }
        }

        let response = ChatResponse {
            content,
            stop_reason,
            usage,
        };

        on_event(StreamEvent::Done(response.clone()));
        Ok(response)
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
        let body = self.build_body(system, messages, tools, false);
        let response_body = self.send_with_retry(&url, &body).await?;

        let choice = &response_body["choices"][0];
        let message = &choice["message"];

        let mut content = Vec::new();

        if let Some(text) = message["content"].as_str() {
            if !text.is_empty() {
                content.push(ContentBlock::text(text));
            }
        }

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

        let usage = parse_openai_usage(&response_body["usage"]);

        Ok(ChatResponse {
            content,
            stop_reason,
            usage,
        })
    }

    async fn chat_stream(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[LlmTool],
        on_event: &StreamCallback,
    ) -> Result<ChatResponse> {
        let body = self.build_body(system, messages, tools, true);
        self.process_stream(body, on_event).await
    }
}

/// Convert a Message to one or more OpenAI Chat Completions API messages.
///
/// OpenAI format differences from Anthropic:
/// - Assistant tool calls go in `tool_calls` array on the message
/// - Tool results are separate messages with `role: "tool"` and `tool_call_id`
/// Parse OpenAI usage object into canonical TokenUsage.
///
/// OpenAI's `prompt_tokens` INCLUDES cached tokens (unlike Anthropic).
/// Uncached input = prompt_tokens - cached_tokens.
/// `prompt_tokens_details` and `completion_tokens_details` may be absent
/// on OpenAI-compatible backends (Ollama, vLLM, Groq).
fn parse_openai_usage(usage: &serde_json::Value) -> Usage {
    let prompt_tokens = usage["prompt_tokens"].as_u64().unwrap_or(0) as u32;
    let cached_tokens = usage["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or(0) as u32;
    let reasoning_tokens = usage["completion_tokens_details"]["reasoning_tokens"]
        .as_u64()
        .unwrap_or(0) as u32;

    Usage {
        input_tokens: prompt_tokens.saturating_sub(cached_tokens),
        output_tokens: usage["completion_tokens"].as_u64().unwrap_or(0) as u32,
        cache_read_tokens: cached_tokens,
        cache_write_tokens: 0, // OpenAI caching has no write premium
        reasoning_tokens,
    }
}

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
