//! Anthropic Claude provider.

use async_trait::async_trait;
use futures::StreamExt;
use reqwest_eventsource::{Event, RequestBuilderExt};

use super::{
    retry::{is_retryable_error, is_retryable_status, RetryPolicy},
    truncate_error_body, ChatResponse, ContentBlock, Message, Provider, ProviderCapabilities,
    StopReason, StreamCallback, StreamEvent, Usage,
};
use crate::config::LlmConfig;
use crate::error::{CoreError, Result};
use crate::mcp::LlmTool;

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";

/// Anthropic Claude provider.
pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    max_tokens: u32,
    retry_policy: RetryPolicy,
}

impl AnthropicProvider {
    pub fn new(config: &LlmConfig) -> Result<Self> {
        let api_key_env = config
            .api_key_env
            .as_ref()
            .ok_or_else(|| CoreError::Config {
                reason: "anthropic provider requires api_key_env".into(),
            })?;

        let api_key = std::env::var(api_key_env).map_err(|_| CoreError::Config {
            reason: format!("environment variable {} not set", api_key_env),
        })?;

        Ok(Self {
            client: reqwest::Client::new(),
            api_key,
            model: config.model.clone(),
            max_tokens: config.max_tokens.unwrap_or(4096),
            retry_policy: RetryPolicy::from_config(config),
        })
    }
}

impl AnthropicProvider {
    /// Send a request with retry on transient failures.
    async fn send_with_retry(&self, body: &serde_json::Value) -> Result<serde_json::Value> {
        let url = "https://api.anthropic.com/v1/messages";
        let mut last_error: Option<CoreError> = None;

        for attempt in 0..=self.retry_policy.max_retries {
            if attempt > 0 {
                let backoff = self.retry_policy.backoff_for_attempt(attempt);
                tracing::warn!(
                    attempt,
                    backoff_ms = backoff.as_millis() as u64,
                    "retrying anthropic request"
                );
                tokio::time::sleep(backoff).await;
            }

            let result = self
                .client
                .post(url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json")
                .json(body)
                .send()
                .await;

            let response = match result {
                Ok(resp) => resp,
                Err(e) => {
                    if is_retryable_error(&e) && attempt < self.retry_policy.max_retries {
                        tracing::warn!(
                            attempt,
                            error = %e,
                            "transient request error, will retry"
                        );
                        last_error = Some(CoreError::Provider {
                            reason: format!("request failed: {}", e),
                        });
                        continue;
                    }
                    return Err(CoreError::Provider {
                        reason: format!("request failed: {}", e),
                    });
                }
            };

            let status = response.status();

            if status.is_success() {
                return response.json().await.map_err(|e| CoreError::Provider {
                    reason: format!("failed to parse response: {}", e),
                });
            }

            let status_code = status.as_u16();
            let body_text = response.text().await.unwrap_or_default();

            if is_retryable_status(status_code) && attempt < self.retry_policy.max_retries {
                tracing::warn!(attempt, status = status_code, "retryable API error");
                last_error = Some(CoreError::Provider {
                    reason: format!("API error {}: {}", status, truncate_error_body(&body_text)),
                });
                continue;
            }

            return Err(CoreError::Provider {
                reason: format!("API error {}: {}", status, truncate_error_body(&body_text)),
            });
        }

        Err(last_error.unwrap_or_else(|| CoreError::Provider {
            reason: "retries exhausted".into(),
        }))
    }

    /// Build the Anthropic request body from messages and tools.
    fn build_body(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[LlmTool],
        stream: bool,
    ) -> serde_json::Value {
        // System prompt as structured content block with cache_control.
        // Anthropic caches the prefix up to the last cache_control breakpoint.
        let system_value = if system.is_empty() {
            serde_json::json!([])
        } else {
            serde_json::json!([{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"}
            }])
        };

        let mut body = serde_json::json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_value,
            "messages": messages,
        });

        if stream {
            body["stream"] = serde_json::Value::Bool(true);
        }

        if !tools.is_empty() {
            let mut anthropic_tools: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.input_schema,
                    })
                })
                .collect();

            // Mark the last tool with cache_control so tools are also cached.
            if let Some(last) = anthropic_tools.last_mut() {
                last["cache_control"] = serde_json::json!({"type": "ephemeral"});
            }

            body["tools"] = serde_json::Value::Array(anthropic_tools);
        }

        body
    }

    /// Process an SSE stream from the Anthropic API.
    async fn process_stream(
        &self,
        body: serde_json::Value,
        on_event: &StreamCallback,
    ) -> Result<ChatResponse> {
        let request = self
            .client
            .post(ANTHROPIC_API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body);

        let mut es = request.eventsource().map_err(|e| CoreError::Provider {
            reason: format!("failed to create event source: {}", e),
        })?;

        // Accumulator state
        let mut content_blocks: Vec<ContentBlock> = Vec::new();
        let mut current_text = String::new();
        let mut current_tool_id = String::new();
        let mut current_tool_name = String::new();
        let mut current_tool_json = String::new();
        let mut in_text_block = false;
        let mut in_tool_block = false;
        let mut stop_reason = StopReason::EndTurn;
        let mut usage = Usage::default();

        while let Some(event) = es.next().await {
            match event {
                Ok(Event::Open) => {}
                Ok(Event::Message(msg)) => {
                    let data: serde_json::Value = match serde_json::from_str(&msg.data) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    let event_type = data["type"].as_str().unwrap_or("");

                    match event_type {
                        "message_start" => {
                            // Extract all input-side token counts from initial message.
                            // Cache fields are authoritative here — do NOT re-read from message_delta.
                            if let Some(u) = data["message"]["usage"].as_object() {
                                usage.input_tokens =
                                    u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0)
                                        as u32;
                                usage.cache_write_tokens =
                                    u.get("cache_creation_input_tokens")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0) as u32;
                                usage.cache_read_tokens =
                                    u.get("cache_read_input_tokens")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0) as u32;
                            }
                        }

                        "content_block_start" => {
                            let block = &data["content_block"];
                            match block["type"].as_str() {
                                Some("text") => {
                                    in_text_block = true;
                                    current_text.clear();
                                }
                                Some("tool_use") => {
                                    in_tool_block = true;
                                    current_tool_id =
                                        block["id"].as_str().unwrap_or("").to_string();
                                    current_tool_name =
                                        block["name"].as_str().unwrap_or("").to_string();
                                    current_tool_json.clear();

                                    on_event(StreamEvent::ToolUseStart {
                                        id: current_tool_id.clone(),
                                        name: current_tool_name.clone(),
                                    });
                                }
                                _ => {}
                            }
                        }

                        "content_block_delta" => {
                            let delta = &data["delta"];
                            match delta["type"].as_str() {
                                Some("text_delta") => {
                                    if let Some(text) = delta["text"].as_str() {
                                        current_text.push_str(text);
                                        on_event(StreamEvent::TextDelta(text.to_string()));
                                    }
                                }
                                Some("input_json_delta") => {
                                    if let Some(json) = delta["partial_json"].as_str() {
                                        current_tool_json.push_str(json);
                                        on_event(StreamEvent::ToolInputDelta {
                                            id: current_tool_id.clone(),
                                            json_chunk: json.to_string(),
                                        });
                                    }
                                }
                                _ => {}
                            }
                        }

                        "content_block_stop" => {
                            if in_text_block {
                                if !current_text.is_empty() {
                                    content_blocks.push(ContentBlock::text(&current_text));
                                }
                                in_text_block = false;
                            }
                            if in_tool_block {
                                let input: serde_json::Value =
                                    serde_json::from_str(&current_tool_json)
                                        .unwrap_or(serde_json::json!({}));
                                content_blocks.push(ContentBlock::tool_use(
                                    &current_tool_id,
                                    &current_tool_name,
                                    input,
                                ));
                                in_tool_block = false;
                            }
                        }

                        "message_delta" => {
                            if let Some(sr) = data["delta"]["stop_reason"].as_str() {
                                stop_reason = match sr {
                                    "end_turn" => StopReason::EndTurn,
                                    "tool_use" => StopReason::ToolUse,
                                    "max_tokens" => StopReason::MaxTokens,
                                    "stop_sequence" => StopReason::StopSequence,
                                    _ => StopReason::EndTurn,
                                };
                            }
                            if let Some(u) = data["usage"].as_object() {
                                usage.output_tokens =
                                    u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0)
                                        as u32;
                            }
                        }

                        "message_stop" => {
                            es.close();
                            break;
                        }

                        _ => {}
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

        let response = ChatResponse {
            content: content_blocks,
            stop_reason,
            usage,
        };

        on_event(StreamEvent::Done(response.clone()));
        Ok(response)
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supports_tools: true,
            supports_vision: true,
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
        let body = self.build_body(system, messages, tools, false);
        let response_body = self.send_with_retry(&body).await?;

        let content = parse_anthropic_content(&response_body)?;
        let stop_reason = match response_body["stop_reason"].as_str() {
            Some("end_turn") => StopReason::EndTurn,
            Some("tool_use") => StopReason::ToolUse,
            Some("max_tokens") => StopReason::MaxTokens,
            Some("stop_sequence") => StopReason::StopSequence,
            _ => StopReason::EndTurn,
        };

        let usage = parse_anthropic_usage(&response_body["usage"]);

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

/// Parse Anthropic usage object into canonical TokenUsage.
///
/// Anthropic's `input_tokens` is the uncached portion only.
/// Total input = input_tokens + cache_creation_input_tokens + cache_read_input_tokens.
fn parse_anthropic_usage(usage: &serde_json::Value) -> Usage {
    Usage {
        input_tokens: usage["input_tokens"].as_u64().unwrap_or(0) as u32,
        output_tokens: usage["output_tokens"].as_u64().unwrap_or(0) as u32,
        cache_write_tokens: usage["cache_creation_input_tokens"].as_u64().unwrap_or(0) as u32,
        cache_read_tokens: usage["cache_read_input_tokens"].as_u64().unwrap_or(0) as u32,
        reasoning_tokens: 0,
    }
}

fn parse_anthropic_content(response: &serde_json::Value) -> Result<Vec<ContentBlock>> {
    let content_array = response["content"]
        .as_array()
        .ok_or_else(|| CoreError::Provider {
            reason: "response missing content array".into(),
        })?;

    let mut blocks = Vec::new();
    for item in content_array {
        match item["type"].as_str() {
            Some("text") => {
                if let Some(text) = item["text"].as_str() {
                    blocks.push(ContentBlock::text(text));
                }
            }
            Some("tool_use") => {
                let id = item["id"].as_str().unwrap_or("").to_string();
                let name = item["name"].as_str().unwrap_or("").to_string();
                let input = item["input"].clone();
                blocks.push(ContentBlock::tool_use(id, name, input));
            }
            _ => {}
        }
    }

    Ok(blocks)
}
