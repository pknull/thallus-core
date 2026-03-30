//! Anthropic Claude provider.

use async_trait::async_trait;

use super::{
    truncate_error_body, ChatResponse, ContentBlock, Message, Provider, ProviderCapabilities,
    StopReason, Usage,
};
use crate::config::LlmConfig;
use crate::error::{CoreError, Result};
use crate::mcp::LlmTool;

/// Anthropic Claude provider.
pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    max_tokens: u32,
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
        })
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
        let url = "https://api.anthropic.com/v1/messages";

        // Build request body
        let mut body = serde_json::json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": messages,
        });

        if !tools.is_empty() {
            // Convert tools to Anthropic format
            let anthropic_tools: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.input_schema,
                    })
                })
                .collect();
            body["tools"] = serde_json::Value::Array(anthropic_tools);
        }

        let response = self
            .client
            .post(url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| CoreError::Provider {
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

        // Parse response
        let content = parse_anthropic_content(&response_body)?;
        let stop_reason = match response_body["stop_reason"].as_str() {
            Some("end_turn") => StopReason::EndTurn,
            Some("tool_use") => StopReason::ToolUse,
            Some("max_tokens") => StopReason::MaxTokens,
            Some("stop_sequence") => StopReason::StopSequence,
            _ => StopReason::EndTurn,
        };

        let usage = Usage {
            input_tokens: response_body["usage"]["input_tokens"].as_u64().unwrap_or(0) as u32,
            output_tokens: response_body["usage"]["output_tokens"]
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
