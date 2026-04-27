//! Claude Code provider -- uses Claude Code CLI via subprocess.
//! No API key needed; authenticates through the installed Claude Code CLI.

use async_trait::async_trait;

use super::{
    ChatResponse, ContentBlock, Message, Provider, ProviderCapabilities, Role, StopReason, Usage,
};
use crate::config::LlmConfig;
use crate::error::{CoreError, Result};
use crate::mcp::LlmTool;

/// Claude Code provider -- uses Claude Code CLI via subprocess.
/// No API key needed; authenticates through the installed Claude Code CLI.
pub struct ClaudeCodeProvider {
    #[allow(dead_code)]
    model: String,
    max_tokens: u32,
}

impl ClaudeCodeProvider {
    pub fn new(config: &LlmConfig) -> Result<Self> {
        Ok(Self {
            model: config.model.clone(),
            max_tokens: config.max_tokens.unwrap_or(4096),
        })
    }
}

#[async_trait]
impl Provider for ClaudeCodeProvider {
    fn name(&self) -> &str {
        "claude-code"
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
        use tokio::io::{AsyncBufReadExt, BufReader};
        use tokio::process::Command;

        // Build prompt from system message and conversation history
        let mut prompt = String::new();

        // Include system context
        if !system.is_empty() {
            prompt.push_str("<system>\n");
            prompt.push_str(system);
            prompt.push_str("\n</system>\n\n");
        }

        // Include available tools in the prompt
        if !tools.is_empty() {
            prompt.push_str("<available_tools>\n");
            for tool in tools {
                let desc = tool.description.as_deref().unwrap_or("No description");
                prompt.push_str(&format!(
                    "- {}: {}\n  Parameters: {}\n",
                    tool.name,
                    desc,
                    serde_json::to_string(&tool.input_schema).unwrap_or_default()
                ));
            }
            prompt.push_str("</available_tools>\n\n");
            prompt.push_str("To use a tool, respond with a JSON block:\n");
            prompt
                .push_str("```tool_use\n{\"name\": \"tool_name\", \"arguments\": {...}}\n```\n\n");
        }

        // Convert message history to text
        for msg in messages {
            let role_label = match msg.role {
                Role::User => "User",
                Role::Assistant => "Assistant",
                Role::System => "System",
            };

            for block in &msg.content {
                match block {
                    ContentBlock::Text { text } => {
                        prompt.push_str(&format!("{}: {}\n", role_label, text));
                    }
                    ContentBlock::ToolUse { name, input, .. } => {
                        prompt.push_str(&format!(
                            "Assistant used tool {}: {}\n",
                            name,
                            serde_json::to_string(input).unwrap_or_default()
                        ));
                    }
                    ContentBlock::ToolResult {
                        content, is_error, ..
                    } => {
                        if *is_error {
                            prompt.push_str(&format!("Tool error: {}\n", content));
                        } else {
                            prompt.push_str(&format!("Tool result: {}\n", content));
                        }
                    }
                }
            }
        }

        tracing::debug!(prompt_len = prompt.len(), "sending to Claude Code CLI");

        // Build command arguments
        let mut cmd = Command::new("claude");
        cmd.arg("--print")
            .arg("--verbose")
            .arg("--output-format")
            .arg("stream-json")
            .arg("--max-turns")
            .arg("1")
            .arg(&prompt)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        // Spawn the process
        let mut child = cmd.spawn().map_err(|e| CoreError::Provider {
            reason: format!("Failed to spawn Claude CLI: {}", e),
        })?;

        let stdout = child.stdout.take().ok_or_else(|| CoreError::Provider {
            reason: "Failed to capture stdout".into(),
        })?;

        let reader = BufReader::new(stdout);
        let mut lines = reader.lines();

        // Collect response
        let mut content = Vec::new();
        let mut accumulated_text = String::new();

        while let Some(line) = lines.next_line().await.map_err(|e| CoreError::Provider {
            reason: format!("Failed to read line: {}", e),
        })? {
            if line.trim().is_empty() {
                continue;
            }

            let json: serde_json::Value = match serde_json::from_str(&line) {
                Ok(j) => j,
                Err(e) => {
                    tracing::trace!(error = %e, line = %line, "Failed to parse JSON line");
                    continue;
                }
            };

            let msg_type = json.get("type").and_then(|v| v.as_str()).unwrap_or("");

            match msg_type {
                "assistant" => {
                    if let Some(message) = json.get("message") {
                        if let Some(contents) = message.get("content").and_then(|v| v.as_array()) {
                            for block in contents {
                                let block_type =
                                    block.get("type").and_then(|v| v.as_str()).unwrap_or("");
                                match block_type {
                                    "text" => {
                                        if let Some(text) =
                                            block.get("text").and_then(|v| v.as_str())
                                        {
                                            accumulated_text.push_str(text);
                                        }
                                    }
                                    "tool_use" => {
                                        let id = block
                                            .get("id")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("")
                                            .to_string();
                                        let name = block
                                            .get("name")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("")
                                            .to_string();
                                        let input = block.get("input").cloned().unwrap_or_default();
                                        content.push(ContentBlock::tool_use(id, name, input));
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                "result" => {
                    break;
                }
                "rate_limit_event" => {
                    tracing::warn!("Rate limit event received, waiting...");
                }
                _ => {
                    tracing::trace!(msg_type = %msg_type, "Skipping message type");
                }
            }
        }

        // Wait for process to complete
        let status = child.wait().await.map_err(|e| CoreError::Provider {
            reason: format!("Failed to wait for Claude CLI: {}", e),
        })?;

        if !status.success() {
            return Err(CoreError::Provider {
                reason: format!("Claude CLI exited with status: {}", status),
            });
        }

        // Parse any tool_use blocks from the accumulated text
        if !accumulated_text.is_empty() {
            let tool_uses = parse_tool_use_blocks(&accumulated_text);
            if tool_uses.is_empty() {
                content.push(ContentBlock::text(accumulated_text));
            } else {
                if let Some(first_idx) = accumulated_text.find("```tool_use") {
                    let before = accumulated_text[..first_idx].trim();
                    if !before.is_empty() {
                        content.push(ContentBlock::text(before));
                    }
                }
                for (name, args) in tool_uses {
                    let id = format!("tool_{}", uuid::Uuid::new_v4());
                    content.push(ContentBlock::tool_use(id, name, args));
                }
            }
        }

        let stop_reason = if content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolUse { .. }))
        {
            StopReason::ToolUse
        } else {
            StopReason::EndTurn
        };

        Ok(ChatResponse {
            content,
            stop_reason,
            usage: Usage::default(),
        })
    }
}

/// Parse tool_use code blocks from markdown response.
fn parse_tool_use_blocks(text: &str) -> Vec<(String, serde_json::Value)> {
    let mut results = Vec::new();
    let mut remaining = text;

    while let Some(start) = remaining.find("```tool_use") {
        let after_marker = &remaining[start + 11..];
        if let Some(end) = after_marker.find("```") {
            let json_str = after_marker[..end].trim();
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(json_str) {
                if let (Some(name), Some(args)) = (
                    json.get("name").and_then(|v| v.as_str()),
                    json.get("arguments"),
                ) {
                    results.push((name.to_string(), args.clone()));
                }
            }
            remaining = &after_marker[end + 3..];
        } else {
            break;
        }
    }

    results
}
