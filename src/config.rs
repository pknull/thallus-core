//! Configuration types shared across Thallus projects.
//!
//! Only contains types needed by the shared modules (MCP pool, providers).
//! Project-specific config types live in their respective crates.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// MCP server configuration.
#[derive(Clone, Deserialize, Serialize)]
pub struct McpServerConfig {
    pub transport: String,
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default = "default_mcp_timeout")]
    pub timeout_secs: u64,
}

fn default_mcp_timeout() -> u64 {
    60
}

/// Every free-text field can carry credentials, so `Debug` renders recognized
/// selectors from a finite list and reports the rest as `<other>`, `command`
/// and `url` as set or unset, and `args` and `env` as counts.
impl std::fmt::Debug for McpServerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpServerConfig")
            .field("transport", &known(&self.transport, &["stdio", "http"]))
            .field("command", &Redacted(self.command.is_some()))
            .field("args", &format_args!("<{} redacted>", self.args.len()))
            .field("env", &format_args!("<{} redacted>", self.env.len()))
            .field("url", &Redacted(self.url.is_some()))
            .field("timeout_secs", &self.timeout_secs)
            .finish()
    }
}

impl std::fmt::Debug for LlmConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmConfig")
            .field(
                "provider",
                &known(
                    &self.provider,
                    &["anthropic", "openai", "ollama", "openai-compat", "mock"],
                ),
            )
            .field("model", &Redacted(true))
            .field("api_key_env", &Redacted(self.api_key_env.is_some()))
            .field("base_url", &Redacted(self.base_url.is_some()))
            .field("max_tokens", &self.max_tokens)
            .field("temperature", &self.temperature)
            .field("max_retries", &self.max_retries)
            .field("initial_backoff_ms", &self.initial_backoff_ms)
            .field("max_backoff_ms", &self.max_backoff_ms)
            .finish()
    }
}

/// The value if it is one of `allowed`, else `<other>`.
fn known<'a>(value: &str, allowed: &[&'a str]) -> &'a str {
    allowed
        .iter()
        .copied()
        .find(|a| *a == value)
        .unwrap_or("<other>")
}

/// Prints `Some(<redacted>)` or `None`.
struct Redacted(bool);

impl std::fmt::Debug for Redacted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0 { "Some(<redacted>)" } else { "None" })
    }
}

/// LLM provider configuration.
#[derive(Clone, Deserialize, Serialize)]
pub struct LlmConfig {
    pub provider: String,
    pub model: String,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Maximum number of retries on transient failures (default: 2).
    #[serde(default)]
    pub max_retries: Option<u32>,
    /// Initial backoff delay in milliseconds (default: 200).
    #[serde(default)]
    pub initial_backoff_ms: Option<u64>,
    /// Maximum backoff delay in milliseconds (default: 2000).
    #[serde(default)]
    pub max_backoff_ms: Option<u64>,
}
