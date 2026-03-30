//! Configuration types shared across Thallus projects.
//!
//! Only contains types needed by the shared modules (MCP pool, providers).
//! Project-specific config types live in their respective crates.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// MCP server configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
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

/// LLM provider configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
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
}
