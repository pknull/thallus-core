//! Error types for thallus-core.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CoreError {
    #[error("Identity not found at {path}")]
    IdentityNotFound { path: String },

    #[error("Invalid keypair: {reason}")]
    InvalidKeypair { reason: String },

    #[error("MCP error: {reason}")]
    Mcp { reason: String },

    #[error("MCP server '{name}' not found")]
    McpServerNotFound { name: String },

    #[error("Invalid arguments for MCP tool '{tool}': {reason}")]
    McpValidation { tool: String, reason: String },

    #[error("LLM provider error: {reason}")]
    Provider { reason: String },

    #[error("Config error: {reason}")]
    Config { reason: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
}

pub type Result<T> = std::result::Result<T, CoreError>;
