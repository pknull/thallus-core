//! MCP (Model Context Protocol) client layer.
//!
//! Provides a unified interface for communicating with MCP servers
//! over stdio (subprocess) and HTTP transports.

pub mod circuit_breaker;
pub mod client;
pub mod http;
pub mod pool;
pub mod stdio;

pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitState};
pub use client::{McpClient, ToolCallResult, ToolContent, ToolDefinition};
pub use pool::{sanitize_tool_output, LlmTool, McpPool};
