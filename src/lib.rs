//! thallus-core -- shared infrastructure for the Thallus decentralized AI network.
//!
//! Provides identity (Ed25519), MCP client pool, and LLM provider abstractions
//! shared between familiar, servitor, and other Thallus projects.

pub mod config;
pub mod error;
pub mod identity;
pub mod mcp;
pub mod provider;

pub use config::{LlmConfig, McpServerConfig};
pub use error::{CoreError, Result};
pub use identity::{Identity, PublicId};
pub use mcp::{LlmTool, McpClient, McpPool};
pub use provider::{create_provider, Provider};
