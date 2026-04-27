//! HTTP MCP client -- JSON-RPC over HTTP.
//!
//! Connects to an MCP server via HTTP transport.

use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use reqwest::Client;

use crate::config::McpServerConfig;
use crate::error::{CoreError, Result};
use crate::mcp::client::*;

/// MCP client using HTTP transport.
pub struct HttpMcpClient {
    name: String,
    url: String,
    http: Client,
    request_id: AtomicU64,
    initialized: bool,
}

impl HttpMcpClient {
    /// Create a new HTTP MCP client.
    pub fn new(name: impl Into<String>, config: &McpServerConfig) -> Result<Self> {
        let url = config.url.as_ref().ok_or_else(|| CoreError::Mcp {
            reason: "HTTP transport requires url".into(),
        })?;

        Ok(Self {
            name: name.into(),
            url: url.clone(),
            http: Client::new(),
            request_id: AtomicU64::new(1),
            initialized: false,
        })
    }

    /// Send a JSON-RPC request.
    async fn rpc<T: serde::de::DeserializeOwned>(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<T> {
        let id = self.request_id.fetch_add(1, Ordering::SeqCst);
        let request = JsonRpcRequest::new(id, method, params);

        let response = self
            .http
            .post(&self.url)
            .json(&request)
            .send()
            .await
            .map_err(|e| CoreError::Mcp {
                reason: format!("HTTP request failed: {}", e),
            })?;

        if !response.status().is_success() {
            return Err(CoreError::Mcp {
                reason: format!("HTTP error: {}", response.status()),
            });
        }

        let rpc_response: JsonRpcResponse = response.json().await.map_err(|e| CoreError::Mcp {
            reason: format!("failed to parse response: {}", e),
        })?;

        if rpc_response.id != id {
            return Err(CoreError::Mcp {
                reason: format!(
                    "response ID mismatch: expected {}, got {}",
                    id, rpc_response.id
                ),
            });
        }

        if let Some(error) = rpc_response.error {
            return Err(CoreError::Mcp {
                reason: format!("MCP error {}: {}", error.code, error.message),
            });
        }

        let result = rpc_response.result.ok_or_else(|| CoreError::Mcp {
            reason: "response missing result".into(),
        })?;

        serde_json::from_value(result).map_err(|e| CoreError::Mcp {
            reason: format!("failed to parse result: {}", e),
        })
    }
}

#[async_trait]
impl McpClient for HttpMcpClient {
    async fn initialize(&mut self) -> Result<InitializeResult> {
        let params = serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "thallus-core",
                "version": env!("CARGO_PKG_VERSION")
            }
        });

        let result: InitializeResult = self.rpc("initialize", Some(params)).await?;
        self.initialized = true;

        tracing::info!(
            name = %self.name,
            server = %result.server_info.name,
            version = %result.server_info.version,
            "initialized HTTP MCP server"
        );

        Ok(result)
    }

    async fn list_tools(&self) -> Result<Vec<ToolDefinition>> {
        #[derive(serde::Deserialize)]
        struct ListToolsResult {
            tools: Vec<ToolDefinition>,
        }

        let result: ListToolsResult = self.rpc("tools/list", None).await?;
        Ok(result.tools)
    }

    async fn call_tool(&self, name: &str, arguments: serde_json::Value) -> Result<ToolCallResult> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments
        });

        self.rpc("tools/call", Some(params)).await
    }

    async fn ping(&self) -> Result<()> {
        let _: serde_json::Value = self.rpc("ping", None).await?;
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<()> {
        self.initialized = false;
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}
