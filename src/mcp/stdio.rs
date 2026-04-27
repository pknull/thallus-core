//! Stdio MCP client -- subprocess JSON-RPC over stdin/stdout.
//!
//! Spawns an MCP server as a subprocess and communicates via JSON-RPC
//! messages over stdin/stdout.

use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

use crate::config::McpServerConfig;
use crate::error::{CoreError, Result};
use crate::mcp::client::*;

/// MCP client using stdio transport (subprocess).
pub struct StdioMcpClient {
    name: String,
    config: McpServerConfig,
    inner: Arc<Mutex<StdioInner>>,
    request_id: AtomicU64,
}

/// Interior state for the stdio client.
struct StdioInner {
    child: Option<Child>,
    stdin: Option<tokio::process::ChildStdin>,
    stdout: Option<BufReader<tokio::process::ChildStdout>>,
    initialized: bool,
}

impl StdioMcpClient {
    /// Create a new stdio MCP client.
    pub fn new(name: impl Into<String>, config: McpServerConfig) -> Self {
        Self {
            name: name.into(),
            config,
            inner: Arc::new(Mutex::new(StdioInner {
                child: None,
                stdin: None,
                stdout: None,
                initialized: false,
            })),
            request_id: AtomicU64::new(1),
        }
    }

    /// Spawn the subprocess.
    async fn spawn(&self) -> Result<()> {
        let command = self.config.command.as_ref().ok_or_else(|| CoreError::Mcp {
            reason: format!("no command specified for MCP server '{}'", self.name),
        })?;

        let mut cmd = Command::new(command);
        cmd.args(&self.config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);

        // Set environment variables
        for (key, value) in &self.config.env {
            cmd.env(key, value);
        }

        let mut child = cmd.spawn().map_err(|e| CoreError::Mcp {
            reason: format!("failed to spawn MCP server '{}': {}", self.name, e),
        })?;

        let stdin = child.stdin.take().ok_or_else(|| CoreError::Mcp {
            reason: "failed to capture stdin".into(),
        })?;

        let stdout = child.stdout.take().ok_or_else(|| CoreError::Mcp {
            reason: "failed to capture stdout".into(),
        })?;

        let mut inner = self.inner.lock().await;
        inner.child = Some(child);
        inner.stdin = Some(stdin);
        inner.stdout = Some(BufReader::new(stdout));

        tracing::debug!(name = %self.name, "spawned MCP server");
        Ok(())
    }

    /// Send a JSON-RPC request and receive the response.
    async fn rpc<T: serde::de::DeserializeOwned>(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<T> {
        let id = self.request_id.fetch_add(1, Ordering::SeqCst);
        let request = JsonRpcRequest::new(id, method, params);

        // Serialize request
        let mut json = serde_json::to_string(&request)?;
        json.push('\n');

        let mut inner = self.inner.lock().await;

        // Write to stdin
        let stdin = inner.stdin.as_mut().ok_or_else(|| CoreError::Mcp {
            reason: "MCP server not started".into(),
        })?;

        stdin
            .write_all(json.as_bytes())
            .await
            .map_err(|e| CoreError::Mcp {
                reason: format!("failed to write to MCP server: {}", e),
            })?;
        stdin.flush().await.map_err(|e| CoreError::Mcp {
            reason: format!("failed to flush MCP server stdin: {}", e),
        })?;

        // Read response from stdout
        let stdout = inner.stdout.as_mut().ok_or_else(|| CoreError::Mcp {
            reason: "MCP server not started".into(),
        })?;

        let mut line = String::new();
        stdout
            .read_line(&mut line)
            .await
            .map_err(|e| CoreError::Mcp {
                reason: format!("failed to read from MCP server: {}", e),
            })?;

        let response: JsonRpcResponse =
            serde_json::from_str(&line).map_err(|e| CoreError::Mcp {
                reason: format!(
                    "failed to parse MCP response: {} (line: {})",
                    e,
                    line.trim()
                ),
            })?;

        if response.id != id {
            return Err(CoreError::Mcp {
                reason: format!("response ID mismatch: expected {}, got {}", id, response.id),
            });
        }

        if let Some(error) = response.error {
            return Err(CoreError::Mcp {
                reason: format!("MCP error {}: {}", error.code, error.message),
            });
        }

        let result = response.result.ok_or_else(|| CoreError::Mcp {
            reason: "MCP response missing result".into(),
        })?;

        serde_json::from_value(result).map_err(|e| CoreError::Mcp {
            reason: format!("failed to parse MCP result: {}", e),
        })
    }

    /// Send a notification (no response expected).
    async fn notify(&self, method: &str) -> Result<()> {
        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method
        });
        let mut json = serde_json::to_string(&notification)?;
        json.push('\n');

        let mut inner = self.inner.lock().await;
        if let Some(stdin) = inner.stdin.as_mut() {
            let _ = stdin.write_all(json.as_bytes()).await;
            let _ = stdin.flush().await;
        }
        Ok(())
    }
}

#[async_trait]
impl McpClient for StdioMcpClient {
    async fn initialize(&mut self) -> Result<InitializeResult> {
        {
            let inner = self.inner.lock().await;
            if inner.child.is_none() {
                drop(inner);
                self.spawn().await?;
            }
        }

        let params = serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "thallus-core",
                "version": env!("CARGO_PKG_VERSION")
            }
        });

        let result: InitializeResult = self.rpc("initialize", Some(params)).await?;

        // Send initialized notification
        self.notify("notifications/initialized").await?;

        {
            let mut inner = self.inner.lock().await;
            inner.initialized = true;
        }

        tracing::info!(
            name = %self.name,
            server = %result.server_info.name,
            version = %result.server_info.version,
            "initialized MCP server"
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
        let mut inner = self.inner.lock().await;
        if let Some(mut child) = inner.child.take() {
            let _ = child.kill().await;
            let _ = child.wait().await;
        }
        inner.stdin = None;
        inner.stdout = None;
        inner.initialized = false;
        tracing::debug!(name = %self.name, "shutdown MCP server");
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}
