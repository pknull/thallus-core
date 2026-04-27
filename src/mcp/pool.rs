//! MCP client pool -- manages multiple MCP server connections.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use jsonschema::JSONSchema;
use tokio::sync::RwLock;

use crate::config::McpServerConfig;
use crate::error::{CoreError, Result};
use crate::mcp::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitState};
use crate::mcp::client::{McpClient, ToolDefinition};
use crate::mcp::http::HttpMcpClient;
use crate::mcp::stdio::StdioMcpClient;

/// Pool of MCP clients with tool introspection.
pub struct McpPool {
    clients: HashMap<String, Arc<RwLock<Box<dyn McpClient>>>>,
    server_runtime: HashMap<String, McpServerRuntime>,
    /// Circuit breakers per server.
    circuit_breakers: HashMap<String, RwLock<CircuitBreaker>>,
    /// All tools with prefixed names, mapped to their server.
    tools: HashMap<String, RegisteredTool>,
}

struct RegisteredTool {
    server_name: String,
    definition: ToolDefinition,
    validator: Option<JSONSchema>,
}

#[derive(Debug, Clone)]
struct McpServerRuntime {
    #[allow(dead_code)]
    transport: String,
    initialized: bool,
}

impl McpPool {
    /// Create a new empty pool.
    pub fn new() -> Self {
        Self {
            clients: HashMap::new(),
            server_runtime: HashMap::new(),
            circuit_breakers: HashMap::new(),
            tools: HashMap::new(),
        }
    }

    /// Add a client to the pool.
    pub fn add_client(&mut self, name: &str, config: &McpServerConfig) -> Result<()> {
        let client: Box<dyn McpClient> = match config.transport.as_str() {
            "stdio" => Box::new(StdioMcpClient::new(name, config.clone())),
            "http" => Box::new(HttpMcpClient::new(name, config)?),
            other => {
                return Err(CoreError::Mcp {
                    reason: format!("unknown transport: {}", other),
                })
            }
        };

        self.clients
            .insert(name.to_string(), Arc::new(RwLock::new(client)));
        self.server_runtime.insert(
            name.to_string(),
            McpServerRuntime {
                transport: config.transport.clone(),
                initialized: false,
            },
        );

        // Add circuit breaker for this server
        let cb_config = CircuitBreakerConfig {
            failure_threshold: 3,
            recovery_timeout: Duration::from_secs(30),
            success_threshold: 1,
        };
        self.circuit_breakers.insert(
            name.to_string(),
            RwLock::new(CircuitBreaker::new(cb_config)),
        );

        Ok(())
    }

    /// Initialize all clients and introspect tools.
    pub async fn initialize_all(&mut self) -> Result<()> {
        let clients: Vec<_> = self
            .clients
            .iter()
            .map(|(name, client)| (name.clone(), Arc::clone(client)))
            .collect();

        for (name, client) in clients {
            let mut client = client.write().await;

            // Initialize the server
            client.initialize().await?;

            // List and register tools
            let tools = client.list_tools().await?;
            for tool in tools {
                let prefixed_name = tool.prefixed_name(&name);
                let validator = compile_validator(&tool)?;
                self.tools.insert(
                    prefixed_name,
                    RegisteredTool {
                        server_name: name.clone(),
                        definition: tool,
                        validator,
                    },
                );
            }

            if let Some(runtime) = self.server_runtime.get_mut(&name) {
                runtime.initialized = true;
            }
        }

        tracing::info!(
            servers = self.clients.len(),
            tools = self.tools.len(),
            "initialized MCP pool"
        );

        Ok(())
    }

    /// Get all available tools with prefixed names.
    pub fn all_tools(&self) -> Vec<(&str, &ToolDefinition)> {
        self.tools
            .iter()
            .map(|(prefixed, tool)| (prefixed.as_str(), &tool.definition))
            .collect()
    }

    /// Get tools formatted for LLM consumption.
    pub fn tools_for_llm(&self) -> Vec<LlmTool> {
        self.tools
            .iter()
            .map(|(prefixed_name, tool)| LlmTool {
                name: prefixed_name.clone(),
                description: tool.definition.description.clone(),
                input_schema: tool.definition.input_schema.clone().unwrap_or_else(|| {
                    serde_json::json!({
                        "type": "object",
                        "properties": {}
                    })
                }),
            })
            .collect()
    }

    /// Parse a prefixed tool name into (server_name, tool_name).
    pub fn parse_tool_name<'a>(&'a self, prefixed: &'a str) -> Option<(&'a str, &'a str)> {
        if let Some(tool) = self.tools.get(prefixed) {
            // Extract the original tool name by removing the prefix
            let prefix_len = tool.server_name.len() + 1; // server_name + underscore
            if prefixed.len() > prefix_len {
                let tool_name = &prefixed[prefix_len..];
                return Some((tool.server_name.as_str(), tool_name));
            }
        }
        None
    }

    /// Call a tool by its prefixed name.
    ///
    /// Respects circuit breaker state -- rejects calls if the server's
    /// circuit is open (too many recent failures).
    pub async fn call_tool(
        &self,
        prefixed_name: &str,
        arguments: serde_json::Value,
    ) -> Result<crate::mcp::client::ToolCallResult> {
        let tool = self
            .tools
            .get(prefixed_name)
            .ok_or_else(|| CoreError::Mcp {
                reason: format!("unknown tool: {}", prefixed_name),
            })?;
        let tool_name = tool.definition.name.as_str();
        let server_name = &tool.server_name;

        // Check circuit breaker
        if let Some(cb) = self.circuit_breakers.get(server_name) {
            let mut cb = cb.write().await;
            if !cb.should_allow() {
                tracing::warn!(
                    server = %server_name,
                    tool = %prefixed_name,
                    "circuit breaker open, rejecting call"
                );
                return Err(CoreError::Mcp {
                    reason: format!(
                        "circuit breaker open for server '{}' -- too many failures",
                        server_name
                    ),
                });
            }
        }

        validate_arguments(prefixed_name, tool, &arguments)?;

        let client = self
            .clients
            .get(server_name)
            .ok_or_else(|| CoreError::McpServerNotFound {
                name: server_name.to_string(),
            })?;

        let client = client.read().await;
        let result = client.call_tool(tool_name, arguments).await;

        // Record success/failure in circuit breaker
        if let Some(cb) = self.circuit_breakers.get(server_name) {
            let mut cb = cb.write().await;
            match &result {
                Ok(_) => cb.record_success(),
                Err(_) => cb.record_failure(),
            }
        }

        // Sanitize tool output to prevent secret leakage into LLM context
        result.map(|mut r| {
            for content in &mut r.content {
                if let crate::mcp::client::ToolContent::Text { text } = content {
                    *text = sanitize_tool_output(text);
                }
            }
            r
        })
    }

    /// Get capability classes (server names).
    pub fn capabilities(&self) -> Vec<String> {
        self.clients.keys().cloned().collect()
    }

    /// Run health checks on all servers and update circuit breakers.
    ///
    /// Call this periodically to proactively detect server failures.
    pub async fn health_check(&self) {
        for (name, client) in &self.clients {
            let Some(runtime) = self.server_runtime.get(name) else {
                continue;
            };

            if !runtime.initialized {
                continue;
            }

            let client = client.read().await;
            let result = client.ping().await;

            if let Some(cb) = self.circuit_breakers.get(name) {
                let mut cb = cb.write().await;
                match result {
                    Ok(()) => {
                        // Only record success if circuit was half-open (testing recovery)
                        if cb.state() == CircuitState::HalfOpen {
                            cb.record_success();
                            tracing::info!(server = %name, "MCP server recovered");
                        }
                    }
                    Err(error) => {
                        cb.record_failure();
                        tracing::warn!(server = %name, error = %error, "MCP server health check failed");
                    }
                }
            }
        }
    }

    /// Get circuit breaker state for a server.
    pub async fn circuit_state(&self, server_name: &str) -> Option<CircuitState> {
        if let Some(cb) = self.circuit_breakers.get(server_name) {
            let cb = cb.read().await;
            Some(cb.state())
        } else {
            None
        }
    }

    /// Manually reset a server's circuit breaker.
    pub async fn reset_circuit(&self, server_name: &str) {
        if let Some(cb) = self.circuit_breakers.get(server_name) {
            let mut cb = cb.write().await;
            cb.reset();
            tracing::info!(server = %server_name, "circuit breaker manually reset");
        }
    }

    /// Shutdown all clients.
    pub async fn shutdown_all(&self) -> Result<()> {
        for client in self.clients.values() {
            let mut client = client.write().await;
            let _ = client.shutdown().await;
        }
        Ok(())
    }
}

impl Default for McpPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Tool definition formatted for LLM consumption.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LlmTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

fn compile_validator(tool: &ToolDefinition) -> Result<Option<JSONSchema>> {
    let Some(schema) = tool.input_schema.as_ref() else {
        return Ok(None);
    };

    JSONSchema::options()
        .compile(schema)
        .map(Some)
        .map_err(|error| CoreError::Mcp {
            reason: format!("invalid input schema for tool '{}': {}", tool.name, error),
        })
}

/// Sanitize tool output to prevent secrets from leaking into LLM context.
///
/// Redacts strings matching common secret patterns:
/// - Prefixed tokens: `sk-*`, `key-*`, `token-*`, `xoxb-*`, `xoxp-*`, `ghp_*`, `gho_*`
/// - Base64-encoded strings >40 chars that look like keys (high entropy, no spaces)
/// - Values that look like known sensitive environment variable values
pub fn sanitize_tool_output(content: &str) -> String {
    use regex::Regex;
    use std::sync::OnceLock;

    static PREFIX_RE: OnceLock<Regex> = OnceLock::new();
    static B64_KEY_RE: OnceLock<Regex> = OnceLock::new();
    static FIELD_RE: OnceLock<Regex> = OnceLock::new();

    // Pattern 1: Common API key prefixes followed by alphanumeric/dash/underscore content
    let prefix_re = PREFIX_RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(sk-[a-zA-Z0-9_-]{20,}|key-[a-zA-Z0-9_-]{20,}|token-[a-zA-Z0-9_-]{20,}|xox[bp]-[a-zA-Z0-9_-]{20,}|ghp_[a-zA-Z0-9]{36,}|gho_[a-zA-Z0-9]{36,}|sk-ant-[a-zA-Z0-9_-]{20,}|sk-proj-[a-zA-Z0-9_-]{20,})"
        ).unwrap()
    });

    // Pattern 3: Field-name patterns (key=value, secret=value, password=value, etc.)
    let field_re = FIELD_RE.get_or_init(|| {
        Regex::new(
            r#"(?i)["']?(?:secret|password|credential|passwd|api_?key|auth_?token|access_?token|private_?key)[_\s]*["']?\s*[:=]\s*["']?([^\s"',}{]+)"#
        ).unwrap()
    });

    // Pattern 2: Long base64-like strings (potential keys/tokens) -- 40+ chars of
    // base64 alphabet without spaces
    let b64_key_re = B64_KEY_RE.get_or_init(|| Regex::new(r"\b[A-Za-z0-9+/=_-]{40,}\b").unwrap());

    let mut result = prefix_re.replace_all(content, "[REDACTED]").to_string();

    // Redact field-name-pattern matches (secret=value, password=value, etc.)
    result = field_re
        .replace_all(&result, |caps: &regex::Captures| {
            let full = &caps[0];
            let value = &caps[1];
            full.replace(value, "[REDACTED]")
        })
        .to_string();

    // For base64-like strings, only redact if they look like keys (high ratio of
    // mixed case and digits, not a normal word or path)
    result = b64_key_re
        .replace_all(&result, |caps: &regex::Captures| {
            let matched = &caps[0];
            // Heuristic: real base64 keys have mixed case and digits
            let has_upper = matched.chars().any(|c| c.is_ascii_uppercase());
            let has_lower = matched.chars().any(|c| c.is_ascii_lowercase());
            let has_digit = matched.chars().any(|c| c.is_ascii_digit());
            if has_upper && has_lower && has_digit && matched.len() >= 40 {
                "[REDACTED]".to_string()
            } else {
                matched.to_string()
            }
        })
        .to_string();

    result
}

fn validate_arguments(
    prefixed_name: &str,
    tool: &RegisteredTool,
    arguments: &serde_json::Value,
) -> Result<()> {
    let Some(validator) = tool.validator.as_ref() else {
        return Ok(());
    };

    let details = match validator.validate(arguments) {
        Ok(()) => return Ok(()),
        Err(errors) => errors
            .take(5)
            .map(|error| {
                let path = error.instance_path.to_string();
                if path.is_empty() {
                    error.to_string()
                } else {
                    format!("{}: {}", path, error)
                }
            })
            .collect::<Vec<_>>()
            .join("; "),
    };

    tracing::warn!(tool = prefixed_name, reason = %details, "rejected MCP tool call");
    Err(CoreError::McpValidation {
        tool: prefixed_name.to_string(),
        reason: details,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;

    use crate::mcp::client::{InitializeResult, ServerCapabilities, ServerInfo, ToolCallResult};

    struct FakeClient {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl McpClient for FakeClient {
        async fn initialize(&mut self) -> Result<InitializeResult> {
            Ok(InitializeResult {
                protocol_version: "2024-11-05".to_string(),
                server_info: ServerInfo {
                    name: "fake".to_string(),
                    version: "1.0.0".to_string(),
                },
                capabilities: ServerCapabilities::default(),
            })
        }

        async fn list_tools(&self) -> Result<Vec<ToolDefinition>> {
            Ok(vec![])
        }

        async fn call_tool(
            &self,
            _name: &str,
            _arguments: serde_json::Value,
        ) -> Result<ToolCallResult> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(ToolCallResult::text("ok"))
        }

        async fn ping(&self) -> Result<()> {
            Ok(())
        }

        async fn shutdown(&mut self) -> Result<()> {
            Ok(())
        }

        fn name(&self) -> &str {
            "fake"
        }
    }

    fn pool_with_tool(schema: serde_json::Value) -> (McpPool, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut pool = McpPool::new();
        pool.clients.insert(
            "shell".to_string(),
            Arc::new(RwLock::new(Box::new(FakeClient {
                calls: calls.clone(),
            }))),
        );

        // Add circuit breaker for the server
        pool.circuit_breakers
            .insert("shell".to_string(), RwLock::new(CircuitBreaker::default()));

        let definition = ToolDefinition {
            name: "execute".to_string(),
            description: Some("Execute a shell command".to_string()),
            input_schema: Some(schema),
        };
        let validator = compile_validator(&definition).unwrap();
        pool.tools.insert(
            "shell_execute".to_string(),
            RegisteredTool {
                server_name: "shell".to_string(),
                definition,
                validator,
            },
        );

        (pool, calls)
    }

    #[tokio::test]
    async fn rejects_invalid_arguments_before_transport_call() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "command": { "type": "string" }
            },
            "required": ["command"],
            "additionalProperties": false
        });
        let (pool, calls) = pool_with_tool(schema);

        let error = pool
            .call_tool("shell_execute", serde_json::json!({ "command": 42 }))
            .await
            .unwrap_err();

        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(matches!(error, CoreError::McpValidation { .. }));
        assert!(error.to_string().contains("command"));
    }

    #[tokio::test]
    async fn allows_valid_arguments_through_to_transport() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "command": { "type": "string" }
            },
            "required": ["command"],
            "additionalProperties": false
        });
        let (pool, calls) = pool_with_tool(schema);

        let result = pool
            .call_tool("shell_execute", serde_json::json!({ "command": "ls /tmp" }))
            .await
            .unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(result.text_content(), "ok");
    }

    #[tokio::test]
    async fn circuit_breaker_rejects_after_failures() {
        use crate::mcp::circuit_breaker::CircuitBreakerConfig;

        let calls = Arc::new(AtomicUsize::new(0));
        let mut pool = McpPool::new();
        pool.clients.insert(
            "failing".to_string(),
            Arc::new(RwLock::new(Box::new(FakeClient {
                calls: calls.clone(),
            }))),
        );

        // Circuit breaker that opens after 2 failures
        let cb_config = CircuitBreakerConfig {
            failure_threshold: 2,
            recovery_timeout: std::time::Duration::from_secs(60),
            success_threshold: 1,
        };
        pool.circuit_breakers.insert(
            "failing".to_string(),
            RwLock::new(CircuitBreaker::new(cb_config)),
        );

        let definition = ToolDefinition {
            name: "test".to_string(),
            description: None,
            input_schema: None,
        };
        pool.tools.insert(
            "failing_test".to_string(),
            RegisteredTool {
                server_name: "failing".to_string(),
                definition,
                validator: None,
            },
        );

        // Manually trip the circuit breaker
        {
            let mut cb = pool.circuit_breakers.get("failing").unwrap().write().await;
            cb.record_failure();
            cb.record_failure();
            assert_eq!(cb.state(), CircuitState::Open);
        }

        // Call should be rejected
        let result = pool.call_tool("failing_test", serde_json::json!({})).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("circuit breaker open"));

        // The underlying client should not have been called
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn circuit_state_query() {
        let pool = McpPool::new();

        // Non-existent server returns None
        assert!(pool.circuit_state("nonexistent").await.is_none());
    }

    #[test]
    fn sanitize_redacts_sk_prefix_tokens() {
        let input = "API key is sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890";
        let output = sanitize_tool_output(input);
        assert!(output.contains("[REDACTED]"));
        assert!(!output.contains("sk-ant-api03"));
    }

    #[test]
    fn sanitize_redacts_ghp_tokens() {
        let input = "token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn";
        let output = sanitize_tool_output(input);
        assert!(output.contains("[REDACTED]"));
        assert!(!output.contains("ghp_ABCDE"));
    }

    #[test]
    fn sanitize_redacts_slack_tokens() {
        let input = "SLACK_TOKEN=xoxb-1234567890123-1234567890123-abcdefghijklmnopqrstuv";
        let output = sanitize_tool_output(input);
        assert!(output.contains("[REDACTED]"));
        assert!(!output.contains("xoxb-"));
    }

    #[test]
    fn sanitize_preserves_normal_text() {
        let input = "The function returned successfully with 42 results.";
        let output = sanitize_tool_output(input);
        assert_eq!(input, output);
    }

    #[test]
    fn sanitize_preserves_short_base64() {
        // Short base64 strings should not be redacted
        let input = "hash: abc123def456";
        let output = sanitize_tool_output(input);
        assert_eq!(input, output);
    }

    #[test]
    fn sanitize_redacts_long_mixed_case_base64() {
        let input = "secret: aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789ABCDEFGH";
        let output = sanitize_tool_output(input);
        assert!(output.contains("[REDACTED]"));
    }
}
