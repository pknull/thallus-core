//! MCP client pool -- manages multiple MCP server connections.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use jsonschema::error::ValidationErrorKind;
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
    /// Servers for which `initialize_all` has completed successfully. Used to gate
    /// `health_check` so it does not poll un-initialized clients (which would trip the
    /// circuit breaker before bootstrap is done). Cleared on `shutdown_all`; re-adding
    /// the same server name via `add_client` clears its prior entry so a replacement
    /// client must re-initialize before its first health probe.
    initialized_servers: HashSet<String>,
    /// Circuit breakers per server.
    circuit_breakers: HashMap<String, RwLock<CircuitBreaker>>,
    /// All tools with prefixed names, mapped to their server.
    tools: HashMap<String, RegisteredTool>,
}

struct RegisteredTool {
    server_name: String,
    /// Position in the server's tool list, for diagnostics.
    index: usize,
    definition: ToolDefinition,
    validator: Option<JSONSchema>,
}

impl RegisteredTool {
    /// Generated identity for diagnostics. The tool's own name came from the
    /// server and can carry anything an identifier allows, so it is never used.
    fn label(&self) -> String {
        format!("tool #{} from server '{}'", self.index, self.server_name)
    }
}

impl McpPool {
    /// Create a new empty pool.
    pub fn new() -> Self {
        Self {
            clients: HashMap::new(),
            initialized_servers: HashSet::new(),
            circuit_breakers: HashMap::new(),
            tools: HashMap::new(),
        }
    }

    /// Add a client to the pool.
    pub fn add_client(&mut self, name: &str, config: &McpServerConfig) -> Result<()> {
        let client: Box<dyn McpClient> = match config.transport.as_str() {
            "stdio" => Box::new(StdioMcpClient::new(name, config.clone())),
            "http" => Box::new(HttpMcpClient::new(name, config)?),
            _ => {
                return Err(CoreError::Mcp {
                    reason: "unknown transport".to_string(),
                })
            }
        };

        self.clients
            .insert(name.to_string(), Arc::new(RwLock::new(client)));
        // Re-adding a server name overwrites the client entry above; make sure any
        // prior init state is cleared so the replacement must re-initialize before
        // health_check pings it.
        self.initialized_servers.remove(name);

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
            let mut skipped = 0usize;
            for (index, tool) in tools.into_iter().enumerate() {
                // Registered names become part of diagnostics, so only bounded
                // identifiers are accepted from the server.
                if !is_valid_tool_name(&tool.name) {
                    skipped += 1;
                    continue;
                }
                let prefixed_name = tool.prefixed_name(&name);
                let validator = compile_validator(&name, index, &tool)?;
                self.tools.insert(
                    prefixed_name,
                    RegisteredTool {
                        server_name: name.clone(),
                        index,
                        definition: tool,
                        validator,
                    },
                );
            }

            if skipped > 0 {
                tracing::warn!(
                    server = %name,
                    skipped,
                    "skipped MCP tools whose advertised names are not valid identifiers"
                );
            }

            self.initialized_servers.insert(name.clone());
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
                reason: "unknown tool".to_string(),
            })?;
        let tool_name = tool.definition.name.as_str();
        let server_name = &tool.server_name;

        // Check circuit breaker
        if let Some(cb) = self.circuit_breakers.get(server_name) {
            let mut cb = cb.write().await;
            if !cb.should_allow() {
                tracing::warn!(
                    server = %server_name,
                    tool = %tool.label(),
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

        validate_arguments(tool, &arguments)?;

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
            if !self.initialized_servers.contains(name) {
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

    /// Shutdown all clients and clear the init-tracking set so a subsequent
    /// `health_check` does not ping shut-down servers.
    pub async fn shutdown_all(&mut self) -> Result<()> {
        for client in self.clients.values() {
            let mut client = client.write().await;
            let _ = client.shutdown().await;
        }
        self.initialized_servers.clear();
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

fn compile_validator(
    server: &str,
    index: usize,
    tool: &ToolDefinition,
) -> Result<Option<JSONSchema>> {
    let Some(schema) = tool.input_schema.as_ref() else {
        return Ok(None);
    };

    JSONSchema::options()
        .compile(schema)
        .map(Some)
        .map_err(|error| {
            // The server's tool name and the schema's own text are response
            // data; report the configured server, the tool's position, the
            // failed rule and its location.
            let mut reason = format!(
                "invalid input schema for tool #{index} from server '{server}': {}",
                rule_name(&error.kind)
            );
            if let ValidationErrorKind::Type { kind } = &error.kind {
                reason.push_str(&format!(" (expected {})", type_kind_text(kind)));
            }
            reason.push_str(&format!(
                " at {}",
                render_path(&error.instance_path, &declared_names(schema))
            ));
            CoreError::Mcp { reason }
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

/// The failed JSON Schema rule as fixed text; no payload is read.
fn rule_name(kind: &ValidationErrorKind) -> &'static str {
    use ValidationErrorKind as K;
    match kind {
        K::AdditionalItems { .. } => "additionalItems",
        K::AdditionalProperties { .. } => "additionalProperties",
        K::AnyOf => "anyOf",
        K::BacktrackLimitExceeded { .. } => "backtrackLimitExceeded",
        K::Constant { .. } => "const",
        K::Contains => "contains",
        K::ContentEncoding { .. } => "contentEncoding",
        K::ContentMediaType { .. } => "contentMediaType",
        K::Custom { .. } => "custom",
        K::Enum { .. } => "enum",
        K::ExclusiveMaximum { .. } => "exclusiveMaximum",
        K::ExclusiveMinimum { .. } => "exclusiveMinimum",
        K::FalseSchema => "falseSchema",
        K::FileNotFound { .. } => "fileNotFound",
        K::Format { .. } => "format",
        K::FromUtf8 { .. } => "fromUtf8",
        K::Utf8 { .. } => "utf8",
        K::JSONParse { .. } => "jsonParse",
        K::InvalidReference { .. } => "invalidReference",
        K::InvalidURL { .. } => "invalidUrl",
        K::MaxItems { .. } => "maxItems",
        K::Maximum { .. } => "maximum",
        K::MaxLength { .. } => "maxLength",
        K::MaxProperties { .. } => "maxProperties",
        K::MinItems { .. } => "minItems",
        K::Minimum { .. } => "minimum",
        K::MinLength { .. } => "minLength",
        K::MinProperties { .. } => "minProperties",
        K::MultipleOf { .. } => "multipleOf",
        K::Not { .. } => "not",
        K::OneOfMultipleValid => "oneOf (multiple valid)",
        K::OneOfNotValid => "oneOf (none valid)",
        K::Pattern { .. } => "pattern",
        K::PropertyNames { .. } => "propertyNames",
        K::Required { .. } => "required",
        K::Schema => "schema",
        K::Type { .. } => "type",
        K::UnevaluatedProperties { .. } => "unevaluatedProperties",
        K::UniqueItems => "uniqueItems",
        K::UnknownReferenceScheme { .. } => "unknownReferenceScheme",
        K::Resolver { .. } => "resolver",
    }
}

/// The expected JSON type names, e.g. `string` or `string | null`.
fn type_kind_text(kind: &jsonschema::error::TypeKind) -> String {
    use jsonschema::error::TypeKind;
    match kind {
        TypeKind::Single(t) => t.to_string(),
        TypeKind::Multiple(set) => set
            .into_iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(" | "),
    }
}

/// JSON Schema keywords that may appear as path segments.
const SCHEMA_KEYWORDS: &[&str] = &[
    "$defs",
    "$id",
    "$ref",
    "$schema",
    "additionalItems",
    "additionalProperties",
    "allOf",
    "anyOf",
    "contains",
    "definitions",
    "dependencies",
    "else",
    "if",
    "items",
    "not",
    "oneOf",
    "patternProperties",
    "properties",
    "propertyNames",
    "then",
    "const",
    "enum",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "format",
    "maxItems",
    "maxLength",
    "maxProperties",
    "maximum",
    "minItems",
    "minLength",
    "minProperties",
    "minimum",
    "multipleOf",
    "pattern",
    "required",
    "type",
    "uniqueItems",
    "unevaluatedProperties",
];

/// A tool name the pool accepts: 1-64 ASCII letters, digits, `_`, `.` or `-`,
/// starting with a letter or digit.
fn is_valid_tool_name(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    name.len() <= 64
        && first.is_ascii_alphanumeric()
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
}

/// The schema drafts this build compiles; see `draft_of`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Draft {
    Four,
    Six,
    Seven,
}

/// The draft jsonschema 0.18 compiles `schema` with: the `$schema` URL (a
/// trailing `#` ignored) when it names an enabled draft, else Draft 7. The
/// 2019-09 and 2020-12 drafts sit behind crate features this crate does not
/// enable, so their URLs also fall back to Draft 7.
fn draft_of(schema: &serde_json::Value) -> Draft {
    match schema
        .get("$schema")
        .and_then(serde_json::Value::as_str)
        .map(|url| url.trim_end_matches('#'))
    {
        Some("http://json-schema.org/draft-06/schema") => Draft::Six,
        Some("http://json-schema.org/draft-04/schema") => Draft::Four,
        _ => Draft::Seven,
    }
}

/// Property names the tool's schema declares.
///
/// Only the applicator keywords of the draft the compiler uses are entered.
/// Keywords that draft ignores, annotation and data keywords (`default`,
/// `examples`, `const`, `enum`, `title`, `description`) and unknown keywords
/// are never visited. `$ref` is not followed; an object with a string `$ref`
/// contributes only its `definitions`, since these drafts ignore siblings.
fn declared_names(root: &serde_json::Value) -> HashSet<&str> {
    use serde_json::Value;

    struct Walk<'a> {
        draft: Draft,
        out: HashSet<&'a str>,
    }

    impl<'a> Walk<'a> {
        fn each_value(&mut self, value: &'a Value) {
            if let Value::Object(map) = value {
                for v in map.values() {
                    self.schema(v);
                }
            }
        }

        fn schema(&mut self, value: &'a Value) {
            let Value::Object(map) = value else {
                return;
            };
            if let Some(defs) = map.get("definitions") {
                self.each_value(defs);
            }
            if map.get("$ref").is_some_and(Value::is_string) {
                return;
            }
            if let Some(Value::Object(props)) = map.get("properties") {
                self.out.extend(props.keys().map(String::as_str));
                for v in props.values() {
                    self.schema(v);
                }
            }
            if let Some(group) = map.get("patternProperties") {
                self.each_value(group);
            }
            // `dependencies` values are schemas or property lists; lists are
            // not objects and are skipped.
            if let Some(Value::Object(deps)) = map.get("dependencies") {
                for v in deps.values() {
                    self.schema(v);
                }
            }
            let mut single = vec!["additionalProperties", "additionalItems"];
            if self.draft >= Draft::Six {
                single.extend(["contains", "propertyNames"]);
            }
            if self.draft >= Draft::Seven {
                single.extend(["if", "then", "else"]);
            }
            single.push("not");
            for key in single {
                if let Some(sub) = map.get(key) {
                    self.schema(sub);
                }
            }
            for key in ["items", "allOf", "anyOf", "oneOf"] {
                match map.get(key) {
                    Some(Value::Array(subs)) => {
                        for v in subs {
                            self.schema(v);
                        }
                    }
                    Some(sub) => self.schema(sub),
                    None => {}
                }
            }
        }
    }

    let mut walk = Walk {
        draft: draft_of(root),
        out: HashSet::new(),
    };
    walk.schema(root);
    walk.out
}

/// Render a JSON pointer keeping array indices, schema keywords and declared
/// property names. Any other object key is caller- or server-supplied data and
/// becomes `<key>`.
fn render_path(path: &jsonschema::paths::JSONPointer, declared: &HashSet<&str>) -> String {
    use jsonschema::paths::PathChunk;
    let mut out = String::new();
    for chunk in path {
        out.push('/');
        match chunk {
            PathChunk::Index(i) => out.push_str(&i.to_string()),
            PathChunk::Keyword(k) => out.push_str(k),
            PathChunk::Property(name) => {
                if declared.contains(&**name) || SCHEMA_KEYWORDS.contains(&&**name) {
                    out.push_str(&name.replace('~', "~0").replace('/', "~1"));
                } else {
                    out.push_str("<key>");
                }
            }
        }
    }
    if out.is_empty() {
        out.push('/');
    }
    out
}

fn validate_arguments(tool: &RegisteredTool, arguments: &serde_json::Value) -> Result<()> {
    let Some(validator) = tool.validator.as_ref() else {
        return Ok(());
    };

    let declared = tool
        .definition
        .input_schema
        .as_ref()
        .map(declared_names)
        .unwrap_or_default();
    let details = match validator.validate(arguments) {
        Ok(()) => return Ok(()),
        Err(errors) => errors
            .take(5)
            .map(|error| {
                // Field path and failed rule only: the message quotes the
                // rejected value and the path can hold caller-chosen keys.
                let rule = rule_name(&error.kind);
                format!(
                    "{}: {rule} failed",
                    render_path(&error.instance_path, &declared)
                )
            })
            .collect::<Vec<_>>()
            .join("; "),
    };

    let label = tool.label();
    tracing::warn!(tool = %label, reason = %details, "rejected MCP tool call");
    Err(CoreError::McpValidation {
        tool: label,
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
        let validator = compile_validator("fake", 0, &definition).unwrap();
        pool.tools.insert(
            "shell_execute".to_string(),
            RegisteredTool {
                server_name: "shell".to_string(),
                index: 0,
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
                index: 0,
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
