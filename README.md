# thallus-core

Shared library for the Thallus decentralized AI infrastructure. Provides identity, MCP client, and LLM provider abstractions used by familiar, servitor, and other Thallus components.

## Modules

### identity

Ed25519 keypair management.

- `Identity` — signing key with load/save/generate
- `PublicId` — 53-character format: `@<base64(32 bytes)>.ed25519`
- File storage: raw 32-byte `secret.key` with Unix 0600 permissions
- `sign(message)` / `sign_hash(hash)` for message or content-hash signing

### mcp

MCP (Model Context Protocol) client pool.

- `McpClient` trait — `call_tool(name, args) -> Result<ToolCallResult>`
- `StdioMcpClient` — subprocess MCP client in `mcp::stdio`
- `HttpMcpClient` — JSON-RPC 2.0-over-POST client in `mcp::http`
- `McpPool` — thread-safe connection pooling with per-server circuit breakers
- `CircuitBreaker` — configurable failure threshold, half-open timeout
- `sanitize_tool_output()` — credential redaction (`sk-*`, `ghp_*`, base64 patterns)
- `LlmTool` — tool definition extraction from MCP server discovery

Public MCP data and circuit-breaker types are re-exported from `thallus_core::mcp`:

- `ToolDefinition` — an MCP tool's name, optional description, and optional input schema; defined in `mcp::client`
- `ToolCallResult` — a tool call's content blocks and error flag; defined in `mcp::client`
- `ToolContent` — text, image, or resource content returned by a tool; defined in `mcp::client`
- `CircuitBreakerConfig` — failure, recovery, and success thresholds; defined in `mcp::circuit_breaker`
- `CircuitState` — the `Closed`, `Open`, and `HalfOpen` states; defined in `mcp::circuit_breaker`

### provider

LLM provider abstraction layer. Used by Familiar for conversation reasoning. Servitor does not use this module.

- `Provider::name() -> &str` returns the provider name used for metrics.
- `Provider::capabilities() -> ProviderCapabilities` reports tool, vision, streaming, and token-limit support.
- `Provider::chat(system, messages, tools) -> Result<ChatResponse>` sends a non-streaming chat request.
- `Provider::chat_stream(system, messages, tools, on_event) -> Result<ChatResponse>` emits stream events through an `&StreamCallback` and returns the accumulated response.
- `AnthropicProvider` — direct API with retry, caching, prompt cache headers
- `OpenAiCompatProvider` — `/v1/chat/completions` compatible (also serves ollama, local models)
- `MockProvider` — canned responses for testing
- `create_provider(config)` — factory dispatch
- `CompletionCache` — optional response caching with TTL
- `pricing` — per-model token rates with cache-aware cost accounting
- `retry` — exponential backoff with configurable limits

#### Streaming

- `StreamEvent::TextDelta(String)` carries incremental text.
- `StreamEvent::ToolUseStart { id, name }` announces a tool-use block.
- `StreamEvent::ToolInputDelta { id, json_chunk }` carries incremental tool-input JSON.
- `StreamEvent::Done(ChatResponse)` carries the final accumulated response.
- `StreamCallback` is `dyn Fn(StreamEvent) + Send + Sync`.

`AnthropicProvider` and `OpenAiCompatProvider` implement native `chat_stream` handling. The trait's default implementation falls back to `chat`, then emits a text delta and `Done`; `MockProvider` uses that fallback and reports `supports_streaming: false`. Consumers should gate native-streaming behavior on `provider.capabilities().supports_streaming`.

### config

Shared configuration types.

- `LlmConfig` — provider, model, api_key_env, base_url, max_tokens, temperature, retry, cache settings
- `McpServerConfig` — transport type, command, args, env, url, timeout

### error

- `CoreError` enum — Config, Io, InvalidKeypair, IdentityNotFound, MCP, Provider
- `Result<T>` type alias

## Usage

```toml
[dependencies]
thallus-core = { path = "../thallus-core" }
```

```rust
use thallus_core::{Identity, McpPool, create_provider, LlmConfig};

// Load or generate identity
let identity = Identity::load_or_generate("~/.familiar/identity")?;
println!("Public ID: {}", identity.public_id());

// Sign a message
let signature = identity.sign(b"hello world");
```

## Build

```bash
cargo build
cargo test
```
