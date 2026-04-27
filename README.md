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
- `StdioTransport` — subprocess spawn with command/args/env
- `HttpTransport` — JSON-RPC 2.0 over POST
- `McpPool` — thread-safe connection pooling with per-server circuit breakers
- `CircuitBreaker` — configurable failure threshold, half-open timeout
- `sanitize_tool_output()` — credential redaction (`sk-*`, `ghp_*`, base64 patterns)
- `LlmTool` — tool definition extraction from MCP server discovery

### provider

LLM provider abstraction layer. Used by Familiar for conversation reasoning. Servitor does not use this module.

- `Provider` trait — `complete(messages, tools) -> Result<ChatResponse>`
- `AnthropicProvider` — direct API with retry, caching, prompt cache headers
- `OpenAiProvider` — `/v1/chat/completions` compatible (also serves ollama, local models)
- `ClaudeCliProvider` — remote trigger via SSH
- `MockProvider` — canned responses for testing
- `create_provider(config)` — factory dispatch
- `CompletionCache` — optional response caching with TTL
- `pricing` — per-model token rates with cache-aware cost accounting
- `retry` — exponential backoff with configurable limits

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
cargo test    # 51 tests
```
