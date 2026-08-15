# Thallus Core

Shared Rust library for Thallus identity, MCP clients, and LLM provider abstractions. Familiar and Servitor consume this crate; it does not depend on Egregore or Scry.

## Module Map

| Module | Location | Purpose |
|--------|----------|---------|
| identity | `src/identity/` | Ed25519 key management, public IDs, signing, and private-key permission checks |
| mcp | `src/mcp/` | MCP client trait and types, stdio/HTTP clients, pooling, output sanitization, and circuit breakers |
| provider | `src/provider/` | LLM provider trait, Anthropic and OpenAI-compatible implementations, mock provider, streaming, retry, caching, and pricing |
| config | `src/config.rs` | Shared LLM and MCP server configuration types |
| error | `src/error.rs` | `CoreError` and the crate-wide `Result<T>` alias |

## Consumer Boundary

Extract a primitive only when two live consumers need the same invariant. Keep consumer-specific orchestration and policy in Familiar or Servitor rather than adding dependencies from this crate to Egregore or Scry.

## Build and Test

```bash
cargo build
cargo test
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

## Code Style

- Rust 2021 edition
- Async interfaces use Tokio and `async-trait`
- Errors use `thiserror` through `CoreError`
