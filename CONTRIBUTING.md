# Contributing to thallus-core

Thanks for your interest. thallus-core is the shared library underneath [Thallus](../) — identity (Ed25519), MCP client abstractions, and LLM provider implementations consumed by Familiar (identity, MCP, providers) and Servitor (identity, MCP — Servitor has no LLM).

## Before You Start

- Read [README.md](README.md) for module layout.
- Understand that this is a **library**, not an application. Changes here affect every downstream consumer.
- For API changes, open an issue first. Breaking changes require a version bump and downstream coordination.

## Development Setup

```bash
git clone <repo>
cd thallus-core
cargo build
cargo test
```

Stable Rust toolchain only.

## Pre-Submit Checklist

```bash
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

CI also runs `cargo audit`.

Note: there's no binary to build — this is a library crate.

## API Stability

thallus-core is a shared dependency. Public APIs should change carefully:

- **Additive changes** (new types, new trait methods with default implementations) — fine
- **Breaking changes** — discuss first, coordinate with consumers (Familiar, Servitor), bump the minor version
- **Rename-only changes** — provide a type alias or deprecation path

If you're unsure whether a change is breaking, assume it is.

## Areas That Need Care

### Identity
`src/identity/` is security-critical. Ed25519 operations go through `ed25519-dalek`. Don't add custom signing, hashing, or encoding without review.

### MCP Client Pool
`src/mcp/pool.rs` is used by both Familiar and Servitor. Changes to the pool API, circuit breaker behaviour, or transport abstraction affect both consumers.

### Provider Trait
`src/provider/` defines the LLM abstraction. Adding a new provider is additive; changing the trait signature is breaking.

### No Application Logic
This crate should stay a library. If your change requires application-specific behaviour (config loading, CLI parsing, daemon logic), it probably belongs in Familiar or Servitor, not here.

## Code Style

- Rust 2021 edition
- `cargo fmt`, `cargo clippy --all-targets`
- Error types use `thiserror` with `CoreError` as the top-level enum
- Public APIs must have doc comments
- Examples in doc comments for non-trivial public functions

## Adding a New LLM Provider

1. Implement the `Provider` trait in `src/provider/<name>.rs`
2. Add the variant to the provider factory in `src/provider/mod.rs`
3. Add pricing data if applicable (`src/provider/pricing.rs`)
4. Add tests exercising the trait methods
5. Document in the README

## Pull Request Process

1. Fork and branch from `master`
2. Make your change; add tests
3. Run the pre-submit checklist
4. If the change is breaking: update Familiar and Servitor's consumption sites in the same PR or in a coordinated PR series
5. Open a PR with a clear description

## License

By contributing, you agree that your contributions will be licensed under [MIT OR Apache-2.0](../LICENSE-MIT).
