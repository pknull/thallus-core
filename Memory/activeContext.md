# Objective

thallus-core is the skeleton of Thallus: the shared Rust library for
identity (Ed25519 keys, public IDs, signing), MCP clients (stdio/HTTP,
pooling, sanitization, circuit breakers) and LLM provider abstractions.
Familiar consumes all three; Servitor consumes identity and MCP only. It
depends on neither Egregore nor Scry.

# State

Verified 2026-09-15. Version 0.3.0; main at 60f9e24 (single main branch CI
trigger, 2026-08-15); CI green on main. Consumed as a path dependency by
familiar and servitor. Provider module covers Anthropic and
OpenAI-compatible implementations with mock, streaming, retry, caching and
pricing. This repo now carries its own Memory v2 pair; cross-component
decisions stay in the Thallus umbrella (private repo pknull/Thallus).

# Next

- None scheduled; API changes follow the consumers' needs and the
  umbrella's Gate 5 evidence rule.

# Blockers

- None.
