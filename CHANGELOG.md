# Changelog

All notable changes to thallus-core are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this crate's pre-1.0 versioning treats minor bumps as the breaking-change signal.

## [Unreleased]

### Added

- `MockProvider` call recording for integration-test assertions and one-shot tool-call responses that let mock-driven agent loops terminate.

### Changed

- Removed the unused `ClaudeCodeProvider` and its `claude-code` factory option.
- Updated dependencies to address known vulnerabilities and updated the CI Actions runtime.

### Fixed

- Provider request-build failures are no longer treated as transient transport failures and retried.
- `McpPool` now requires reinitialization after replacing a client and clears initialization state during shutdown.
- Corrected README names for the public MCP clients and OpenAI-compatible provider, and documented the current provider and streaming interfaces.

## [0.3.0] - 2026-04-27

### Added

- `identity::permissions::validate_private_key()` — rejects loading a private key whose file mode allows group or other access (Unix). `Identity::load()` now invokes this validation before reading the key file.
- `mcp::McpClient::drain_notifications()` trait method, with a default implementation returning an empty vector. Lets callers opportunistically pull notifications out of transports that surface them; existing implementations remain compatible without changes.
- `mcp::McpNotification` struct (`method` + optional `params`).
- `README.md`, `CONTRIBUTING.md`, dual `LICENSE-APACHE` / `LICENSE-MIT`, and `.github/workflows/ci.yml` (fmt + clippy + test + build with `-Dwarnings`).

### Changed

- `Identity::load()` now refuses keys with insecure file permissions instead of loading them silently. Existing keys created with `Identity::save()` already use mode 0600 and are unaffected; users who placed keys into the load path with broader modes (e.g. 0644 from a manual copy) must `chmod 600` before upgrading.
- `identity` module-level documentation realigned with the post-reconciliation contract: the node identity signs node-authored messages, not separate per-component "attestations".

### Style

- `cargo fmt` sweep across `mcp/` and `provider/`. No semantic changes.

## [0.2.0] - prior

Earlier history is preserved in `git log`. Highlights: TokenUsage, SSE streaming, retry, prompt cache, pricing, completion cache, mock provider for testing.
