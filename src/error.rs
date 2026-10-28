//! Error types for thallus-core.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CoreError {
    #[error("Identity not found at {path}")]
    IdentityNotFound { path: String },

    #[error("Invalid keypair: {reason}")]
    InvalidKeypair { reason: String },

    #[error("MCP error: {reason}")]
    Mcp { reason: String },

    #[error("MCP server '{name}' not found")]
    McpServerNotFound { name: String },

    #[error("Invalid arguments for MCP tool '{tool}': {reason}")]
    McpValidation { tool: String, reason: String },

    #[error("LLM provider error: {reason}")]
    Provider { reason: String },

    #[error("Config error: {reason}")]
    Config { reason: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Built through `From<reqwest::Error>`, which keeps only bounded facts.
    #[error("HTTP error: {0}")]
    Http(HttpError),
}

impl From<reqwest::Error> for CoreError {
    fn from(err: reqwest::Error) -> Self {
        Self::Http(HttpError::from(err))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HttpErrorKind {
    Builder,
    Request,
    Redirect,
    Status,
    Body,
    Decode,
}

/// An HTTP failure reduced to bounded facts: the kind, an optional status
/// code, and connect/timeout flags.
///
/// The request URL, server reason phrases and source errors can all carry
/// credentials or response text, so none of them are retained. The fields are
/// private and the only constructor is `From<reqwest::Error>`, so a value
/// cannot be built around that reduction. `Display`, `Debug` and `source()`
/// are all safe to log and publish.
#[derive(Clone, PartialEq, Eq)]
pub struct HttpError {
    kind: HttpErrorKind,
    status: Option<u16>,
    connect: bool,
    timeout: bool,
}

impl HttpError {
    /// The request failed while connecting.
    #[must_use]
    pub fn is_connect(&self) -> bool {
        self.connect
    }

    /// The request or response timed out.
    #[must_use]
    pub fn is_timeout(&self) -> bool {
        self.timeout
    }

    /// The request could not be built.
    #[must_use]
    pub fn is_builder(&self) -> bool {
        self.kind == HttpErrorKind::Builder
    }

    /// The response body could not be decoded.
    #[must_use]
    pub fn is_decode(&self) -> bool {
        self.kind == HttpErrorKind::Decode
    }

    /// The failure came from a non-success status code.
    #[must_use]
    pub fn is_status(&self) -> bool {
        self.kind == HttpErrorKind::Status
    }

    /// The numeric status code, when the failure carries one.
    #[must_use]
    pub fn status(&self) -> Option<u16> {
        self.status
    }
}

impl From<reqwest::Error> for HttpError {
    fn from(err: reqwest::Error) -> Self {
        let kind = if err.is_builder() {
            HttpErrorKind::Builder
        } else if err.is_redirect() {
            HttpErrorKind::Redirect
        } else if err.is_status() {
            HttpErrorKind::Status
        } else if err.is_decode() {
            HttpErrorKind::Decode
        } else if err.is_body() {
            HttpErrorKind::Body
        } else {
            HttpErrorKind::Request
        };
        Self {
            kind,
            status: err.status().map(|s| s.as_u16()),
            connect: err.is_connect(),
            timeout: err.is_timeout(),
        }
    }
}

impl std::fmt::Display for HttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind {
            HttpErrorKind::Builder => f.write_str("builder error"),
            HttpErrorKind::Request => f.write_str("error sending request"),
            HttpErrorKind::Redirect => f.write_str("error following redirect"),
            HttpErrorKind::Body => f.write_str("request or response body error"),
            HttpErrorKind::Decode => f.write_str("error decoding response body"),
            HttpErrorKind::Status => match self.status {
                Some(code) => write!(f, "HTTP status error ({})", status_text(code)),
                None => f.write_str("HTTP status error"),
            },
        }?;
        if self.timeout {
            f.write_str(" (timed out)")?;
        } else if self.connect {
            f.write_str(" (connect)")?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for HttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpError")
            .field("kind", &self.kind)
            .field("status", &self.status)
            .field("connect", &self.connect)
            .field("timeout", &self.timeout)
            .finish()
    }
}

/// No `source()`: the wrapped `reqwest::Error` is dropped on conversion.
impl std::error::Error for HttpError {}

/// Numeric status code plus its canonical reason, never a server-sent phrase.
pub(crate) fn status_text(code: u16) -> String {
    match reqwest::StatusCode::from_u16(code)
        .ok()
        .and_then(|s| s.canonical_reason())
    {
        Some(reason) => format!("{code} {reason}"),
        None => code.to_string(),
    }
}

/// Safe text for a `reqwest::Error`.
pub(crate) fn http_error_text(err: reqwest::Error) -> String {
    HttpError::from(err).to_string()
}

/// Bounded description of a serde_json error: its category and position,
/// never its message, which can quote the offending value.
pub(crate) fn json_error_text(err: &serde_json::Error) -> String {
    format!(
        "{:?} error at line {} column {}",
        err.classify(),
        err.line(),
        err.column()
    )
}

pub type Result<T> = std::result::Result<T, CoreError>;
