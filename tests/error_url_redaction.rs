//! Error text never carries a request URL.
//!
//! An operator-configured URL can hold userinfo or a query token, and consumers
//! publish `CoreError` text, so every HTTP failure path must drop the URL while
//! keeping the error kind.

use std::fmt::Write as _;
use std::sync::{Arc, Mutex};

use thallus_core::mcp::http::HttpMcpClient;
use thallus_core::mcp::stdio::StdioMcpClient;
use thallus_core::provider::{Message, StreamEvent};
use thallus_core::{create_provider, CoreError, LlmConfig, McpClient, McpPool, McpServerConfig};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// A local port with nothing listening on it.
fn closed_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

fn secret_url(port: u16) -> String {
    format!("http://alice:s3cret@127.0.0.1:{port}/private-path?token=tok-9f2")
}

/// The host stays off this list: the `Debug` form of a connect failure names
/// the resolved socket address, which is not the URL.
fn assert_no_url(text: &str, url: &str) {
    let fragments = ["alice", "s3cret", "private-path", "tok-9f2", "for url"];
    for leaked in std::iter::once(url).chain(fragments) {
        assert!(
            !text.contains(leaked),
            "error text leaks {leaked:?}: {text}"
        );
    }
}

fn openai_compat(base_url: &str) -> LlmConfig {
    LlmConfig {
        provider: "openai-compat".into(),
        model: "test".into(),
        api_key_env: None,
        base_url: Some(base_url.into()),
        max_tokens: None,
        temperature: None,
        max_retries: Some(0),
        initial_backoff_ms: None,
        max_backoff_ms: None,
    }
}

#[tokio::test]
async fn mcp_http_request_error_omits_url() {
    let url = secret_url(closed_port());
    let config = McpServerConfig {
        transport: "http".into(),
        command: None,
        args: Vec::new(),
        env: Default::default(),
        url: Some(url.clone()),
        timeout_secs: 5,
    };
    let client = HttpMcpClient::new("leaky", &config).unwrap();

    let err = client.ping().await.unwrap_err();

    let text = err.to_string();
    assert_no_url(&text, &url);
    assert_no_url(&format!("{err:?}"), &url);
    assert!(text.contains("error sending request"), "kind lost: {text}");
}

#[tokio::test]
async fn provider_request_error_omits_url() {
    let url = secret_url(closed_port());
    let provider = create_provider(&openai_compat(&url)).unwrap();

    let err = provider
        .chat("", &[Message::user("hi")], &[])
        .await
        .unwrap_err();

    let text = err.to_string();
    assert_no_url(&text, &url);
    assert!(text.contains("error sending request"), "kind lost: {text}");
}

#[tokio::test]
async fn provider_stream_error_omits_url() {
    let url = secret_url(closed_port());
    let provider = create_provider(&openai_compat(&url)).unwrap();

    let err = provider
        .chat_stream("", &[Message::user("hi")], &[], &|_: StreamEvent| {})
        .await
        .unwrap_err();

    let text = err.to_string();
    assert_no_url(&text, &url);
    assert!(text.contains("error sending request"), "kind lost: {text}");
}

#[tokio::test]
async fn provider_build_error_omits_url() {
    // A URL without a host fails at build time; reqwest attaches it to the error.
    let url = "data:alice-s3cret?token=tok-9f2";
    let provider = create_provider(&openai_compat(url)).unwrap();

    let err = provider
        .chat("", &[Message::user("hi")], &[])
        .await
        .unwrap_err();

    let text = err.to_string();
    assert_no_url(&text, url);
    assert!(text.contains("builder error"), "kind lost: {text}");
}

#[tokio::test]
async fn http_error_conversion_omits_url() {
    let url = secret_url(closed_port());
    let source = reqwest::Client::new().get(&url).send().await.unwrap_err();
    assert!(source.url().is_some(), "fixture must carry a URL");

    let err = CoreError::from(source);

    assert_no_url(&err.to_string(), &url);
    assert_no_url(&format!("{err:?}"), &url);
    match err {
        CoreError::Http(e) => assert!(e.is_connect(), "kind lost: {e}"),
        other => panic!("expected CoreError::Http, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Harness: a scripted local server and a log capture.
// ---------------------------------------------------------------------------

/// Marker planted in every server-controlled or caller-controlled value that
/// must never reach error or log text.
const LEAK: &str = "LEAK-7c1d";

/// A credential-bearing URL that a server body or selector can carry.
const SERVER_URL: &str = "http://alice:s3cret@h/p?token=tok-9f2";

fn assert_no_leak(text: &str) {
    assert!(!text.contains(LEAK), "text leaks the marker: {text}");
}

fn response(status_line: &str, content_type: &str, body: &str) -> Vec<u8> {
    format!(
        "HTTP/1.1 {status_line}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .into_bytes()
}

fn json_ok(body: &serde_json::Value) -> Vec<u8> {
    response("200 OK", "application/json", &body.to_string())
}

/// Answer every connection with `handler(request_json)`; returns the port.
async fn serve<F>(handler: F) -> u16
where
    F: Fn(serde_json::Value) -> Vec<u8> + Send + Sync + 'static,
{
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let handler = Arc::new(handler);
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let handler = Arc::clone(&handler);
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                let body = loop {
                    let n = socket.read(&mut chunk).await.unwrap_or(0);
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                    let Some(split) = buf.windows(4).position(|w| w == b"\r\n\r\n") else {
                        continue;
                    };
                    let head = String::from_utf8_lossy(&buf[..split]).to_lowercase();
                    let len = head
                        .lines()
                        .find_map(|l| l.strip_prefix("content-length:"))
                        .and_then(|v| v.trim().parse::<usize>().ok())
                        .unwrap_or(0);
                    if buf.len() >= split + 4 + len {
                        break buf[split + 4..split + 4 + len].to_vec();
                    }
                };
                let request = serde_json::from_slice(&body).unwrap_or(serde_json::Value::Null);
                let _ = socket.write_all(&handler(request)).await;
                let _ = socket.shutdown().await;
            });
        }
    });
    port
}

struct Capture(Arc<Mutex<Vec<String>>>);

struct Line(String);

impl tracing::field::Visit for Line {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        let _ = write!(self.0, "{}={:?} ", field.name(), value);
    }
}

impl tracing::Subscriber for Capture {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        let mut line = Line(String::new());
        event.record(&mut line);
        self.0.lock().unwrap().push(line.0);
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

/// Capture every tracing event on this thread until the guard drops.
fn capture_logs() -> (tracing::subscriber::DefaultGuard, Arc<Mutex<Vec<String>>>) {
    let lines = Arc::new(Mutex::new(Vec::new()));
    let guard = tracing::subscriber::set_default(Capture(Arc::clone(&lines)));
    (guard, lines)
}

fn http_mcp(port: u16) -> McpServerConfig {
    McpServerConfig {
        transport: "http".into(),
        command: None,
        args: Vec::new(),
        env: Default::default(),
        url: Some(format!("http://127.0.0.1:{port}/rpc")),
        timeout_secs: 5,
    }
}

fn sh_mcp(script: &str) -> McpServerConfig {
    McpServerConfig {
        transport: "stdio".into(),
        command: Some("sh".into()),
        args: vec!["-c".into(), script.into()],
        env: Default::default(),
        url: None,
        timeout_secs: 5,
    }
}

fn rpc_id(request: &serde_json::Value) -> u64 {
    request["id"].as_u64().unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Route 1: response and validation text stays out of errors and logs.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn mcp_http_rpc_error_message_stays_out() {
    let port = serve(|req| {
        json_ok(&serde_json::json!({
            "jsonrpc": "2.0", "id": rpc_id(&req),
            "error": {"code": -32001, "message": format!("{LEAK} http://u:p@h/x?token=t")}
        }))
    })
    .await;
    let client = HttpMcpClient::new("srv", &http_mcp(port)).unwrap();

    let text = client.ping().await.unwrap_err().to_string();

    assert_no_leak(&text);
    assert!(text.contains("-32001"), "code lost: {text}");
}

#[tokio::test]
async fn mcp_http_result_decode_error_stays_out() {
    let port = serve(|req| {
        json_ok(&serde_json::json!({
            "jsonrpc": "2.0", "id": rpc_id(&req), "result": {"tools": LEAK}
        }))
    })
    .await;
    let client = HttpMcpClient::new("srv", &http_mcp(port)).unwrap();

    let text = client.list_tools().await.unwrap_err().to_string();

    assert_no_leak(&text);
    assert!(text.contains("tools/list"), "operation lost: {text}");
}

#[tokio::test]
async fn mcp_http_malformed_body_stays_out() {
    let port = serve(|_| response("200 OK", "application/json", &format!("{LEAK} {{"))).await;
    let client = HttpMcpClient::new("srv", &http_mcp(port)).unwrap();

    let text = client.ping().await.unwrap_err().to_string();

    assert_no_leak(&text);
}

#[tokio::test]
async fn mcp_stdio_malformed_line_stays_out() {
    let mut client =
        StdioMcpClient::new("srv", sh_mcp(&format!("read l; echo '{LEAK} {{not json'")));

    let text = client.initialize().await.unwrap_err().to_string();
    let _ = client.shutdown().await;

    assert_no_leak(&text);
}

#[tokio::test]
async fn mcp_stdio_rpc_error_message_stays_out() {
    let script = format!(
        r#"read l; echo '{{"jsonrpc":"2.0","id":1,"error":{{"code":-32002,"message":"{LEAK}"}}}}'"#
    );
    let mut client = StdioMcpClient::new("srv", sh_mcp(&script));

    let text = client.initialize().await.unwrap_err().to_string();
    let _ = client.shutdown().await;

    assert_no_leak(&text);
    assert!(text.contains("-32002"), "code lost: {text}");
}

#[tokio::test]
async fn mcp_stdio_result_decode_error_stays_out() {
    let script = format!(
        r#"read l; echo '{{"jsonrpc":"2.0","id":1,"result":{{"protocolVersion":"x","serverInfo":"{LEAK}"}}}}'"#
    );
    let mut client = StdioMcpClient::new("srv", sh_mcp(&script));

    let text = client.initialize().await.unwrap_err().to_string();
    let _ = client.shutdown().await;

    assert_no_leak(&text);
}

#[tokio::test]
async fn pool_validation_rejection_stays_out_of_error_and_log() {
    let port = serve(|req| {
        let result = match req["method"].as_str() {
            Some("tools/list") => serde_json::json!({"tools": [{
                "name": "run",
                "inputSchema": {
                    "type": "object",
                    "properties": {"mode": {"enum": ["fast", "slow"]}},
                    "additionalProperties": false
                }
            }]}),
            _ => serde_json::json!({
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "s", "version": "1"},
                "capabilities": {}
            }),
        };
        json_ok(&serde_json::json!({"jsonrpc": "2.0", "id": rpc_id(&req), "result": result}))
    })
    .await;
    let mut pool = McpPool::new();
    pool.add_client("srv", &http_mcp(port)).unwrap();
    pool.initialize_all().await.unwrap();
    let (_guard, logs) = capture_logs();

    let bad_value = pool
        .call_tool("srv_run", serde_json::json!({"mode": LEAK}))
        .await
        .unwrap_err();
    let bad_key = pool
        .call_tool("srv_run", serde_json::json!({LEAK: 1}))
        .await
        .unwrap_err();

    for err in [&bad_value, &bad_key] {
        assert!(matches!(err, CoreError::McpValidation { .. }), "{err:?}");
        assert_no_leak(&err.to_string());
    }
    let logs = logs.lock().unwrap().join("\n");
    assert!(logs.contains("rejected"), "warning not captured: {logs}");
    assert_no_leak(&logs);
}

#[tokio::test]
async fn pool_health_check_warning_stays_out() {
    let port = serve(|req| {
        let body = match req["method"].as_str() {
            Some("ping") => serde_json::json!({
                "jsonrpc": "2.0", "id": rpc_id(&req),
                "error": {"code": -32003, "message": LEAK}
            }),
            Some("tools/list") => {
                serde_json::json!({"jsonrpc": "2.0", "id": rpc_id(&req), "result": {"tools": []}})
            }
            _ => serde_json::json!({
                "jsonrpc": "2.0", "id": rpc_id(&req),
                "result": {"protocolVersion": "2024-11-05",
                           "serverInfo": {"name": "s", "version": "1"}, "capabilities": {}}
            }),
        };
        json_ok(&body)
    })
    .await;
    let mut pool = McpPool::new();
    pool.add_client("srv", &http_mcp(port)).unwrap();
    pool.initialize_all().await.unwrap();
    let (_guard, logs) = capture_logs();

    pool.health_check().await;

    let logs = logs.lock().unwrap().join("\n");
    assert!(logs.contains("health check failed"), "not captured: {logs}");
    assert_no_leak(&logs);
}

fn retrying(base_url: String) -> LlmConfig {
    LlmConfig {
        max_retries: Some(2),
        initial_backoff_ms: Some(1),
        max_backoff_ms: Some(2),
        ..openai_compat(&base_url)
    }
}

#[tokio::test]
async fn provider_error_body_stays_out_of_error_and_retry_warning() {
    let port = serve(|_| {
        response(
            "503 Service Unavailable",
            "text/plain",
            &format!("{LEAK} http://alice:s3cret@h/p?token=tok-9f2"),
        )
    })
    .await;
    let provider = create_provider(&retrying(format!("http://127.0.0.1:{port}"))).unwrap();
    let (_guard, logs) = capture_logs();

    let err = provider
        .chat("", &[Message::user("hi")], &[])
        .await
        .unwrap_err();

    let text = err.to_string();
    assert_no_leak(&text);
    assert_no_url(&text, SERVER_URL);
    assert!(text.contains("503"), "status lost: {text}");
    let logs = logs.lock().unwrap().join("\n");
    assert!(logs.contains("retryable API error"), "no retry: {logs}");
    assert_no_leak(&logs);
    assert_no_url(&logs, SERVER_URL);
}

#[tokio::test]
async fn provider_transport_retry_warning_omits_url() {
    let url = secret_url(closed_port());
    let provider = create_provider(&retrying(url.clone())).unwrap();
    let (_guard, logs) = capture_logs();

    let err = provider
        .chat("", &[Message::user("hi")], &[])
        .await
        .unwrap_err();

    assert_no_url(&err.to_string(), &url);
    let logs = logs.lock().unwrap().join("\n");
    assert!(logs.contains("will retry"), "no retry warning: {logs}");
    assert!(logs.contains("error sending request"), "kind lost: {logs}");
    assert_no_url(&logs, &url);
}

#[tokio::test]
async fn provider_malformed_success_body_stays_out() {
    let port = serve(|_| response("200 OK", "application/json", &format!("{LEAK} {{"))).await;
    let provider = create_provider(&openai_compat(&format!("http://127.0.0.1:{port}"))).unwrap();

    let err = provider
        .chat("", &[Message::user("hi")], &[])
        .await
        .unwrap_err();

    assert_no_leak(&err.to_string());
}

// ---------------------------------------------------------------------------
// Route 2: the HTTP error payload is safe through Display, Debug and source().
// ---------------------------------------------------------------------------

#[tokio::test]
async fn http_error_status_reason_phrase_stays_out_of_debug() {
    let port = serve(|_| {
        format!("HTTP/1.1 418 {LEAK}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .into_bytes()
    })
    .await;
    let url = format!("http://alice:s3cret@127.0.0.1:{port}/private-path?token=tok-9f2");
    let source = reqwest::Client::new()
        .get(&url)
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap_err();

    let err = CoreError::from(source);

    for text in [err.to_string(), format!("{err:?}"), format!("{err:#?}")] {
        assert_no_leak(&text);
        assert_no_url(&text, &url);
    }
    assert!(err.to_string().contains("418"), "status lost: {err}");
    let mut source = std::error::Error::source(&err);
    while let Some(s) = source {
        assert_no_leak(&s.to_string());
        assert_no_leak(&format!("{s:?}"));
        source = s.source();
    }
}

#[tokio::test]
async fn http_error_decode_source_stays_out() {
    let port = serve(|_| response("200 OK", "application/json", &format!("\"{LEAK}\""))).await;
    let url = format!("http://alice:s3cret@127.0.0.1:{port}/private-path?token=tok-9f2");
    let source = reqwest::Client::new()
        .get(&url)
        .send()
        .await
        .unwrap()
        .json::<Vec<u8>>()
        .await
        .unwrap_err();

    let err = CoreError::from(source);

    for text in [err.to_string(), format!("{err:?}")] {
        assert_no_leak(&text);
        assert_no_url(&text, &url);
    }
    let mut source = std::error::Error::source(&err);
    while let Some(s) = source {
        assert_no_leak(&s.to_string());
        assert_no_leak(&format!("{s:?}"));
        source = s.source();
    }
}

// ---------------------------------------------------------------------------
// Route 3: every SSE error variant maps to safe text.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sse_invalid_content_type_stays_out() {
    let port = serve(|_| response("200 OK", LEAK, "x")).await;
    let provider = create_provider(&openai_compat(&format!("http://127.0.0.1:{port}"))).unwrap();

    let err = provider
        .chat_stream("", &[Message::user("hi")], &[], &|_: StreamEvent| {})
        .await
        .unwrap_err();

    assert_no_leak(&err.to_string());
    assert!(err.to_string().contains("SSE"), "{err}");
}

#[tokio::test]
async fn sse_invalid_status_stays_out() {
    let port = serve(|_| response("401 Unauthorized", "text/plain", LEAK)).await;
    let provider = create_provider(&openai_compat(&format!("http://127.0.0.1:{port}"))).unwrap();

    let err = provider
        .chat_stream("", &[Message::user("hi")], &[], &|_: StreamEvent| {})
        .await
        .unwrap_err();

    assert_no_leak(&err.to_string());
    assert!(err.to_string().contains("401"), "status lost: {err}");
}

// ---------------------------------------------------------------------------
// Route 4: configuration text and Debug do not echo configured values.
// ---------------------------------------------------------------------------

#[test]
fn unknown_provider_selector_is_not_echoed() {
    let mut config = openai_compat("http://127.0.0.1:1");
    config.provider = format!("{LEAK}-http://alice:s3cret@h/p?token=tok-9f2");

    let err = create_provider(&config).err().expect("must be rejected");

    assert_no_leak(&err.to_string());
    assert_no_url(&err.to_string(), SERVER_URL);
}

#[test]
fn unknown_transport_selector_is_not_echoed() {
    let mut config = http_mcp(1);
    config.transport = LEAK.into();

    let err = McpPool::new().add_client("srv", &config).unwrap_err();

    assert_no_leak(&err.to_string());
}

#[test]
fn missing_api_key_env_name_is_not_echoed() {
    let mut config = openai_compat("http://127.0.0.1:1");
    config.provider = "anthropic".into();
    config.api_key_env = Some(format!("{LEAK}_NOT_SET"));

    let err = create_provider(&config).err().expect("must be rejected");

    assert_no_leak(&err.to_string());
}

#[test]
fn config_debug_redacts_every_free_text_field() {
    let url = "http://alice:s3cret@host/private-path?token=tok-9f2";
    let mut mcp = http_mcp(1);
    mcp.transport = format!("{LEAK}-{url}");
    mcp.command = Some(format!("{LEAK}-cmd"));
    mcp.url = Some(url.into());
    mcp.env.insert(format!("{LEAK}_ENV_NAME"), LEAK.into());
    mcp.args = vec![format!("--token={LEAK}")];
    let mut llm = openai_compat(url);
    llm.provider = format!("{LEAK}-{url}");
    llm.model = format!("{LEAK}-model");
    llm.api_key_env = Some(format!("{LEAK}_KEY"));

    for text in [
        format!("{mcp:?}"),
        format!("{mcp:#?}"),
        format!("{llm:?}"),
        format!("{llm:#?}"),
    ] {
        assert_no_url(&text, url);
        assert_no_leak(&text);
    }
}

#[test]
fn config_debug_keeps_known_selectors_and_counts() {
    let mut mcp = http_mcp(1);
    mcp.env.insert("A".into(), "x".into());
    mcp.env.insert("B".into(), "y".into());
    let llm = openai_compat("http://h");

    let mcp = format!("{mcp:?}");
    let llm = format!("{llm:?}");

    assert!(mcp.contains("http"), "{mcp}");
    assert!(mcp.contains('2'), "env count lost: {mcp}");
    assert!(llm.contains("openai-compat"), "{llm}");
}

fn anthropic(base_url: &str, key_env: &str) -> LlmConfig {
    std::env::set_var(key_env, "test-key");
    LlmConfig {
        provider: "anthropic".into(),
        api_key_env: Some(key_env.into()),
        ..openai_compat(base_url)
    }
}

#[tokio::test]
async fn anthropic_build_error_omits_url() {
    let url = "data:alice-s3cret?token=tok-9f2";
    let provider = create_provider(&anthropic(url, "THALLUS_TEST_KEY_BUILD")).unwrap();

    let err = provider
        .chat("", &[Message::user("hi")], &[])
        .await
        .unwrap_err();

    assert_no_url(&err.to_string(), url);
    assert!(err.to_string().contains("builder error"), "{err}");
}

#[tokio::test]
async fn anthropic_request_error_omits_url() {
    let url = secret_url(closed_port());
    let provider = create_provider(&anthropic(&url, "THALLUS_TEST_KEY_REQ")).unwrap();

    let err = provider
        .chat("", &[Message::user("hi")], &[])
        .await
        .unwrap_err();

    assert_no_url(&err.to_string(), &url);
    assert!(err.to_string().contains("error sending request"), "{err}");
}

#[tokio::test]
async fn anthropic_stream_transport_error_omits_url() {
    let url = secret_url(closed_port());
    let provider = create_provider(&anthropic(&url, "THALLUS_TEST_KEY_SSE")).unwrap();

    let err = provider
        .chat_stream("", &[Message::user("hi")], &[], &|_: StreamEvent| {})
        .await
        .unwrap_err();

    assert_no_url(&err.to_string(), &url);
    assert!(err.to_string().contains("error sending request"), "{err}");
}

#[tokio::test]
async fn anthropic_stream_invalid_content_type_stays_out() {
    let port = serve(|_| response("200 OK", LEAK, "x")).await;
    let base = format!("http://127.0.0.1:{port}");
    let provider = create_provider(&anthropic(&base, "THALLUS_TEST_KEY_CT")).unwrap();

    let err = provider
        .chat_stream("", &[Message::user("hi")], &[], &|_: StreamEvent| {})
        .await
        .unwrap_err();

    assert_no_leak(&err.to_string());
    assert!(err.to_string().contains("SSE"), "{err}");
}

// ---------------------------------------------------------------------------
// Review round: argument keys, response-supplied names, JSON errors.
// ---------------------------------------------------------------------------

fn init_result(name: &str, version: &str) -> serde_json::Value {
    serde_json::json!({
        "protocolVersion": "2024-11-05",
        "serverInfo": {"name": name, "version": version},
        "capabilities": {}
    })
}

/// A pool over a scripted HTTP server exposing one tool `tool_name` with `schema`.
async fn pool_with_schema(
    tool_name: &str,
    schema: serde_json::Value,
) -> Result<McpPool, CoreError> {
    let tool_name = tool_name.to_string();
    let port = serve(move |req| {
        let result = match req["method"].as_str() {
            Some("tools/list") => {
                serde_json::json!({"tools": [{"name": tool_name, "inputSchema": schema}]})
            }
            _ => init_result("s", "1"),
        };
        json_ok(&serde_json::json!({"jsonrpc": "2.0", "id": rpc_id(&req), "result": result}))
    })
    .await;
    let mut pool = McpPool::new();
    pool.add_client("srv", &http_mcp(port)).unwrap();
    pool.initialize_all().await?;
    Ok(pool)
}

#[tokio::test]
async fn validation_path_replaces_undeclared_keys() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "known": {"type": "object", "additionalProperties": {"type": "integer"}},
            "pat": {"type": "object", "patternProperties": {"^x": {"type": "integer"}}},
            "list": {"type": "array", "items": {
                "type": "object", "additionalProperties": {"type": "integer"}}}
        },
        "additionalProperties": {"type": "integer"}
    });
    let pool = pool_with_schema("run", schema).await.unwrap();
    let (_guard, logs) = capture_logs();
    let key = format!("x-{LEAK}");

    let cases = [
        (serde_json::json!({ (key.clone()): "s" }), "/<key>"),
        (
            serde_json::json!({"known": { (key.clone()): "s" }}),
            "/known/<key>",
        ),
        (
            serde_json::json!({"pat": { (key.clone()): "s" }}),
            "/pat/<key>",
        ),
        (
            serde_json::json!({"list": [{ (key.clone()): "s" }]}),
            "/list/0/<key>",
        ),
    ];
    for (args, expected) in cases {
        let err = pool.call_tool("srv_run", args).await.unwrap_err();
        for text in [err.to_string(), format!("{err:?}"), format!("{err:#?}")] {
            assert_no_leak(&text);
        }
        assert!(
            err.to_string().contains(expected),
            "{expected} missing: {err}"
        );
    }
    let logs = logs.lock().unwrap().join("\n");
    assert!(logs.contains("rejected"), "{logs}");
    assert_no_leak(&logs);
}

#[tokio::test]
async fn schema_compile_error_is_bounded_and_actionable() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"mode": {"type": 5}}
    });
    let tool = format!("{LEAK}-http://alice:s3cret@h/p?token=tok-9f2");
    let (_guard, logs) = capture_logs();

    let err = pool_with_schema(&tool, schema)
        .await
        .err()
        .expect("must fail");

    let text = err.to_string();
    assert_no_leak(&text);
    assert_no_url(&text, SERVER_URL);
    assert!(text.contains("srv"), "configured server name lost: {text}");
    assert!(
        text.contains("/properties/mode/type"),
        "location lost: {text}"
    );
    assert_no_leak(&logs.lock().unwrap().join("\n"));
    assert_no_leak(&format!("{err:?}"));
}

#[tokio::test]
async fn schema_compile_error_location_hides_undeclared_keys() {
    let schema = serde_json::json!({
        "type": "object",
        "patternProperties": {(format!("^{LEAK}")): {"type": 5}}
    });

    let err = pool_with_schema("run", schema)
        .await
        .err()
        .expect("must fail");

    assert_no_leak(&err.to_string());
}

#[tokio::test]
async fn http_init_log_omits_server_supplied_identity() {
    let port = serve(|req| {
        json_ok(&serde_json::json!({
            "jsonrpc": "2.0", "id": rpc_id(&req), "result": init_result(LEAK, LEAK)
        }))
    })
    .await;
    let mut client = HttpMcpClient::new("srv", &http_mcp(port)).unwrap();
    let (_guard, logs) = capture_logs();

    client.initialize().await.unwrap();

    let logs = logs.lock().unwrap().join("\n");
    assert!(logs.contains("initialized"), "not captured: {logs}");
    assert!(logs.contains("srv"), "configured name lost: {logs}");
    assert_no_leak(&logs);
}

#[tokio::test]
async fn stdio_init_log_omits_server_supplied_identity() {
    let script = format!(
        r#"read l; echo '{{"jsonrpc":"2.0","id":1,"result":{{"protocolVersion":"x","serverInfo":{{"name":"{LEAK}","version":"{LEAK}"}}}}}}'; read l"#
    );
    let mut client = StdioMcpClient::new("srv", sh_mcp(&script));
    let (_guard, logs) = capture_logs();

    let result = client.initialize().await;
    let _ = client.shutdown().await;

    result.unwrap();
    let logs = logs.lock().unwrap().join("\n");
    assert!(logs.contains("initialized"), "not captured: {logs}");
    assert_no_leak(&logs);
}

#[test]
fn json_error_conversion_stays_out() {
    let source = serde_json::from_str::<Vec<u8>>(&format!("\"{LEAK}\"")).unwrap_err();
    assert!(
        source.to_string().contains(LEAK),
        "fixture must quote the value"
    );

    let err = CoreError::from(source);

    for text in [err.to_string(), format!("{err:?}"), format!("{err:#?}")] {
        assert_no_leak(&text);
    }
    assert!(std::error::Error::source(&err).is_none());
    assert!(err.to_string().contains("line 1"), "position lost: {err}");
}

#[test]
fn invalid_public_id_is_not_echoed() {
    let id = thallus_core::PublicId(format!("{LEAK}-http://alice:s3cret@h/p?token=tok-9f2"));
    let err = id.to_verifying_key().unwrap_err();
    assert_no_leak(&err.to_string());

    let id = thallus_core::PublicId(format!("@{LEAK}!.ed25519"));
    let err = id.to_verifying_key().unwrap_err();
    assert_no_leak(&err.to_string());
}
