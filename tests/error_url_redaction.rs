//! Error text never carries a request URL.
//!
//! An operator-configured URL can hold userinfo or a query token, and consumers
//! publish `CoreError` text, so every HTTP failure path must drop the URL while
//! keeping the error kind.

use thallus_core::mcp::http::HttpMcpClient;
use thallus_core::provider::{Message, StreamEvent};
use thallus_core::{create_provider, CoreError, LlmConfig, McpClient, McpServerConfig};

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
