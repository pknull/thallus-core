//! Completion response cache — skip inference when inputs are identical.
//!
//! Input-keyed LRU cache with TTL. The fingerprint covers model + system prompt +
//! messages (including tool results) + tool definitions. If any part changes, the
//! cache misses automatically — no manual invalidation needed.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use lru::LruCache;
use sha2::{Digest, Sha256};

use super::{ChatResponse, Message, TokenUsage};
use crate::mcp::LlmTool;

/// Cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
}

/// Thread-safe completion response cache.
#[derive(Clone)]
pub struct CompletionCache {
    inner: Arc<Mutex<CacheInner>>,
    ttl: Duration,
}

struct CacheInner {
    cache: LruCache<String, CacheEntry>,
    stats: CacheStats,
}

struct CacheEntry {
    response: ChatResponse,
    inserted_at: Instant,
}

impl CompletionCache {
    /// Create a new cache with the given capacity and TTL.
    pub fn new(max_entries: usize, ttl: Duration) -> Self {
        Self {
            inner: Arc::new(Mutex::new(CacheInner {
                cache: LruCache::new(
                    std::num::NonZeroUsize::new(max_entries)
                        .unwrap_or(std::num::NonZeroUsize::new(1).unwrap()),
                ),
                stats: CacheStats::default(),
            })),
            ttl,
        }
    }

    /// Look up a cached response by request fingerprint.
    pub fn lookup(
        &self,
        model: &str,
        system: &str,
        messages: &[Message],
        tools: &[LlmTool],
    ) -> Option<ChatResponse> {
        let key = fingerprint(model, system, messages, tools);
        let mut inner = self.inner.lock().ok()?;

        // Check if entry exists and whether it's expired
        let expired = inner
            .cache
            .peek(&key)
            .map(|e| e.inserted_at.elapsed() >= self.ttl);

        match expired {
            Some(false) => {
                // Valid cache hit — promote in LRU
                let entry = inner.cache.get(&key).unwrap();
                let response = entry.response.clone();
                inner.stats.hits += 1;
                Some(response)
            }
            Some(true) => {
                // Expired — evict
                inner.cache.pop(&key);
                inner.stats.misses += 1;
                None
            }
            None => {
                inner.stats.misses += 1;
                None
            }
        }
    }

    /// Store a response in the cache.
    pub fn store(
        &self,
        model: &str,
        system: &str,
        messages: &[Message],
        tools: &[LlmTool],
        response: &ChatResponse,
    ) {
        let key = fingerprint(model, system, messages, tools);
        if let Ok(mut inner) = self.inner.lock() {
            inner.cache.put(
                key,
                CacheEntry {
                    response: response.clone(),
                    inserted_at: Instant::now(),
                },
            );
        }
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        self.inner
            .lock()
            .map(|inner| inner.stats.clone())
            .unwrap_or_default()
    }
}

/// Compute a SHA-256 fingerprint of the full request.
///
/// Covers: model name, system prompt, all messages (including tool results),
/// and tool definitions (sorted by name for stability).
fn fingerprint(model: &str, system: &str, messages: &[Message], tools: &[LlmTool]) -> String {
    let mut hasher = Sha256::new();

    hasher.update(model.as_bytes());
    hasher.update(b"|");
    hasher.update(system.as_bytes());
    hasher.update(b"|");

    // Messages — serialize to JSON for stable representation
    if let Ok(json) = serde_json::to_string(messages) {
        hasher.update(json.as_bytes());
    }
    hasher.update(b"|");

    // Tools — sort by name for stability (order shouldn't matter)
    let mut tool_names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    tool_names.sort();
    for name in &tool_names {
        hasher.update(name.as_bytes());
        hasher.update(b",");
    }

    let result = hasher.finalize();
    base64::Engine::encode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{ContentBlock, StopReason};

    fn mock_response(text: &str) -> ChatResponse {
        ChatResponse {
            content: vec![ContentBlock::text(text)],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
        }
    }

    #[test]
    fn store_and_retrieve() {
        let cache = CompletionCache::new(10, Duration::from_secs(60));
        let msgs = vec![Message::user("hello")];
        let resp = mock_response("world");

        cache.store("model", "system", &msgs, &[], &resp);
        let cached = cache.lookup("model", "system", &msgs, &[]);

        assert!(cached.is_some());
        assert_eq!(cached.unwrap().text(), "world");
    }

    #[test]
    fn different_messages_miss() {
        let cache = CompletionCache::new(10, Duration::from_secs(60));
        let msgs1 = vec![Message::user("hello")];
        let msgs2 = vec![Message::user("goodbye")];
        let resp = mock_response("world");

        cache.store("model", "system", &msgs1, &[], &resp);
        let cached = cache.lookup("model", "system", &msgs2, &[]);

        assert!(cached.is_none());
    }

    #[test]
    fn different_system_prompt_misses() {
        let cache = CompletionCache::new(10, Duration::from_secs(60));
        let msgs = vec![Message::user("hello")];
        let resp = mock_response("world");

        cache.store("model", "system-v1", &msgs, &[], &resp);
        let cached = cache.lookup("model", "system-v2", &msgs, &[]);

        assert!(cached.is_none());
    }

    #[test]
    fn expired_entry_misses() {
        let cache = CompletionCache::new(10, Duration::from_millis(1));
        let msgs = vec![Message::user("hello")];
        let resp = mock_response("world");

        cache.store("model", "system", &msgs, &[], &resp);
        std::thread::sleep(Duration::from_millis(10));
        let cached = cache.lookup("model", "system", &msgs, &[]);

        assert!(cached.is_none());
    }

    #[test]
    fn cache_stats_track_correctly() {
        let cache = CompletionCache::new(10, Duration::from_secs(60));
        let msgs = vec![Message::user("hello")];
        let resp = mock_response("world");

        cache.store("model", "system", &msgs, &[], &resp);
        let _ = cache.lookup("model", "system", &msgs, &[]); // hit
        let _ = cache.lookup("model", "system-other", &msgs, &[]); // miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn identical_fingerprints() {
        let msgs = vec![Message::user("hello")];
        let f1 = fingerprint("model", "system", &msgs, &[]);
        let f2 = fingerprint("model", "system", &msgs, &[]);
        assert_eq!(f1, f2);
    }

    #[test]
    fn tool_order_does_not_affect_fingerprint() {
        let msgs = vec![Message::user("hello")];
        let tool_a = LlmTool {
            name: "alpha".into(),
            description: Some("a".into()),
            input_schema: serde_json::json!({}),
        };
        let tool_b = LlmTool {
            name: "beta".into(),
            description: Some("b".into()),
            input_schema: serde_json::json!({}),
        };

        let f1 = fingerprint("model", "system", &msgs, &[tool_a.clone(), tool_b.clone()]);
        let f2 = fingerprint("model", "system", &msgs, &[tool_b, tool_a]);
        assert_eq!(f1, f2);
    }
}
