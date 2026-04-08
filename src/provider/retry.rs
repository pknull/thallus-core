//! Retry policy with exponential backoff and jitter.
//!
//! Used by HTTP-based providers (Anthropic, OpenAI) to handle transient
//! failures: rate limits (429), server errors (5xx), timeouts, and
//! connection failures.

use std::time::Duration;

use rand::Rng;

use crate::config::LlmConfig;

/// Retry policy configuration.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of retries (not counting the initial attempt).
    pub max_retries: u32,
    /// Base delay for the first retry.
    pub initial_backoff: Duration,
    /// Maximum delay cap.
    pub max_backoff: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 2,
            initial_backoff: Duration::from_millis(200),
            max_backoff: Duration::from_secs(2),
        }
    }
}

impl RetryPolicy {
    /// Create from LLM config, falling back to defaults.
    pub fn from_config(config: &LlmConfig) -> Self {
        Self {
            max_retries: config.max_retries.unwrap_or(2),
            initial_backoff: Duration::from_millis(config.initial_backoff_ms.unwrap_or(200)),
            max_backoff: Duration::from_millis(config.max_backoff_ms.unwrap_or(2000)),
        }
    }

    /// Calculate backoff duration for a given attempt (1-indexed).
    ///
    /// Uses exponential backoff with +-25% jitter:
    ///   delay = min(max_backoff, initial_backoff * 2^(attempt-1)) * (0.75..1.25)
    pub fn backoff_for_attempt(&self, attempt: u32) -> Duration {
        let base_ms = self.initial_backoff.as_millis() as u64;
        let multiplier = 1u64
            .checked_shl(attempt.saturating_sub(1))
            .unwrap_or(u64::MAX);
        let raw = base_ms.saturating_mul(multiplier);
        let capped = raw.min(self.max_backoff.as_millis() as u64);

        // Jitter: +-25%
        let jitter_factor = rand::thread_rng().gen_range(0.75..1.25);
        let jittered = (capped as f64 * jitter_factor) as u64;

        Duration::from_millis(jittered.max(1))
    }
}

/// Check if an HTTP status code is retryable.
pub fn is_retryable_status(status: u16) -> bool {
    matches!(status, 408 | 429 | 500 | 502 | 503 | 504)
}

/// Check if a reqwest error is retryable (connection/timeout).
pub fn is_retryable_error(err: &reqwest::Error) -> bool {
    err.is_connect() || err.is_timeout() || err.is_request()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.max_retries, 2);
        assert_eq!(policy.initial_backoff, Duration::from_millis(200));
        assert_eq!(policy.max_backoff, Duration::from_secs(2));
    }

    #[test]
    fn backoff_increases_exponentially() {
        let policy = RetryPolicy {
            max_retries: 5,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(10),
        };

        // Run multiple samples to account for jitter
        for _ in 0..10 {
            let d1 = policy.backoff_for_attempt(1).as_millis();
            let d2 = policy.backoff_for_attempt(2).as_millis();
            let d3 = policy.backoff_for_attempt(3).as_millis();

            // With jitter, d1 ~ 75-125ms, d2 ~ 150-250ms, d3 ~ 300-500ms
            assert!(d1 >= 50 && d1 <= 200, "d1={}", d1);
            assert!(d2 >= 100 && d2 <= 400, "d2={}", d2);
            assert!(d3 >= 200 && d3 <= 700, "d3={}", d3);
        }
    }

    #[test]
    fn backoff_respects_cap() {
        let policy = RetryPolicy {
            max_retries: 10,
            initial_backoff: Duration::from_millis(500),
            max_backoff: Duration::from_millis(1000),
        };

        for _ in 0..10 {
            let d = policy.backoff_for_attempt(10).as_millis();
            // Capped at 1000ms, with jitter: 750-1250ms
            assert!(d <= 1500, "d={}", d);
        }
    }

    #[test]
    fn retryable_status_codes() {
        assert!(is_retryable_status(429));
        assert!(is_retryable_status(500));
        assert!(is_retryable_status(502));
        assert!(is_retryable_status(503));
        assert!(is_retryable_status(504));
        assert!(is_retryable_status(408));

        assert!(!is_retryable_status(200));
        assert!(!is_retryable_status(400));
        assert!(!is_retryable_status(401));
        assert!(!is_retryable_status(403));
        assert!(!is_retryable_status(404));
    }
}
