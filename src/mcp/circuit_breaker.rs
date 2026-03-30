//! Circuit breaker pattern for MCP server resilience.
//!
//! Prevents repeated calls to failing servers by tracking failures
//! and temporarily rejecting requests ("tripping" the circuit).

use std::time::{Duration, Instant};

/// Circuit breaker states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation -- calls go through
    Closed,
    /// Rejecting calls -- too many failures
    Open,
    /// Testing recovery -- allowing one call through
    HalfOpen,
}

/// Configuration for circuit breaker behavior.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures before opening circuit
    pub failure_threshold: u32,
    /// How long to wait before trying again (half-open state)
    pub recovery_timeout: Duration,
    /// Number of successes in half-open state before closing
    pub success_threshold: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 3,
            recovery_timeout: Duration::from_secs(30),
            success_threshold: 1,
        }
    }
}

/// Circuit breaker for a single MCP server.
#[derive(Debug)]
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: CircuitState,
    consecutive_failures: u32,
    consecutive_successes: u32,
    last_failure_time: Option<Instant>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker with the given configuration.
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: CircuitState::Closed,
            consecutive_failures: 0,
            consecutive_successes: 0,
            last_failure_time: None,
        }
    }

    /// Check if a call should be allowed through.
    ///
    /// Returns `true` if the call should proceed, `false` if rejected.
    pub fn should_allow(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if recovery timeout has elapsed
                if let Some(last_failure) = self.last_failure_time {
                    if last_failure.elapsed() >= self.config.recovery_timeout {
                        self.state = CircuitState::HalfOpen;
                        self.consecutive_successes = 0;
                        tracing::info!("circuit breaker transitioning to half-open");
                        return true;
                    }
                }
                false
            }
            CircuitState::HalfOpen => true,
        }
    }

    /// Record a successful call.
    pub fn record_success(&mut self) {
        self.consecutive_failures = 0;

        match self.state {
            CircuitState::HalfOpen => {
                self.consecutive_successes += 1;
                if self.consecutive_successes >= self.config.success_threshold {
                    self.state = CircuitState::Closed;
                    tracing::info!("circuit breaker closed after successful recovery");
                }
            }
            CircuitState::Closed => {
                // Already closed, nothing to do
            }
            CircuitState::Open => {
                // Shouldn't happen, but handle gracefully
                self.state = CircuitState::Closed;
            }
        }
    }

    /// Record a failed call.
    pub fn record_failure(&mut self) {
        self.consecutive_failures += 1;
        self.consecutive_successes = 0;
        self.last_failure_time = Some(Instant::now());

        match self.state {
            CircuitState::Closed => {
                if self.consecutive_failures >= self.config.failure_threshold {
                    self.state = CircuitState::Open;
                    tracing::warn!(
                        failures = self.consecutive_failures,
                        "circuit breaker opened after consecutive failures"
                    );
                }
            }
            CircuitState::HalfOpen => {
                // Failed during recovery test, go back to open
                self.state = CircuitState::Open;
                tracing::warn!("circuit breaker re-opened after half-open failure");
            }
            CircuitState::Open => {
                // Already open, update timestamp
            }
        }
    }

    /// Get the current state.
    pub fn state(&self) -> CircuitState {
        self.state
    }

    /// Get the number of consecutive failures.
    pub fn consecutive_failures(&self) -> u32 {
        self.consecutive_failures
    }

    /// Manually reset the circuit breaker to closed state.
    pub fn reset(&mut self) {
        self.state = CircuitState::Closed;
        self.consecutive_failures = 0;
        self.consecutive_successes = 0;
        self.last_failure_time = None;
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new(CircuitBreakerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_closed() {
        let cb = CircuitBreaker::default();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.consecutive_failures() == 0);
    }

    #[test]
    fn allows_calls_when_closed() {
        let mut cb = CircuitBreaker::default();
        assert!(cb.should_allow());
    }

    #[test]
    fn opens_after_failure_threshold() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let mut cb = CircuitBreaker::new(config);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn rejects_calls_when_open() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            recovery_timeout: Duration::from_secs(60),
            ..Default::default()
        };
        let mut cb = CircuitBreaker::new(config);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.should_allow());
    }

    #[test]
    fn success_resets_failure_count() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let mut cb = CircuitBreaker::new(config);

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.consecutive_failures(), 2);

        cb.record_success();
        assert_eq!(cb.consecutive_failures(), 0);
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn transitions_to_half_open_after_timeout() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            recovery_timeout: Duration::from_millis(10),
            ..Default::default()
        };
        let mut cb = CircuitBreaker::new(config);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        // Wait for recovery timeout
        std::thread::sleep(Duration::from_millis(15));

        assert!(cb.should_allow());
        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn closes_after_success_in_half_open() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            recovery_timeout: Duration::from_millis(1),
            success_threshold: 1,
        };
        let mut cb = CircuitBreaker::new(config);

        cb.record_failure();
        std::thread::sleep(Duration::from_millis(5));
        cb.should_allow(); // Transition to half-open

        assert_eq!(cb.state(), CircuitState::HalfOpen);

        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn reopens_after_failure_in_half_open() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            recovery_timeout: Duration::from_millis(1),
            success_threshold: 1,
        };
        let mut cb = CircuitBreaker::new(config);

        cb.record_failure();
        std::thread::sleep(Duration::from_millis(5));
        cb.should_allow(); // Transition to half-open

        assert_eq!(cb.state(), CircuitState::HalfOpen);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn reset_returns_to_closed() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            ..Default::default()
        };
        let mut cb = CircuitBreaker::new(config);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        cb.reset();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.consecutive_failures(), 0);
    }
}
