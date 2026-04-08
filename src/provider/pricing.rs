//! Model pricing and cost estimation.
//!
//! Maps (model name, token usage) → estimated USD cost.
//! Default pricing table ships for known Anthropic and OpenAI models.
//! Config can override per-model for proxies (Requesty, OpenRouter) or local models.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::TokenUsage;

/// Per-model pricing rates (USD per million tokens).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPricing {
    pub input_per_mtok: f64,
    pub output_per_mtok: f64,
    #[serde(default)]
    pub cache_read_per_mtok: f64,
    #[serde(default)]
    pub cache_write_per_mtok: f64,
}

/// Cost breakdown from a single API call.
#[derive(Debug, Clone, Default)]
pub struct CostEstimate {
    pub total_usd: f64,
    pub input_usd: f64,
    pub output_usd: f64,
    pub cache_read_usd: f64,
    pub cache_write_usd: f64,
}

impl std::fmt::Display for CostEstimate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "${:.6}", self.total_usd)
    }
}

/// Fallback pricing for unknown models (mid-tier: roughly Sonnet-class).
const FALLBACK_PRICING: ModelPricing = ModelPricing {
    input_per_mtok: 3.0,
    output_per_mtok: 15.0,
    cache_read_per_mtok: 0.3,
    cache_write_per_mtok: 3.75,
};

/// Zero pricing for local/free models.
pub const FREE_PRICING: ModelPricing = ModelPricing {
    input_per_mtok: 0.0,
    output_per_mtok: 0.0,
    cache_read_per_mtok: 0.0,
    cache_write_per_mtok: 0.0,
};

/// Build the default pricing table.
fn default_pricing_table() -> HashMap<&'static str, ModelPricing> {
    let mut table = HashMap::new();

    // Anthropic
    table.insert(
        "claude-haiku-4-5",
        ModelPricing {
            input_per_mtok: 1.0,
            output_per_mtok: 5.0,
            cache_read_per_mtok: 0.1,
            cache_write_per_mtok: 1.25,
        },
    );
    table.insert(
        "claude-sonnet-4-6",
        ModelPricing {
            input_per_mtok: 3.0,
            output_per_mtok: 15.0,
            cache_read_per_mtok: 0.3,
            cache_write_per_mtok: 3.75,
        },
    );
    table.insert(
        "claude-opus-4-6",
        ModelPricing {
            input_per_mtok: 5.0,
            output_per_mtok: 25.0,
            cache_read_per_mtok: 0.5,
            cache_write_per_mtok: 6.25,
        },
    );

    // OpenAI
    table.insert(
        "gpt-4o",
        ModelPricing {
            input_per_mtok: 2.5,
            output_per_mtok: 10.0,
            cache_read_per_mtok: 1.25,
            cache_write_per_mtok: 0.0,
        },
    );
    table.insert(
        "gpt-4o-mini",
        ModelPricing {
            input_per_mtok: 0.15,
            output_per_mtok: 0.6,
            cache_read_per_mtok: 0.075,
            cache_write_per_mtok: 0.0,
        },
    );
    table.insert(
        "o1",
        ModelPricing {
            input_per_mtok: 15.0,
            output_per_mtok: 60.0,
            cache_read_per_mtok: 7.5,
            cache_write_per_mtok: 0.0,
        },
    );
    table.insert(
        "o3-mini",
        ModelPricing {
            input_per_mtok: 1.1,
            output_per_mtok: 4.4,
            cache_read_per_mtok: 0.55,
            cache_write_per_mtok: 0.0,
        },
    );

    table
}

/// Normalize a model name by stripping version/date suffixes.
///
/// `claude-sonnet-4-6-20250514` → `claude-sonnet-4-6`
/// `gpt-4o-2024-08-06` → `gpt-4o`
/// `claude-haiku-4-5-20251001` → `claude-haiku-4-5`
pub fn normalize_model_name(model: &str) -> String {
    // Strip trailing date suffixes like -20250514 or -2024-08-06
    let re = regex::Regex::new(r"-\d{4,}(-\d{2}(-\d{2})?)?$").unwrap();
    re.replace(model, "").to_string()
}

fn estimate_cost_with_pricing(usage: &TokenUsage, pricing: &ModelPricing) -> CostEstimate {
    let input_usd = usage.input_tokens as f64 * pricing.input_per_mtok / 1_000_000.0;
    let output_usd = usage.output_tokens as f64 * pricing.output_per_mtok / 1_000_000.0;
    let cache_read_usd = usage.cache_read_tokens as f64 * pricing.cache_read_per_mtok / 1_000_000.0;
    let cache_write_usd =
        usage.cache_write_tokens as f64 * pricing.cache_write_per_mtok / 1_000_000.0;

    CostEstimate {
        total_usd: input_usd + output_usd + cache_read_usd + cache_write_usd,
        input_usd,
        output_usd,
        cache_read_usd,
        cache_write_usd,
    }
}

/// Look up pricing for a model, checking overrides first, then defaults.
pub fn resolve_pricing(model: &str, override_pricing: Option<&ModelPricing>) -> ModelPricing {
    if let Some(p) = override_pricing {
        return p.clone();
    }
    let normalized = normalize_model_name(model);
    let table = default_pricing_table();
    table
        .get(normalized.as_str())
        .cloned()
        .unwrap_or_else(|| FALLBACK_PRICING.clone())
}

/// Estimate cost using resolved pricing.
pub fn estimate_cost_for_model(
    model: &str,
    usage: &TokenUsage,
    override_pricing: Option<&ModelPricing>,
) -> CostEstimate {
    let pricing = resolve_pricing(model, override_pricing);
    estimate_cost_with_pricing(usage, &pricing)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_model_pricing() {
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 500_000,
            ..Default::default()
        };
        let cost = estimate_cost_for_model("claude-sonnet-4-6", &usage, None);
        // 1M input * $3/MTok + 500K output * $15/MTok = $3 + $7.5 = $10.5
        assert!((cost.total_usd - 10.5).abs() < 0.001);
        assert!((cost.input_usd - 3.0).abs() < 0.001);
        assert!((cost.output_usd - 7.5).abs() < 0.001);
    }

    #[test]
    fn unknown_model_uses_fallback() {
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..Default::default()
        };
        let cost = estimate_cost_for_model("some-unknown-model", &usage, None);
        // Fallback: $3 input + $15 output = $18
        assert!((cost.total_usd - 18.0).abs() < 0.001);
    }

    #[test]
    fn config_override_takes_precedence() {
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..Default::default()
        };
        let custom = ModelPricing {
            input_per_mtok: 1.0,
            output_per_mtok: 2.0,
            cache_read_per_mtok: 0.0,
            cache_write_per_mtok: 0.0,
        };
        let cost = estimate_cost_for_model("claude-sonnet-4-6", &usage, Some(&custom));
        // Override: $1 + $2 = $3 (not the default Sonnet pricing)
        assert!((cost.total_usd - 3.0).abs() < 0.001);
    }

    #[test]
    fn cache_tokens_priced_correctly() {
        let usage = TokenUsage {
            input_tokens: 100_000,
            output_tokens: 50_000,
            cache_read_tokens: 900_000,
            cache_write_tokens: 100_000,
            reasoning_tokens: 0,
        };
        let cost = estimate_cost_for_model("claude-sonnet-4-6", &usage, None);
        // 100K * $3 + 50K * $15 + 900K * $0.3 + 100K * $3.75 = $0.3 + $0.75 + $0.27 + $0.375 = $1.695
        assert!(
            (cost.total_usd - 1.695).abs() < 0.001,
            "got {}",
            cost.total_usd
        );
    }

    #[test]
    fn model_name_normalization() {
        assert_eq!(
            normalize_model_name("claude-sonnet-4-6-20250514"),
            "claude-sonnet-4-6"
        );
        assert_eq!(normalize_model_name("gpt-4o-2024-08-06"), "gpt-4o");
        assert_eq!(
            normalize_model_name("claude-haiku-4-5-20251001"),
            "claude-haiku-4-5"
        );
        assert_eq!(normalize_model_name("claude-opus-4-6"), "claude-opus-4-6");
        assert_eq!(normalize_model_name("o3-mini"), "o3-mini");
    }

    #[test]
    fn local_model_free() {
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..Default::default()
        };
        let cost = estimate_cost_for_model("llama3", &usage, Some(&FREE_PRICING));
        assert!((cost.total_usd).abs() < 0.0001);
    }

    #[test]
    fn zero_usage_zero_cost() {
        let cost = estimate_cost_for_model("claude-sonnet-4-6", &TokenUsage::default(), None);
        assert!((cost.total_usd).abs() < 0.0001);
    }
}
