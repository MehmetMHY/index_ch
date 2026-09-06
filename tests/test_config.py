"""Tests for config.py: TIME_RANGES, MODEL_PROVIDER, PRICING_TTL, import guards.

estimate_cost tests live in test_pricing.py now that pricing is fetched from
the models.dev catalog instead of hardcoded in config.PRICING.
"""

from config import (
    TIME_RANGES,
    SUMMARY_MODEL,
    EMBEDDING_MODEL,
    RERANK_MODEL,
    QUERY_EXPANSION_MODEL,
    MODEL_PROVIDER,
    PRICING_TTL,
)


class TestModelProvider:
    """Every declared model maps to a models.dev provider key."""

    def test_all_models_mapped(self):
        models = (SUMMARY_MODEL, EMBEDDING_MODEL, RERANK_MODEL, QUERY_EXPANSION_MODEL)
        for m in models:
            assert m in MODEL_PROVIDER

    def test_openai_models_map_to_openai(self):
        assert MODEL_PROVIDER[SUMMARY_MODEL] == "openai"
        assert MODEL_PROVIDER[EMBEDDING_MODEL] == "openai"

    def test_retrieval_models_map_to_openai(self):
        assert MODEL_PROVIDER[RERANK_MODEL] == "openai"
        assert MODEL_PROVIDER[QUERY_EXPANSION_MODEL] == "openai"


class TestPricingTtl:
    def test_ttl_is_at_least_a_few_days(self):
        assert PRICING_TTL >= 3 * 86_400

    def test_ttl_is_at_most_a_week(self):
        assert PRICING_TTL <= 7 * 86_400


class TestTimeRanges:
    def test_1d(self):
        assert TIME_RANGES["1d"] == 86400

    def test_3d(self):
        assert TIME_RANGES["3d"] == 3 * 86400

    def test_1w(self):
        assert TIME_RANGES["1w"] == 7 * 86400

    def test_1m_approximate(self):
        assert TIME_RANGES["1m"] == 30 * 86400

    def test_1y_approximate(self):
        assert TIME_RANGES["1y"] == 365 * 86400

    def test_keys(self):
        assert set(TIME_RANGES.keys()) == {"1d", "3d", "1w", "1m", "1y"}
