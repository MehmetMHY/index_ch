"""Tests for config.py: estimate_cost, TIME_RANGES, import guards."""

import pytest

from config import estimate_cost, PRICING, TIME_RANGES, SUMMARY_MODEL, EMBEDDING_MODEL


class TestEstimateCost:
    def test_zero_tokens(self):
        assert estimate_cost(SUMMARY_MODEL, 0, 0) == 0.0

    def test_input_only(self):
        # embeddings have no output price
        cost = estimate_cost(EMBEDDING_MODEL, input_tokens=1_000_000)
        assert cost == pytest.approx(0.02)

    def test_output_only(self):
        cost = estimate_cost(SUMMARY_MODEL, output_tokens=1_000_000)
        assert cost == pytest.approx(1.25)

    def test_both_tokens(self):
        cost = estimate_cost(SUMMARY_MODEL, 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.20 + 1.25)

    def test_small_token_count(self):
        cost = estimate_cost(SUMMARY_MODEL, 100, 50)
        expected = 100 / 1e6 * 0.20 + 50 / 1e6 * 1.25
        assert cost == pytest.approx(expected)

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError):
            estimate_cost("nonexistent-model")

    def test_all_models_in_pricing(self):
        for model in (
            SUMMARY_MODEL,
            EMBEDDING_MODEL,
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ):
            assert model in PRICING

    def test_embedding_output_price_is_zero(self):
        assert PRICING[EMBEDDING_MODEL][1] == 0.0


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
