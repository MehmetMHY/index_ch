"""Tests for pricing.py: cache load/save, refresh triggers, fallback to stale,
None on unknown, and the _fetch_catalog retry loop. No external calls: every
network touchpoint is mocked via monkeypatch on _fetch_catalog or urlopen.
"""

import json
import os
import time
import urllib.error

import pytest

import pricing
from config import (
    SUMMARY_MODEL,
    EMBEDDING_MODEL,
    RERANK_MODEL,
    QUERY_EXPANSION_MODEL,
    PRICING_CACHE_PATH,
)

# --- helpers / fixtures -----------------------------------------------------


def make_catalog(openai_in=0.20, openai_out=1.25, groq_in=0.15, groq_out=0.60):
    """A minimal catalog with the four project models and their providers."""
    return {
        "openai": {
            "models": {
                SUMMARY_MODEL: {"cost": {"input": openai_in, "output": openai_out}},
                EMBEDDING_MODEL: {"cost": {"input": 0.02, "output": 0}},
            }
        },
        "groq": {
            "models": {
                RERANK_MODEL: {"cost": {"input": groq_in, "output": groq_out}},
                QUERY_EXPANSION_MODEL: {"cost": {"input": 0.075, "output": 0.30}},
            }
        },
    }


@pytest.fixture(autouse=True)
def reset_state(monkeypatch, tmp_path):
    """Each test gets a clean in-memory state and an isolated cache file path
    so no test reads or writes the real ~/.ch/index/pricing_cache.json."""
    pricing._state["cache"] = None
    pricing._state["loaded"] = False
    monkeypatch.setattr(
        pricing.config, "PRICING_CACHE_PATH", str(tmp_path / "pricing_cache.json")
    )
    yield
    pricing._state["cache"] = None
    pricing._state["loaded"] = False


# --- estimate_cost / get_price with a warm cache ----------------------------


def _seed_cache(monkeypatch, catalog, age=0):
    """Write a cache file built from `catalog` with fetched_at = now - age,
    and load it into the module state."""
    snap = pricing._snapshot_from_catalog(catalog)
    if age:
        snap["fetched_at"] = int(time.time()) - age
    with open(pricing.config.PRICING_CACHE_PATH, "w") as f:
        json.dump(snap, f)


class TestEstimateCost:
    def test_known_model_returns_float(self, monkeypatch):
        _seed_cache(monkeypatch, make_catalog())
        # force loaded so get_price uses the file we just wrote
        pricing._state["loaded"] = True
        pricing._state["cache"] = pricing._load_cache_file()
        cost = pricing.estimate_cost(SUMMARY_MODEL, 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.20 + 1.25)

    def test_embedding_no_output_tokens(self, monkeypatch):
        _seed_cache(monkeypatch, make_catalog())
        pricing._state["loaded"] = True
        pricing._state["cache"] = pricing._load_cache_file()
        cost = pricing.estimate_cost(EMBEDDING_MODEL, 1_000_000)
        assert cost == pytest.approx(0.02)

    def test_zero_tokens_returns_zero(self, monkeypatch):
        _seed_cache(monkeypatch, make_catalog())
        pricing._state["loaded"] = True
        pricing._state["cache"] = pricing._load_cache_file()
        assert pricing.estimate_cost(SUMMARY_MODEL, 0, 0) == 0.0

    def test_unknown_model_returns_none(self, monkeypatch):
        _seed_cache(monkeypatch, make_catalog())
        pricing._state["loaded"] = True
        pricing._state["cache"] = pricing._load_cache_file()
        # an unknown model would normally trigger a refresh; stub the fetch to
        # return the same catalog (still no such model) so we land on None
        monkeypatch.setattr(pricing, "_fetch_catalog", lambda: make_catalog())
        assert pricing.estimate_cost("nonexistent-model") is None

    def test_unknown_provider_returns_none(self, monkeypatch):
        # model declared in MODEL_PROVIDER but provider key absent from catalog
        _seed_cache(monkeypatch, {"openai": {"models": {}}, "groq": {"models": {}}})
        pricing._state["loaded"] = True
        pricing._state["cache"] = pricing._load_cache_file()
        monkeypatch.setattr(
            pricing,
            "_fetch_catalog",
            lambda: {"openai": {"models": {}}, "groq": {"models": {}}},
        )
        assert pricing.estimate_cost(SUMMARY_MODEL) is None


# --- refresh triggers --------------------------------------------------------


class TestRefreshTriggers:
    def test_fresh_cache_no_refresh(self, monkeypatch):
        _seed_cache(monkeypatch, make_catalog(), age=0)
        called = {"n": 0}

        def boom():
            called["n"] += 1
            return None

        monkeypatch.setattr(pricing, "_fetch_catalog", boom)
        pricing._ensure_loaded()
        # cache is loaded from file via _ensure_loaded in get_price
        assert pricing.get_price(SUMMARY_MODEL) == (0.20, 1.25)
        assert called["n"] == 0  # no refresh needed

    def test_stale_cache_triggers_refresh(self, monkeypatch):
        # age beyond PRICING_TTL
        _seed_cache(
            monkeypatch,
            make_catalog(openai_in=0.99, openai_out=9.99),
            age=pricing.config.PRICING_TTL + 3600,
        )
        monkeypatch.setattr(pricing, "_fetch_catalog", lambda: make_catalog())
        # stale cache had 0.99; after refresh we should see 0.20
        assert pricing.get_price(SUMMARY_MODEL) == (0.20, 1.25)

    def test_missing_model_triggers_refresh(self, monkeypatch):
        # cache has only the openai models, not the groq ones
        partial = {
            "openai": {
                "models": {
                    SUMMARY_MODEL: {"cost": {"input": 0.20, "output": 1.25}},
                    EMBEDDING_MODEL: {"cost": {"input": 0.02, "output": 0}},
                }
            },
            "groq": {"models": {}},
        }
        _seed_cache(monkeypatch, partial, age=0)  # fresh, but model missing
        monkeypatch.setattr(pricing, "_fetch_catalog", lambda: make_catalog())
        assert pricing.get_price(RERANK_MODEL) == (0.15, 0.60)

    def test_missing_cache_file_triggers_refresh(self, monkeypatch):
        # no cache file written at all
        monkeypatch.setattr(pricing, "_fetch_catalog", lambda: make_catalog())
        assert pricing.get_price(SUMMARY_MODEL) == (0.20, 1.25)
        # and the cache file should now exist on disk
        assert os.path.exists(pricing.config.PRICING_CACHE_PATH)

    def test_malformed_cache_file_triggers_refresh(self, monkeypatch, tmp_path):
        with open(pricing.config.PRICING_CACHE_PATH, "w") as f:
            f.write("{not valid json")
        monkeypatch.setattr(pricing, "_fetch_catalog", lambda: make_catalog())
        assert pricing.get_price(SUMMARY_MODEL) == (0.20, 1.25)


# --- fallback on refresh failure --------------------------------------------


class TestRefreshFailure:
    def test_falls_back_to_stale_cache(self, monkeypatch):
        _seed_cache(
            monkeypatch,
            make_catalog(openai_in=0.42, openai_out=4.20),
            age=pricing.config.PRICING_TTL + 3600,
        )
        monkeypatch.setattr(pricing, "_fetch_catalog", lambda: None)
        # stale but usable; refresh fails; we keep the stale price
        assert pricing.get_price(SUMMARY_MODEL) == (0.42, 4.20)

    def test_no_cache_no_refresh_returns_none(self, monkeypatch):
        # _fetch_catalog is mocked to fail directly, so no warning is printed
        # here (the warning is emitted inside the real _fetch_catalog, tested
        # separately in TestFetchRetries); the contract we check here is the
        # return value: None, with no crash.
        monkeypatch.setattr(pricing, "_fetch_catalog", lambda: None)
        assert pricing.get_price(SUMMARY_MODEL) is None

    def test_model_not_in_fresh_catalog_returns_none(self, monkeypatch):
        _seed_cache(monkeypatch, make_catalog(), age=pricing.config.PRICING_TTL + 3600)
        # fresh catalog lacks our model entirely
        empty = {"openai": {"models": {}}, "groq": {"models": {}}}
        monkeypatch.setattr(pricing, "_fetch_catalog", lambda: empty)
        assert pricing.get_price(SUMMARY_MODEL) is None


class TestWarm:
    def test_warm_fresh_cache_no_fetch(self, monkeypatch):
        _seed_cache(monkeypatch, make_catalog(), age=0)
        monkeypatch.setattr(
            pricing,
            "_fetch_catalog",
            lambda: (_ for _ in ()).throw(AssertionError("should not fetch")),
        )
        pricing.warm()
        # cache should be loaded from file, no refresh

    def test_warm_stale_triggers_refresh(self, monkeypatch):
        _seed_cache(
            monkeypatch,
            make_catalog(openai_in=0.99, openai_out=9.99),
            age=pricing.config.PRICING_TTL + 3600,
        )
        monkeypatch.setattr(pricing, "_fetch_catalog", lambda: make_catalog())
        pricing.warm()
        # after warm, get_price should see the refreshed price without fetching
        monkeypatch.setattr(
            pricing,
            "_fetch_catalog",
            lambda: (_ for _ in ()).throw(AssertionError("should not fetch twice")),
        )
        assert pricing.get_price(SUMMARY_MODEL) == (0.20, 1.25)

    def test_warm_missing_model_triggers_refresh(self, monkeypatch):
        partial = {
            "openai": {
                "models": {
                    SUMMARY_MODEL: {"cost": {"input": 0.20, "output": 1.25}},
                    EMBEDDING_MODEL: {"cost": {"input": 0.02, "output": 0}},
                }
            },
            "groq": {"models": {}},
        }
        _seed_cache(monkeypatch, partial, age=0)  # fresh, but groq models missing
        monkeypatch.setattr(pricing, "_fetch_catalog", lambda: make_catalog())
        pricing.warm()
        assert pricing.get_price(RERANK_MODEL) == (0.15, 0.60)

    def test_warm_no_cache_triggers_refresh(self, monkeypatch):
        monkeypatch.setattr(pricing, "_fetch_catalog", lambda: make_catalog())
        pricing.warm()
        assert os.path.exists(pricing.config.PRICING_CACHE_PATH)

    def test_warm_refresh_failure_does_not_raise(self, monkeypatch, capsys):
        _seed_cache(
            monkeypatch,
            make_catalog(openai_in=0.42, openai_out=4.20),
            age=pricing.config.PRICING_TTL + 3600,
        )
        monkeypatch.setattr(pricing, "_fetch_catalog", lambda: None)
        pricing.warm()  # must not raise
        # stale cache still usable
        assert pricing.get_price(SUMMARY_MODEL) == (0.42, 4.20)

    def test_warm_no_cache_no_fetch_does_not_raise(self, monkeypatch, capsys):
        monkeypatch.setattr(pricing, "_fetch_catalog", lambda: None)
        pricing.warm()  # must not raise
        assert pricing.get_price(SUMMARY_MODEL) is None


# --- malformed entries ------------------------------------------------------


class TestMalformedEntries:
    def test_non_numeric_input_returns_none(self, monkeypatch):
        bad = {
            "openai": {
                "models": {SUMMARY_MODEL: {"cost": {"input": "free", "output": 1.0}}}
            },
            "groq": {"models": {}},
        }
        _seed_cache(monkeypatch, bad, age=0)
        monkeypatch.setattr(pricing, "_fetch_catalog", lambda: bad)
        assert pricing.get_price(SUMMARY_MODEL) is None

    def test_missing_cost_key_returns_none(self, monkeypatch):
        bad = {
            "openai": {"models": {SUMMARY_MODEL: {}}},
            "groq": {"models": {}},
        }
        _seed_cache(monkeypatch, bad, age=0)
        monkeypatch.setattr(pricing, "_fetch_catalog", lambda: bad)
        assert pricing.get_price(SUMMARY_MODEL) is None


# --- cache file format / persistence ----------------------------------------


class TestCacheFile:
    def test_snapshot_has_epoch_and_utc(self):
        snap = pricing._snapshot_from_catalog(make_catalog())
        assert "fetched_at" in snap
        assert isinstance(snap["fetched_at"], int)
        assert "fetched_at_utc" in snap
        # ISO 8601 with +00:00 offset
        assert snap["fetched_at_utc"].endswith("+00:00")

    def test_save_then_load_roundtrip(self, monkeypatch):
        snap = pricing._snapshot_from_catalog(make_catalog())
        pricing._save_cache_file(snap)
        loaded = pricing._load_cache_file()
        assert loaded["fetched_at"] == snap["fetched_at"]
        assert loaded["providers"]["openai"][SUMMARY_MODEL] == {
            "input": 0.20,
            "output": 1.25,
        }
        assert loaded["providers"]["groq"][RERANK_MODEL] == {
            "input": 0.15,
            "output": 0.60,
        }

    def test_save_is_atomic_no_tmp_left(self, monkeypatch):
        snap = pricing._snapshot_from_catalog(make_catalog())
        pricing._save_cache_file(snap)
        tmp = pricing.config.PRICING_CACHE_PATH + ".tmp"
        assert not os.path.exists(tmp)
        assert os.path.exists(pricing.config.PRICING_CACHE_PATH)

    def test_load_missing_file_returns_none(self):
        assert pricing._load_cache_file() is None

    def test_load_malformed_file_returns_none(self):
        with open(pricing.config.PRICING_CACHE_PATH, "w") as f:
            f.write("garbage")
        assert pricing._load_cache_file() is None


# --- _fetch_catalog retry loop (mocked urlopen) -----------------------------


class TestFetchRetries:
    def _patch_urlopen(self, monkeypatch, side_effects):
        """Make urlopen return a fake context manager whose read() yields the
        next bytes from `side_effects`, or raise the next exception."""
        calls = {"n": 0}

        class FakeResp:
            def __init__(self, payload):
                self._payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                if isinstance(self._payload, Exception):
                    raise self._payload
                return self._payload.encode()

        def fake_urlopen(req, timeout=None):
            i = calls["n"]
            calls["n"] += 1
            se = side_effects[i]
            if isinstance(se, Exception):
                raise se
            return FakeResp(se)

        monkeypatch.setattr(pricing.urllib.request, "urlopen", fake_urlopen)
        # speed up the test: no real sleeping
        monkeypatch.setattr(pricing.time, "sleep", lambda s: None)
        return calls

    def test_succeeds_first_try(self, monkeypatch):
        self._patch_urlopen(monkeypatch, [json.dumps(make_catalog())])
        assert pricing._fetch_catalog() is not None

    def test_retries_then_succeeds(self, monkeypatch):
        calls = self._patch_urlopen(
            monkeypatch,
            [
                urllib.error.URLError("transient"),
                json.dumps(make_catalog()),
            ],
        )
        assert pricing._fetch_catalog() is not None
        assert calls["n"] == 2

    def test_all_attempts_fail_returns_none(self, monkeypatch, capsys):
        calls = self._patch_urlopen(
            monkeypatch,
            [
                urllib.error.URLError("err1"),
                urllib.error.URLError("err2"),
                urllib.error.URLError("err3"),
            ],
        )
        assert pricing._fetch_catalog() is None
        assert calls["n"] == 3  # initial + 2 retries
        out = capsys.readouterr().out
        assert "could not refresh pricing catalog" in out

    def test_json_decode_error_retries(self, monkeypatch):
        calls = self._patch_urlopen(
            monkeypatch,
            ["not json", "still not json", json.dumps(make_catalog())],
        )
        assert pricing._fetch_catalog() is not None
        assert calls["n"] == 3
