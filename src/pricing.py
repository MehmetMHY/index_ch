"""Price lookups for the cost estimates the scripts print.

Catalog source: https://models.dev/api.json (provider -> models -> {cost: ...}).
The catalog is cached locally at config.PRICING_CACHE_PATH as a flat
provider -> model -> {input, output, cache_read} snapshot with a fetched_at
epoch and UTC timestamp, so repeated lookups never hit the network.

Refresh triggers (any one): cache file missing, cache older than
config.PRICING_TTL, the requested model missing from the cache, or its entry
malformed. On refresh failure the stale cache is still used if it has the
model; only when there is no usable price do we give up, in which case
estimate_cost returns None and callers print "?" rather than a misleading
number. No hardcoded fallback: the catalog is the single source of truth.

The refresh never raises and never blocks a run on its own account: 2 retries
with growing delays, a 20s per-request timeout, and a one-line warning on
final failure. estimate_cost always returns either a float or None.
"""

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import config

CATALOG_URL = "https://models.dev/api.json"
TIMEOUT = 20.0  # seconds per HTTP request
RETRIES = 2  # extra attempts after the first
RETRY_DELAYS = (2.0, 5.0)  # seconds slept before each retry

# in-memory cache state, loaded lazily from the cache file on first lookup
_state = {"cache": None, "loaded": False}


def _utc_iso(epoch):
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _fetch_catalog():
    """Fetch and parse the catalog, retrying on transient errors.

    Returns the parsed dict on success, or None on final failure (with a
    one-line warning printed in the project style).
    """
    last_err = None
    for attempt in range(RETRIES + 1):
        try:
            req = urllib.request.Request(
                CATALOG_URL, headers={"User-Agent": "index_ch"}
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            last_err = exc
            if attempt < RETRIES:
                time.sleep(RETRY_DELAYS[attempt])
    print(f"(could not refresh pricing catalog: {last_err})")
    return None


def _load_cache_file():
    """Read and parse the cache file; return None if missing or malformed."""
    try:
        with open(config.PRICING_CACHE_PATH) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _save_cache_file(snapshot):
    """Write the cache atomically (write .tmp, rename) so a crash mid-write
    cannot leave a half-written file. Best-effort: a write failure prints a
    warning but does not raise, since the in-memory snapshot still works."""
    tmp = config.PRICING_CACHE_PATH + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(snapshot, f, indent=4)
        os.replace(tmp, config.PRICING_CACHE_PATH)
    except OSError as exc:
        print(f"(could not write pricing cache: {exc})")


def _snapshot_from_catalog(catalog):
    """Flatten the raw catalog into just provider -> model -> cost, plus the
    fetched_at timestamps. Keeps the cache file small and lookups cheap."""
    fetched_at = int(time.time())
    providers = {}
    for prov, pdata in catalog.items():
        models = {}
        for mid, m in pdata.get("models", {}).items():
            cost = m.get("cost") or {}
            models[mid] = {
                "input": cost.get("input"),
                "output": cost.get("output"),
            }
        providers[prov] = models
    return {
        "fetched_at": fetched_at,
        "fetched_at_utc": _utc_iso(fetched_at),
        "providers": providers,
    }


def _is_stale(cache):
    if not cache or "fetched_at" not in cache:
        return True
    return time.time() - cache["fetched_at"] > config.PRICING_TTL


def _extract(cache, model):
    """Return (input, output) tuple from the cache, or None if the model is
    absent, its provider is unknown, or the price entry is malformed."""
    if not cache:
        return None
    prov = config.MODEL_PROVIDER.get(model)
    if not prov:
        return None
    entry = cache.get("providers", {}).get(prov, {}).get(model)
    if not entry:
        return None
    try:
        return float(entry.get("input")), float(entry.get("output"))
    except (TypeError, ValueError):
        return None


def _refresh():
    """Fetch the catalog, persist it, and return the new snapshot. Returns
    None if the fetch failed (the caller falls back to the stale cache)."""
    catalog = _fetch_catalog()
    if catalog is None:
        return None
    snapshot = _snapshot_from_catalog(catalog)
    _save_cache_file(snapshot)
    return snapshot


def _ensure_loaded():
    if not _state["loaded"]:
        _state["cache"] = _load_cache_file()
        _state["loaded"] = True


def warm():
    """Ensure the pricing cache is fresh for every declared model, fetching the
    catalog now if needed. Call this at startup of any script that prints cost
    estimates, so the refresh is predictable and not coupled to whether the
    run actually did work (process.py with nothing to process still warms the
    cache). Best-effort: failures print a warning and leave the stale cache
    in place. Never raises.
    """
    _ensure_loaded()
    cache = _state["cache"]
    needs = _is_stale(cache) or any(
        _extract(cache, m) is None for m in config.MODEL_PROVIDER
    )
    if not needs:
        return
    fresh = _refresh()
    if fresh is not None:
        _state["cache"] = fresh


def get_price(model):
    """Return (input, output) USD-per-1M-tokens tuple, or None if unknown.

    Refreshes the cache when it is missing, stale, or lacks a usable entry for
    the model. On refresh failure, falls back to the stale cache. Always
    returns either a tuple or None; never raises. Call warm() at startup to
    make the refresh eager and predictable.
    """
    # a model not declared in MODEL_PROVIDER has no provider to look up, so no
    # refresh can help; skip the network round trip entirely.
    if model not in config.MODEL_PROVIDER:
        return None
    _ensure_loaded()
    cache = _state["cache"]
    price = _extract(cache, model)
    if price is not None and not _is_stale(cache):
        return price
    # missing, broken, or stale -> try a refresh
    fresh = _refresh()
    if fresh is not None:
        _state["cache"] = fresh
        return _extract(fresh, model)
    # refresh failed: fall back to whatever stale cache we have
    return _extract(cache, model)


def estimate_cost(model, input_tokens=0, output_tokens=0):
    """Return estimated USD cost, or None if the model's price is unknown.

    Callers should format None as "?" (or similar) rather than printing a
    misleading number. Never raises.
    """
    price = get_price(model)
    if price is None:
        return None
    price_in, price_out = price
    return input_tokens / 1e6 * price_in + output_tokens / 1e6 * price_out
