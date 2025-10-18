# integrations/coinalyze_api.py
import os
import time
import requests

BASE_URL = (os.getenv("COINALYZER_API_BASE") or "https://api.coinalyze.net").rstrip("/")
API_KEY = os.getenv("COINALYZER_API_KEY", "")
TIMEOUT = float(os.getenv("COINALYZER_TIMEOUT", "10"))
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
}
if API_KEY:
    HEADERS["Authorization"] = f"Bearer {API_KEY}"
    HEADERS["X-API-KEY"] = API_KEY

# Normalize timeframes (1min → 1m, 1hour → 1h)
_INTERVAL_MAP = {
    "1min": "1m", "3min": "3m", "5min": "5m", "15min": "15m",
    "30min": "30m", "1hour": "1h", "4hour": "4h", "12hour": "12h", "1day": "1d"
}
def _norm_tf(tf): return _INTERVAL_MAP.get(tf.lower().strip(), tf)

# ---------- HTTP core ----------
def _get(url, params):
    """Try ?symbol= first, then fallback to ?symbols= for backward compat."""
    try:
        r = requests.get(url, params={**params, "symbol": params.get("symbol", params.get("symbols"))},
                         headers=HEADERS, timeout=TIMEOUT)
        if r.status_code >= 400:
            raise requests.HTTPError(f"Bad request ({r.status_code})", response=r)
        return r.json()
    except requests.HTTPError as e:
        # fallback attempt
        status = getattr(e.response, "status_code", None)
        if status and 400 <= status < 500:
            r2 = requests.get(url, params={**params, "symbols": params.get("symbol", params.get("symbols"))},
                              headers=HEADERS, timeout=TIMEOUT)
            r2.raise_for_status()
            return r2.json()
        raise

# ---------- API Endpoints ----------
def get_exchanges(): return _get(f"{BASE_URL}/v1/exchanges", {})

def get_future_markets(): return _get(f"{BASE_URL}/v1/futures-markets", {})

def get_open_interest(symbol): return _get(f"{BASE_URL}/v1/open-interest", {"symbol": symbol})

def get_funding_rate(symbol): return _get(f"{BASE_URL}/v1/funding-rate", {"symbol": symbol})

def get_open_interest_history(symbol, interval, t0, t1):
    return _get(f"{BASE_URL}/v1/open-interest-history",
                {"symbol": symbol, "interval": _norm_tf(interval), "from": int(t0), "to": int(t1)})

def get_funding_rate_history(symbol, interval, t0, t1):
    return _get(f"{BASE_URL}/v1/funding-rate-history",
                {"symbol": symbol, "interval": _norm_tf(interval), "from": int(t0), "to": int(t1)})

def get_predicted_funding_rate_history(symbol, interval, t0, t1):
    return _get(f"{BASE_URL}/v1/predicted-funding-rate-history",
                {"symbol": symbol, "interval": _norm_tf(interval), "from": int(t0), "to": int(t1)})

def get_liquidation_history(symbol, interval, t0, t1):
    return _get(f"{BASE_URL}/v1/liquidation-history",
                {"symbol": symbol, "interval": _norm_tf(interval), "from": int(t0), "to": int(t1)})

def get_long_short_ratio_history(symbol, interval, t0, t1):
    return _get(f"{BASE_URL}/v1/long-short-ratio-history",
                {"symbol": symbol, "interval": _norm_tf(interval), "from": int(t0), "to": int(t1)})

def get_ohlcv_history(symbol, interval, t0, t1):
    return _get(f"{BASE_URL}/v1/ohlcv-history",
                {"symbol": symbol, "interval": _norm_tf(interval), "from": int(t0), "to": int(t1)})
