# coinalyze_api.py — robust client (symbol/symbols fallback + tf normalization + retries)
import os
import time
import typing as T

import requests
from requests import Response, Session

# ---------- Configuration ----------
BASE_URL  = (os.getenv("COINALYZER_API_BASE") or "https://api.coinalyze.net").rstrip("/")
API_KEY   = os.getenv("COINALYZER_API_KEY", "")
TIMEOUT   = float(os.getenv("COINALYZER_TIMEOUT", "10"))
RETRIES   = int(os.getenv("COINALYZER_RETRIES", "2"))

# Some projects use different header names; include both to be safe.
_BASE_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
}
if API_KEY:
    _BASE_HEADERS["X-API-KEY"] = API_KEY
    _BASE_HEADERS["Authorization"] = f"Bearer {API_KEY}"

# 1min/1hour to 1m/1h normalization
_INTERVAL_MAP = {
    "1min":"1m","3min":"3m","5min":"5m","12min":"12m","15min":"15m","30min":"30m","45min":"45m",
    "1hour":"1h","2hour":"2h","4hour":"4h","6hour":"6h","8hour":"8h","12hour":"12h",
    "1day":"1d","1week":"1w"
}
def _norm_tf(tf: str) -> str:
    tf = (tf or "").strip().lower()
    return _INTERVAL_MAP.get(tf, tf)

# ---------- HTTP core with small retry ----------
_session: Session = requests.Session()

def _should_retry(resp: Response, exc: Exception|None) -> bool:
    if exc is not None:
        return True
    if resp is None:
        return True
    if resp.status_code in (429, 500, 502, 503, 504):
        return True
    return False

def _req(method: str, url: str, *, params: dict|None = None, timeout: float|None = None) -> dict:
    tries = max(1, 1 + RETRIES)
    last_exc: Exception|None = None
    for i in range(tries):
        resp = None
        exc = None
        try:
            resp = _session.request(method.upper(), url, params=params, headers=_BASE_HEADERS, timeout=timeout or TIMEOUT)
            if resp.status_code >= 400:
                # raise to go through retry gate or caller handling
                resp.raise_for_status()
            return resp.json()
        except Exception as e:
            exc = e
            if not _should_retry(resp, exc) or i == tries - 1:
                # Final attempt failed: bubble the error up
                raise
            # small backoff
            time.sleep(0.5 * (i + 1))
    # should never reach
    raise RuntimeError("unreachable")

# ---------- convenience builders ----------
def _url(path: str) -> str:
    path = path if path.startswith("/") else f"/{path}"
    return f"{BASE_URL}{path}"

def _hist_with_fallback(endpoint: str, *, symbol: str, interval: str, t0: int, t1: int) -> dict:
    """
    Calls /v1/<endpoint> with ?symbol=... first; if client error, retries with ?symbols=...
    This preserves BTC behavior and fixes ETH/SOL 400s without breaking anything.
    """
    interval = _norm_tf(interval)
    # Try 'symbol=' (correct for single-asset calls)
    p1 = {"symbol": symbol, "interval": interval, "from": int(t0), "to": int(t1)}
    url = _url(f"/v1/{endpoint}")
    try:
        return _req("GET", url, params=p1)
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status is not None and 400 <= status < 500:
            # fallback legacy path 'symbols='
            p2 = {"symbols": symbol, "interval": interval, "from": int(t0), "to": int(t1)}
            return _req("GET", url, params=p2)
        # Otherwise bubble up (5xx etc get retried by _req already)
        raise

def _snap_with_fallback(endpoint: str, *, symbol: str) -> dict:
    """
    Snapshot endpoints (open-interest, funding-rate). Try `symbol=` then fallback to `symbols=`.
    """
    url = _url(f"/v1/{endpoint}")
    try:
        return _req("GET", url, params={"symbol": symbol})
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status is not None and 400 <= status < 500:
            return _req("GET", url, params={"symbols": symbol})
        raise

# ---------- Public API (used by your collector) ----------

def get_exchanges() -> dict:
    return _req("GET", _url("/v1/exchanges"))

def get_future_markets() -> dict:
    # returns the full markets list; filtering is done in loop code
    return _req("GET", _url("/v1/futures-markets"))

# --- Snapshots ---
def get_open_interest(symbol: str) -> dict:
    return _snap_with_fallback("open-interest", symbol=symbol)

def get_funding_rate(symbol: str) -> dict:
    return _snap_with_fallback("funding-rate", symbol=symbol)

# --- Histories ---
def get_open_interest_history(symbol: str, interval: str, t0: int, t1: int) -> dict:
    return _hist_with_fallback("open-interest-history", symbol=symbol, interval=interval, t0=t0, t1=t1)

def get_funding_rate_history(symbol: str, interval: str, t0: int, t1: int) -> dict:
    return _hist_with_fallback("funding-rate-history", symbol=symbol, interval=interval, t0=t0, t1=t1)

def get_predicted_funding_rate_history(symbol: str, interval: str, t0: int, t1: int) -> dict:
    return _hist_with_fallback("predicted-funding-rate-history", symbol=symbol, interval=interval, t0=t0, t1=t1)

def get_liquidation_history(symbol: str, interval: str, t0: int, t1: int) -> dict:
    return _hist_with_fallback("liquidation-history", symbol=symbol, interval=interval, t0=t0, t1=t1)

def get_long_short_ratio_history(symbol: str, interval: str, t0: int, t1: int) -> dict:
    return _hist_with_fallback("long-short-ratio-history", symbol=symbol, interval=interval, t0=t0, t1=t1)

def get_ohlcv_history(symbol: str, interval: str, t0: int, t1: int) -> dict:
    return _hist_with_fallback("ohlcv-history", symbol=symbol, interval=interval, t0=t0, t1=t1)
