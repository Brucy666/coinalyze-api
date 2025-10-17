import os, json, time, glob
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Query, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np

APP_NAME = "Coinalyze API"
VERSION  = "1.1.1"

# Local dir inside API container (no Railway volume required)
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Token used by collector to push snapshots
INGEST_TOKEN = os.getenv("INGEST_TOKEN", "changeme")
RING_MAX = int(os.getenv("RING_MAX", "500"))
_ring: List[Dict[str, Any]] = []

app = FastAPI(title=APP_NAME, version=VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------- ring / files ----------------------------
def _list_snapshot_files():
    return sorted(glob.glob(str(DATA_DIR / "*.json")), key=os.path.getmtime, reverse=True)

def _push_ring(payload: Dict[str, Any]):
    _ring.insert(0, payload)
    if len(_ring) > RING_MAX:
        _ring.pop()

def _latest_from_ring():
    return _ring[0] if _ring else None

# ---------------------------- helpers ----------------------------
def _to_float(x) -> Optional[float]:
    try:
        if x is None: return None
        if isinstance(x, (int, float)): return float(x)
        s = str(x).replace(",", "").strip().lower()
        if s in ("", "none", "null", "nan"): return None
        return float(s)
    except Exception:
        return None

def _extract_closes_vols(ohlcv_list):
    """Return (closes, volumes) from history.ohlcv list of bars."""
    closes, vols = [], []
    if not isinstance(ohlcv_list, list): return closes, vols
    for b in ohlcv_list:
        if not isinstance(b, dict): continue
        c = _to_float(b.get("c") or b.get("close") or b.get("price") or b.get("C"))
        v = _to_float(b.get("v") or b.get("volume") or b.get("V"))
        if c is not None:
            closes.append(c)
            vols.append(v if v is not None else 0.0)
    return closes, vols

def _compute_vwap(closes, vols, lookback=120):
    if not closes or not vols: return None
    p = closes[-lookback:] if len(closes) >= lookback else closes
    v = vols[-lookback:] if len(vols) >= lookback else vols
    denom = float(np.sum(v))
    return float(np.sum(np.array(p, float) * np.array(v, float)) / denom) if denom else None

def _compute_rsi14(closes):
    c = np.asarray(closes, dtype=float)
    d = np.diff(c)
    if d.size < 14: return None
    up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    au = up[:14].mean(); ad = dn[:14].mean() or 1e-6
    rs = au / ad; rsi = 100 - 100 / (1 + rs)
    for i in range(14, d.size):
        au = (au * 13 + max(d[i], 0)) / 14
        ad = (ad * 13 + max(-d[i], 0)) / 14 or 1e-6
        rs = au / ad; rsi = 100 - 100 / (1 + rs)
    return float(rsi)

def _extract_cvd_delta(cvd_hist):
    """From history.cvd list; supports dict items with 'cvd'/'value' keys."""
    if not isinstance(cvd_hist, list) or not cvd_hist: return None, None
    def _cvd_of(x):
        if isinstance(x, dict):
            return _to_float(x.get("cvd") or x.get("value") or x.get("v"))
        return _to_float(x)
    last = _cvd_of(cvd_hist[-1])
    prev = _cvd_of(cvd_hist[-2]) if len(cvd_hist) >= 2 else None
    dlt  = (last - prev) if (last is not None and prev is not None) else None
    return last, dlt

# ---------------------------- routes ----------------------------
@app.get("/")
def root():
    return {"status": "online", "name": APP_NAME, "version": VERSION}

@app.get("/health")
def health():
    files = _list_snapshot_files()
    return {
        "status": "ok",
        "data_dir": str(DATA_DIR),
        "file_count": len(files),
        "latest_file": os.path.basename(files[0]) if files else None,
        "ring_items": len(_ring),
    }

@app.get("/v1/metrics/latest")
def get_latest():
    latest = _latest_from_ring()
    if latest is not None:
        return latest
    files = _list_snapshot_files()
    if not files:
        raise HTTPException(status_code=404, detail="No metrics found")
    with open(files[0], "r") as f:
        return json.load(f)

@app.get("/v1/metrics/all")
def get_all(limit: int = Query(50, ge=1, le=1000)):
    out = list(_ring[:limit])
    remaining = max(0, limit - len(out))
    if remaining:
        files = _list_snapshot_files()
        for fp in files[:remaining]:
            try:
                with open(fp, "r") as f:
                    out.append(json.load(f))
            except Exception:
                continue
    if not out:
        raise HTTPException(status_code=404, detail="No metrics found")
    return JSONResponse(content={"count": len(out), "metrics": out})

@app.post("/v1/ingest")
async def ingest(request: Request, x_auth_token: str = Header(default="")):
    if x_auth_token not in (INGEST_TOKEN, f"Bearer {INGEST_TOKEN}"):
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Payload must be a JSON object")
        ts = int(payload.get("timestamp") or time.time())
        payload["timestamp"] = ts
        # write to local dir (best-effort)
        fpath = DATA_DIR / f"{ts}.json"
        with open(fpath, "w") as f:
            json.dump(payload, f, separators=(",", ":"), ensure_ascii=False)
        # keep in memory
        _push_ring(payload)
        return {"status": "ok", "stored": True, "file": fpath.name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/v1/metrics/summary")
def metrics_summary():
    """
    Build a compact, null-safe summary from the collector pack:
    - price from history.ohlcv last close
    - vwap over last N candles (if volume is available)
    - rsi(14) from closes (if not provided)
    - cvd & delta from history.cvd list
    - (optional) oi / funding / long_short_ratio / liq when present
    """
    latest = _latest_from_ring()
    if latest is None:
        files = _list_snapshot_files()
        if not files:
            raise HTTPException(status_code=404, detail="No metrics files found")
        with open(files[0], "r") as f:
            latest = json.load(f)

    # If the collector already flattened values, keep using them.
    flat_price = _to_float(latest.get("price"))
    flat_vwap  = _to_float(latest.get("vwap"))
    flat_rsi   = _to_float(latest.get("rsi"))
    flat_cvd   = _to_float(latest.get("cvd"))
    flat_delta = _to_float(latest.get("delta"))
    flat_vol   = _to_float(latest.get("volume"))

    # Prefer structured extraction when flat fields are null/empty
    hist  = latest.get("history", {}) if isinstance(latest, dict) else {}
    snaps = latest.get("snapshots", {}) if isinstance(latest, dict) else {}

    closes, vols = _extract_closes_vols(hist.get("ohlcv"))
    price = flat_price if flat_price is not None else (closes[-1] if closes else None)
    vwap  = flat_vwap  if flat_vwap  is not None else (_compute_vwap(closes, vols, lookback=120) if (closes and vols) else None)

    rsi = flat_rsi
    if rsi is None and closes and len(closes) >= 15:
        rsi = _compute_rsi14(closes)

    cvd, delta = flat_cvd, flat_delta
    if cvd is None or delta is None:
        cvd_hist = hist.get("cvd")
        c, d = _extract_cvd_delta(cvd_hist)
        if cvd is None:   cvd = c
        if delta is None: delta = d

    vol_now = flat_vol if flat_vol is not None else (vols[-1] if vols else None)

    # Optional extras (exposed if the collector wrote them)
    oi = _to_float(snaps.get("oi_value") or snaps.get("open_interest") or latest.get("oi"))
    funding_rate = _to_float(snaps.get("fr_value") or snaps.get("funding_rate") or latest.get("funding_rate"))
    long_short_ratio = None
    lsr_hist = hist.get("long_short_ratio")
    if isinstance(lsr_hist, list) and lsr_hist:
        last_lsr = lsr_hist[-1]
        long_short_ratio = _to_float(last_lsr.get("ratio") if isinstance(last_lsr, dict) else last_lsr)
    if long_short_ratio is None:
        long_short_ratio = _to_float(snaps.get("long_short_ratio") or latest.get("long_short_ratio"))

    liq = None
    liq_hist = hist.get("liquidations")
    if isinstance(liq_hist, list) and liq_hist:
        # Sum last 50 bars if numeric, else report count
        vals = []
        for e in liq_hist[-50:]:
            v = None
            if isinstance(e, dict):
                v = _to_float(e.get("qty") or e.get("amount") or e.get("value"))
            else:
                v = _to_float(e)
            if v is not None:
                vals.append(v)
        liq = float(sum(vals)) if vals else float(len(liq_hist[-50:]))

    payload = {
        "timestamp": latest.get("fetched_at") or latest.get("timestamp"),
        "price": price,
        "volume": vol_now,
        "vwap": vwap,
        "rsi": rsi,
        "cvd": cvd if cvd is not None else 0.0,
        "delta": delta if delta is not None else 0.0,
        # extras if present
        "oi": oi,
        "funding_rate": funding_rate,
        "long_short_ratio": long_short_ratio,
        "liquidations": liq,
        "source_file": latest.get("source_file") or None
    }
    return JSONResponse(content=payload)
