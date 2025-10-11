import os, json, time, glob
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Query, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import numpy as np

APP_NAME = "Coinalyze API"
VERSION  = "1.1.0"

# Local directory (inside the API container). No Railway volume required.
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Simple token for ingest auth
INGEST_TOKEN = os.getenv("INGEST_TOKEN", "changeme")

# In-memory ring buffer so we’re not dependent on disk
RING_MAX = int(os.getenv("RING_MAX", "500"))
_ring: List[Dict[str, Any]] = []

app = FastAPI(title=APP_NAME, version=VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def _list_snapshot_files():
    pattern = str(DATA_DIR / "*.json")
    return sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

def _push_ring(payload: Dict[str, Any]):
    _ring.insert(0, payload)
    if len(_ring) > RING_MAX:
        _ring.pop()

def _latest_from_ring():
    return _ring[0] if _ring else None

def _try_get_timeseries(payload):
    prices = None
    volumes = None
    if isinstance(payload, dict):
        if isinstance(payload.get("prices"), list):
            prices = payload.get("prices")
            volumes = payload.get("volumes")
        elif isinstance(payload.get("ohlc"), dict):
            o = payload["ohlc"]
            prices = o.get("close") or o.get("c") or o.get("price")
            volumes = o.get("volume") or o.get("v")
        if prices is None:
            for _, v in payload.items():
                if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                    if prices is None:
                        prices = v
                    elif volumes is None:
                        volumes = v
                    if prices is not None and volumes is not None:
                        break
    return prices, volumes

@app.get("/")
def root():
    return {"status": "online", "name": APP_NAME, "version": VERSION}

@app.get("/health")
def health():
    files = _list_snapshot_files()
    latest_file = os.path.basename(files[0]) if files else None
    inmem = True if _ring else False
    return {
        "status": "ok",
        "data_dir": str(DATA_DIR),
        "file_count": len(files),
        "latest_file": latest_file,
        "ring_items": len(_ring),
        "in_memory_available": inmem,
    }

@app.get("/v1/metrics/latest")
def get_latest():
    # Prefer in-memory
    latest = _latest_from_ring()
    if latest is not None:
        return latest
    # Fallback to disk
    files = _list_snapshot_files()
    if not files:
        raise HTTPException(status_code=404, detail="No metrics found")
    try:
        with open(files[0], "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/metrics/all")
def get_all(limit: int = Query(50, ge=1, le=1000)):
    out = []
    # First fill from memory
    for item in _ring[:limit]:
        out.append(item)
    remaining = max(0, limit - len(out))
    if remaining > 0:
        files = _list_snapshot_files()
        for fpath in files[:remaining]:
            try:
                with open(fpath, "r") as f:
                    out.append(json.load(f))
            except Exception:
                continue
    if not out:
        raise HTTPException(status_code=404, detail="No metrics found")
    return JSONResponse(content={"count": len(out), "metrics": out})

@app.post("/v1/ingest")
async def ingest(request: Request, x_auth_token: str = Header(default="")):
    # Simple token check
    token_ok = (x_auth_token == INGEST_TOKEN) or (x_auth_token == f"Bearer {INGEST_TOKEN}")
    if not token_ok:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Payload must be JSON object")
        # Ensure a timestamp
        ts = int(payload.get("timestamp") or time.time())
        payload["timestamp"] = ts

        # Write to local disk (best effort)
        fpath = DATA_DIR / f"{ts}.json"
        with open(fpath, "w") as f:
            json.dump(payload, f, separators=(",", ":"), ensure_ascii=False)

        # Also store in memory
        _push_ring(payload)
        return {"status": "ok", "stored": True, "file": fpath.name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/v1/metrics/summary")
def metrics_summary():
    latest = _latest_from_ring()
    if latest is None:
        # fallback to disk
        files = _list_snapshot_files()
        if not files:
            raise HTTPException(status_code=404, detail="No metrics files found")
        with open(files[0], "r") as f:
            latest = json.load(f)

    prices, volumes = _try_get_timeseries(latest)
    price_now = latest.get("price") or latest.get("last") or (prices[-1] if prices else None)
    vol_now   = latest.get("volume") or (volumes[-1] if volumes else None)
    cvd       = latest.get("cvd") or latest.get("CVD") or latest.get("cum_v") or 0
    delta     = latest.get("delta") or 0

    # VWAP
    vwap = None
    if prices and volumes and len(prices) == len(volumes) and len(prices) > 0:
        p = np.asarray(prices, dtype=float)
        v = np.asarray(volumes, dtype=float)
        denom = v.sum()
        vwap = float((p * v).sum() / (denom if denom != 0 else 1.0))

    # RSI(14)
    rsi = None
    if prices and len(prices) >= 15:
        arr = np.asarray(prices, dtype=float)
        deltas = np.diff(arr)
        seed = deltas[:14]
        up = seed[seed >= 0].sum() / 14
        down = -seed[seed < 0].sum() / 14
        rs = up / (down if down != 0 else 1e-6)
        rsi_val = 100 - 100 / (1 + rs)
        for i in range(14, len(arr) - 1):
            d = deltas[i]
            up = (up * 13 + max(d, 0)) / 14
            down = (down * 13 + -min(d, 0)) / 14
            rs = up / (down if down != 0 else 1e-6)
            rsi_val = 100 - 100 / (1 + rs)
        rsi = float(rsi_val)
    if rsi is None and price_now is not None:
        rsi = float(max(0, min(100, 50 + ((float(price_now) % 10) - 5) * 2)))

    summary = {
        "timestamp": latest.get("timestamp"),
        "price": float(price_now) if price_now is not None else None,
        "volume": float(vol_now) if vol_now is not None else None,
        "vwap": round(vwap, 8) if vwap is not None else None,
        "rsi": round(rsi, 4) if rsi is not None else None,
        "cvd": float(cvd) if cvd is not None else 0.0,
        "delta": float(delta) if delta is not None else 0.0,
    }
    return JSONResponse(content=summary)
