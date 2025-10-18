import os, json, time, glob
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Query, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np

APP_NAME = "Coinalyze API"
VERSION  = "1.1.1"

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

INGEST_TOKEN = os.getenv("INGEST_TOKEN", "changeme")
RING_MAX = int(os.getenv("RING_MAX", "500"))
_ring: List[Dict[str, Any]] = []

app = FastAPI(title=APP_NAME, version=VERSION)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ---------- file helpers ----------
def _list_snapshot_files() -> List[str]:
    return sorted(glob.glob(str(DATA_DIR / "*.json")), key=os.path.getmtime, reverse=True)

def _push_ring(payload: Dict[str, Any]):
    _ring.insert(0, payload)
    if len(_ring) > RING_MAX:
        _ring.pop()

def _latest_from_ring() -> Optional[Dict[str, Any]]:
    return _ring[0] if _ring else None

def _to_float(x):
    try:
        if x is None: return None
        if isinstance(x, (int, float)): return float(x)
        s = str(x).replace(",","").strip().lower()
        if s in ("", "none", "null", "nan"): return None
        return float(s)
    except Exception:
        return None

def _extract_closes_vols(ohlcv_list):
    closes, vols = [], []
    if not isinstance(ohlcv_list, list):
        return closes, vols
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
    up = np.where(d>0, d, 0.0)
    dn = np.where(d<0, -d, 0.0)
    au = up[:14].mean()
    ad = dn[:14].mean() or 1e-6
    rs = au/ad
    rsi = 100 - 100/(1+rs)
    for i in range(14, d.size):
        au = (au*13 + max(d[i],0))/14
        ad = (ad*13 + max(-d[i],0))/14 or 1e-6
        rs = au/ad
        rsi = 100 - 100/(1+rs)
    return float(rsi)

def _extract_cvd_delta(cvd_hist):
    if not isinstance(cvd_hist, list) or not cvd_hist: return None, None
    def _cvd_of(x):
        if isinstance(x, dict):
            return _to_float(x.get("cvd") or x.get("value") or x.get("v"))
        return _to_float(x)
    last = _cvd_of(cvd_hist[-1])
    prev = _cvd_of(cvd_hist[-2]) if len(cvd_hist) >= 2 else None
    dlt  = (last - prev) if (last is not None and prev is not None) else None
    return last, dlt

# ---------- routes ----------
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
        fpath = DATA_DIR / f"{ts}.json"
        with open(fpath, "w") as f:
            json.dump(payload, f, separators=(",", ":"), ensure_ascii=False)
        _push_ring(payload)
        return {"status": "ok", "stored": True, "file": fpath.name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/v1/metrics/summary")
def metrics_summary(symbol: str | None = Query(default=None)):
    """
    Builds a compact summary from the newest pack (optionally the newest pack for `symbol`).
    Reads only local snapshots; no external requests.
    """
    latest = _latest_from_ring()

    if latest is None:
        files = _list_snapshot_files()
        if not files:
            raise HTTPException(status_code=404, detail="No metrics files found")

        # If symbol is specified, pick the first file in reverse order matching it
        if symbol:
            symu = symbol.upper()
            for fp in files:
                try:
                    with open(fp, "r") as f:
                        pack = json.load(f)
                    if (pack.get("symbol") or "").upper().endswith(symu) or (pack.get("symbol") or "").upper() == symu:
                        latest = pack
                        break
                except Exception:
                    continue
        if latest is None:
            with open(files[0], "r") as f:
                latest = json.load(f)

    hist  = latest.get("history", {}) if isinstance(latest, dict) else {}
    snaps = latest.get("snapshots", {}) if isinstance(latest, dict) else {}

    closes, vols = _extract_closes_vols(hist.get("ohlcv"))
    price = _to_float(latest.get("price")) or (closes[-1] if closes else None)
    vwap  = _to_float(latest.get("vwap"))  or (_compute_vwap(closes, vols, lookback=120) if (closes and vols) else None)
    rsi   = _to_float(latest.get("rsi"))
    if rsi is None and closes and len(closes) >= 15:
        rsi = _compute_rsi14(closes)

    cvd, delta = _to_float(latest.get("cvd")), _to_float(latest.get("delta"))
    if cvd is None or delta is None:
        c, d = _extract_cvd_delta(hist.get("cvd"))
        if cvd is None:   cvd = c
        if delta is None: delta = d

    vol_now = _to_float(latest.get("volume")) or (vols[-1] if vols else None)

    oi = _to_float(snaps.get("oi_value") or snaps.get("open_interest") or latest.get("oi"))
    funding_rate = _to_float(snaps.get("fr_value") or snaps.get("funding_rate") or latest.get("funding_rate"))

    # optional extras
    lsr_hist = hist.get("long_short_ratio")
    long_short_ratio = None
    if isinstance(lsr_hist, list) and lsr_hist:
        last = lsr_hist[-1]
        long_short_ratio = _to_float(last.get("ratio") if isinstance(last, dict) else last)
    if long_short_ratio is None:
        long_short_ratio = _to_float(snaps.get("long_short_ratio") or latest.get("long_short_ratio"))

    liq_hist = hist.get("liquidations")
    liq = None
    if isinstance(liq_hist, list) and liq_hist:
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
        "symbol": latest.get("symbol"),
        "price": price,
        "volume": vol_now,
        "vwap": vwap,
        "rsi": rsi,
        "cvd": cvd if cvd is not None else 0.0,
        "delta": delta if delta is not None else 0.0,
        "oi": oi,
        "funding_rate": funding_rate,
        "long_short_ratio": long_short_ratio,
        "liquidations": liq,
    }
    return JSONResponse(content=payload)
