import os, json, glob
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

APP_NAME = "Coinalyze API"
VERSION = "1.0.0"
DATA_DIR = os.getenv("DATA_DIR", "/data")  # Railway Volume mount path

app = FastAPI(title=APP_NAME, version=VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def _list_snapshot_files():
    pattern = os.path.join(DATA_DIR, "*.json")
    return sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

@app.get("/health")
def health():
    exists = os.path.isdir(DATA_DIR)
    files = _list_snapshot_files()
    return {
        "status": "ok",
        "data_dir": DATA_DIR,
        "dir_exists": exists,
        "file_count": len(files),
        "latest_file": os.path.basename(files[0]) if files else None,
    }

@app.get("/v1/metrics/latest")
def get_latest():
    files = _list_snapshot_files()
    if not files:
        raise HTTPException(status_code=404, detail="No metrics files found")
    try:
        with open(files[0], "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/metrics/all")
def get_all(limit: int = Query(50, ge=1, le=1000)):
    files = _list_snapshot_files()
    if not files:
        raise HTTPException(status_code=404, detail="No metrics files found")
    out = []
    for fpath in files[:limit]:
        try:
            with open(fpath, "r") as f:
                out.append(json.load(f))
        except Exception as e:
            # skip corrupt files, continue
            continue
    return JSONResponse(content={"count": len(out), "metrics": out})

@app.get("/")
def root():
    return {"status": "online", "name": APP_NAME, "version": VERSION}
