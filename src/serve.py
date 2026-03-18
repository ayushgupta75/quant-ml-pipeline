from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path
import subprocess, uuid, json, time
import yaml
import pandas as pd
from src.ingest import ingest
from src.features import build_features

app = FastAPI(title="Quant-ML Pipeline API")

# Use the deploy bundle your backtest writes
MODEL_PATH = Path("artifacts/best_model.joblib")
JOBS_DIR = Path("artifacts/jobs")

model = None
feature_cols = None
bundle_kind = None


def _write_job(job_id: str, payload: dict):
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    (JOBS_DIR / f"{job_id}.json").write_text(json.dumps(payload, indent=2))


def load_model_bundle():
    global model, feature_cols, bundle_kind
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Missing {MODEL_PATH}. Run: python -m src.backtest first.")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    bundle_kind = bundle.get("kind", "baseline")  # baseline / tcn (if you saved it)


@app.on_event("startup")
def startup():
    # Load model if present; don't crash container if missing
    if MODEL_PATH.exists():
        load_model_bundle()


# -------------------- TRAIN --------------------

class TrainRequest(BaseModel):
    symbols: list[str]
    start_date: str | None = None
    end_date: str | None = None


@app.post("/train")
def train(req: TrainRequest):
    if not req.symbols:
        raise HTTPException(400, "symbols list cannot be empty")

    job_id = str(uuid.uuid4())[:8]
    _write_job(job_id, {
        "job_id": job_id,
        "status": "running",
        "symbols": req.symbols,
        "started_at": time.time(),
    })

    # Kick off training as a subprocess (OK for local/dev).
    # In production (Cloud Run), replace this with a Cloud Run Job.
    cmd = ["python", "-m", "src.backtest", "--symbols", ",".join(req.symbols)]
    subprocess.Popen(cmd)

    return {"job_id": job_id, "status": "started", "cmd": " ".join(cmd)}


@app.get("/train/{job_id}")
def train_status(job_id: str):
    p = JOBS_DIR / f"{job_id}.json"
    if not p.exists():
        raise HTTPException(404, "job_id not found")
    return json.loads(p.read_text())


# -------------------- INFERENCE --------------------

class PredictRequest(BaseModel):
    features: dict  # {"ret_1d": 0.01, ...}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "bundle_kind": bundle_kind,
        "model_path": str(MODEL_PATH),
    }


@app.post("/reload")
def reload_model():
    """Reload the latest artifacts/best_model.joblib without restarting the container."""
    try:
        load_model_bundle()
        return {"status": "reloaded", "bundle_kind": bundle_kind}
    except Exception as e:
        raise HTTPException(500, f"Reload failed: {e}")


@app.post("/predict")
def predict(req: PredictRequest):
    if model is None or feature_cols is None:
        raise HTTPException(503, "Model not loaded. Run /reload after training or ensure best_model.joblib exists.")

    x = np.array([[float(req.features.get(c, 0.0)) for c in feature_cols]], dtype=float)

    if hasattr(model, "predict_proba"):
        p = float(model.predict_proba(x)[0, 1])
        return {"prob_up": p, "pred": int(p >= 0.5)}

    y = float(model.predict(x)[0])
    return {"prediction": y}

@app.get("/predict_symbol")
def predict_symbol(symbol: str):
    if model is None or feature_cols is None:
        raise HTTPException(503, "Model not loaded. Run /reload or ensure artifacts/best_model.joblib exists.")

    symbol = symbol.strip().upper()
    if not symbol:
        raise HTTPException(400, "symbol is required")

    # Load config so this endpoint uses the same pipeline settings
    cfg = yaml.safe_load(open("config.yaml"))
    cfg["symbols"] = [symbol]

    # Ingest + build features for this symbol
    prices, _ = ingest(cfg["symbols"], cfg["start_date"], cfg.get("end_date"))
    feat_df, _ = build_features(prices, cfg)

    # Ensure required cols exist
    missing = [c for c in feature_cols if c not in feat_df.columns]
    if missing:
        raise HTTPException(500, f"Missing feature columns in feat_df: {missing[:10]}")

    # Need last lookback rows for TCN
    lookback = int(bundle.get("lookback", 60)) if "bundle" in globals() and bundle else 60
    # If you stored lookback in your bundle dict, use that
    try:
        lookback = int(joblib.load("artifacts/best_model.joblib").get("lookback", lookback))
    except Exception:
        pass

    sym_df = feat_df.sort_values("Date").tail(lookback)
    if len(sym_df) < lookback:
        raise HTTPException(400, f"Not enough history for {symbol}. Need {lookback} rows, got {len(sym_df)}.")

    # Build tensor: (1, F, T)
    X = np.array([[sym_df[c].astype(float).tolist() for c in feature_cols]], dtype=np.float32)

    # Predict
    import torch
    with torch.no_grad():
        logits = model(torch.tensor(X))
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        p = torch.sigmoid(logits).cpu().numpy().reshape(-1)[0]

    p = float(p)
    return {
        "symbol": symbol,
        "as_of": str(sym_df["Date"].iloc[-1].date()),
        "prob_up": p,
        "pred": int(p >= 0.5),
        "lookback": lookback
    }