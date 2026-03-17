# quant-ml-pipeline

An end-to-end financial machine learning pipeline that ingests OHLCV + fundamentals + news signals, builds leakage-safe features, trains baseline ML models and a deep learning TCN (Temporal Convolutional Network), evaluates using walk-forward backtesting, tracks experiments with MLflow, and serves predictions via a FastAPI inference service.

Note: This is an educational/research project, not financial advice.

---

## What this project does

Data pipeline
• Downloads/ingests OHLCV (price data) for multiple tickers
• Pulls fundamentals (slow-moving features)
• Builds news signals (per-symbol + optional global) and lags them to reduce leakage

Modeling
• Baseline model training (classification/regression)
• Deep learning model: TCN for time-series direction prediction (classification)
• Feature engineering: returns, volatility, moving averages, RSI, plus news features

Evaluation
• Leakage-safe walk-forward validation/backtesting
• Metrics logged per split in MLflow
• Saves results to artifacts/walk_forward_results.csv

Deployment
• Exports a deployable bundle: artifacts/best_model.joblib
• Runs a FastAPI server with endpoints for health, predict, train trigger, reload

---

## Repository structure

quant-ml-pipeline/
• Dockerfile
• config.yaml
• requirements.txt
• mlflow.db
• data/
– raw/
– processed/
• artifacts/
– walk_forward_results.csv
– best_model.joblib
– jobs/
• src/
– ingest.py
– news.py
– features.py
– split.py
– train_baseline.py
– datasets.py
– train_deep.py
– backtest.py
– serve.py

---

## Setup

Step 1: Create and activate virtual environment

1. python -m venv .venv
2. source .venv/bin/activate
3. python -m pip install --upgrade pip
4. pip install -r requirements.txt

---

## Configuration (config.yaml)

Common knobs:
• symbols: tickers to train on
• start_date, end_date
• task: classification or regression
• horizon_days: prediction horizon
• walk_forward: train/test/step window sizes
• deep.lookback: TCN sequence length
• news: RSS feeds + lag_days (to reduce leakage)
• mlflow.tracking_uri: recommended sqlite:///mlflow.db

Example config settings (conceptual)
• task: classification
• horizon_days: 1
• walk_forward train_days: 2520
• walk_forward test_days: 63
• walk_forward step_days: 63
• deep lookback: 60
• mlflow tracking_uri: sqlite:///mlflow.db
• mlflow experiment_name: quant-ml-pipeline

---

## Run walk-forward backtest (training + evaluation)

Run default (from config.yaml)
• python -m src.backtest

Override tickers at runtime
• python -m src.backtest --symbols SPY,AAPL,MSFT,TSLA

Outputs created
• artifacts/walk_forward_results.csv
• artifacts/best_model.joblib

---

## MLflow tracking

If using SQLite backend in config.yaml (recommended):
• tracking_uri: sqlite:///mlflow.db

Start MLflow UI
• mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

Open in browser
• [http://localhost:5001](http://localhost:5001)

---

## Serve predictions (FastAPI)

Start the API server
• uvicorn src.serve:app --host 0.0.0.0 --port 8080 --reload

Health check
• curl [http://localhost:8080/health](http://localhost:8080/health)

Expected response should indicate:
• status ok
• model_loaded true
• bundle_kind tcn or baseline
• model_path artifacts/best_model.joblib

---

## Predict endpoint (TCN bundle)

If the exported bundle kind is TCN, /predict expects a sequence payload:
• A JSON object with key “seq”
• “seq” is a list of length lookback
• Each element is a feature dictionary for one timestep

How to generate a valid payload automatically (writes payload.json):

1. Run a Python snippet to:
   – load artifacts/best_model.joblib
   – read feature_cols and lookback
   – create seq of zeros
   – write payload.json

Then call predict using the payload file:
• curl -X POST [http://localhost:8080/predict](http://localhost:8080/predict) -H "Content-Type: application/json" --data-binary @payload.json

The response returns:
• prob_up
• pred (0/1)

---

## Change training tickers via API (retrain trigger)

Start a training run for new tickers
• curl -X POST [http://localhost:8080/train](http://localhost:8080/train) -H "Content-Type: application/json" -d {"symbols":["SPY","AAPL","MSFT"]}

Reload the latest model after training finishes
• curl -X POST [http://localhost:8080/reload](http://localhost:8080/reload)

Note: /train runs training via subprocess for local/dev convenience. For production, replace this with Cloud Run Jobs / Scheduler / VM cron.

---

## Why TCN?

TCN (Temporal Convolutional Network) is a strong deep-learning model for time series because it:
• trains faster than RNNs (parallelizable)
• captures temporal dependencies with causal/dilated convolutions
• fits rolling-window feature pipelines cleanly

---

## Roadmap / Next improvements

• Cache OHLCV/features so predict endpoints don’t re-download data each request
• Store model artifacts in cloud storage (S3/GCS) and load by version
• Add drift monitoring (feature PSI/KL + performance decay)
• Add realistic trading evaluation: transaction costs, slippage, turnover
• Add per-symbol evaluation and calibration
