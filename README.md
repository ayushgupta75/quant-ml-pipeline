# Quant-ML Pipeline

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Code style: Black](https://img.shields.io/badge/Code%20Style-Black-000?style=flat-square)](https://github.com/psf/black)
![Status: Active Development](https://img.shields.io/badge/Status-Active%20Development-brightgreen?style=flat-square)

An end-to-end production-ready financial machine learning pipeline for equities price direction prediction. Ingests OHLCV data, fundamentals, and news signals with leakage-safe feature engineering. Trains baseline ML and deep learning (TCN) models, evaluates via walk-forward backtesting, tracks experiments with MLflow, and serves predictions via FastAPI.

**⚠️ Disclaimer**: This is an educational/research project. Not financial advice. Use at your own risk.

---

## 📋 Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Monitoring & Tracking](#monitoring--tracking)
- [API Reference](#api-reference)
- [Design Principles](#design-principles)
- [Future Work](#future-work)
- [Contributing](#contributing)

---

## ✨ Features

### Data Ingestion & Processing
- **Multi-source data**: OHLCV (yfinance), fundamentals, sentiment signals
- **News integration**: Per-symbol Yahoo Finance + global RSS feeds with configurable lags
- **Leakage prevention**: Automatic lagging of news signals to avoid look-ahead bias
- **Efficient caching**: Reduced redundant API calls through smart data management

### Feature Engineering
- **Technical indicators**: Returns (multiple windows), volatility, moving averages, RSI
- **Sentiment features**: News-based sentiment scoring (VADER)
- **Configurable windows**: All feature windows easily tunable via `config.yaml`

### Modeling
- **Baseline models**: Sklearn-based classification/regression (LightGBM, Random Forest, etc.)
- **Deep learning**: Temporal Convolutional Networks (TCN) for time-series patterns
- **Sequence generation**: Automatic windowed sequence creation for deep models

### Backtesting & Evaluation
- **Walk-forward validation**: Time-series aware, leakage-free train/test splits
- **Rich metrics**: Sharpe ratio, hit rate, trade rate, average trade returns
- **Experiment tracking**: MLflow integration for reproducibility and comparison
- **Results export**: CSV reports for analysis and sharing

### Deployment
- **Model bundling**: Serialized models with feature metadata
- **FastAPI service**: Production-ready REST API with async training support
- **Docker support**: Easy containerization for cloud deployment
- **Health checks**: Built-in model validation endpoints

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip or conda

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/quant-ml-pipeline.git
cd quant-ml-pipeline

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Run Full Pipeline

```bash
# Default: trains on AAPL, MSFT, AMZN from config.yaml
python -m src.backtest

# Override tickers at runtime
python -m src.backtest --symbols SPY,AAPL,MSFT,TSLA

# Specify custom config
python -m src.backtest --config custom_config.yaml
```

### Start API Server

```bash
# Start prediction server (runs on http://localhost:8080)
uvicorn src.serve:app --host 0.0.0.0 --port 8080 --reload

# In another terminal, check health
curl http://localhost:8080/health
```

### View Experiment Results

```bash
# Start MLflow UI (runs on http://localhost:5001)
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

---

## 🏗️ Architecture

```
Data Acquisition
    ↓
┌─────────────────────────────────────┐
│ 1. INGEST (ingest.py)              │
│    - OHLCV via yfinance            │
│    - Fundamentals                  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 2. FEATURES (features.py)           │
│    - Technical indicators           │
│    - News sentiment (news.py)       │
│    - Lag-aware construction         │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 3. SPLIT (split.py)                 │
│    - Walk-forward windows           │
│    - Time-aware train/test splits   │
└─────────────────────────────────────┘
    ↓
┌──────────────────┬──────────────────┐
│ 4a. BASELINE     │ 4b. DEEP LEARNING│
│ (train_baseline) │ (train_deep.py)  │
│ Sklearn models   │ TCN + PyTorch    │
└──────────────────┴──────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 5. BACKTEST (backtest.py)           │
│    - Walk-forward evaluation        │
│    - MLflow tracking                │
│    - Best model selection & export  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 6. SERVE (serve.py)                 │
│    - FastAPI REST endpoints         │
│    - Model predictions              │
│    - Async retraining               │
└─────────────────────────────────────┘
```

---

## 📁 Project Structure

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
