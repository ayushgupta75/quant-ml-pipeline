# src/backtest.py
import yaml
import pandas as pd
import mlflow
import argparse
from pathlib import Path
import numpy as np

from src.datasets import make_seq_train_test
from src.train_deep import train_tcn

from src.ingest import ingest
from src.news import daily_news_signals
from src.features import build_features
from src.split import walk_forward_splits
from src.train_baseline import train_baseline


def strat_metrics_from_probs(meta_df, probs, threshold, horizon_days: int):
    future_ret = meta_df["future_ret"].values.astype(float)
    pos = (probs >= threshold).astype(int)
    strat = pos * future_ret

    mean_ret = float(np.mean(strat))
    std_ret = float(np.std(strat) + 1e-12)
    ann_factor = float(np.sqrt(252.0 / max(horizon_days, 1)))
    sharpe = float((mean_ret / std_ret) * ann_factor)

    trade_rate = float(np.mean(pos))
    avg_trade_ret = float(np.mean(strat[pos == 1]) if pos.sum() > 0 else 0.0)
    hit_rate = float(np.mean((strat[pos == 1] > 0).astype(int)) if pos.sum() > 0 else 0.0)

    return {
        "mean_strat_ret": mean_ret,
        "sharpe": sharpe,
        "trade_rate": trade_rate,
        "avg_trade_ret": avg_trade_ret,
        "hit_rate": hit_rate,
    }


def run(cfg_path="config.yaml"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=cfg_path)
    parser.add_argument("--symbols", default=None, help="Comma-separated tickers, e.g. SPY,AAPL,MSFT")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    if args.symbols:
        cfg["symbols"] = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    # Guard: prevent accidental regression when you expect classification
    if cfg.get("task") not in ("classification", "regression"):
        raise ValueError(f"cfg['task'] must be 'classification' or 'regression', got: {cfg.get('task')}")

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    # Ingest
    prices, fundamentals = ingest(cfg["symbols"], cfg["start_date"], cfg["end_date"])

    # Features (includes y + future_ret)
    feat_df, feature_cols = build_features(prices, cfg)

    print("y head:", feat_df["y"].head().tolist())
    print("y mean:", float(np.mean(feat_df["y"].values)))
    print("feat_df rows:", len(feat_df))
    print("Date min:", feat_df["Date"].min(), "Date max:", feat_df["Date"].max())
    print("Symbols:", feat_df["Symbol"].nunique())

    # If classification, y must be 0/1
    if cfg["task"] == "classification":
        uniq = pd.Series(feat_df["y"]).dropna().unique()
        if not set(uniq).issubset({0, 1}):
            raise ValueError(
                f"Task is classification but y has non-binary values. "
                f"Check config.yaml task/horizon and features.py. Unique y sample: {uniq[:10]}"
            )

    # ---- NEWS: per-symbol signals (Date + Symbol)
    news_daily = daily_news_signals(cfg["symbols"], cfg["news"])
    print("news_daily rows:", len(news_daily))
    print(news_daily.groupby("Symbol")["news_count"].sum().sort_values(ascending=False).head(10))
    print("latest news date:", news_daily["Date"].max())

    # Merge per-symbol news
    feat_df = feat_df.merge(news_daily, on=["Date", "Symbol"], how="left")
    feat_df["news_count"] = feat_df.get("news_count", 0).fillna(0).astype(float)
    feat_df["news_sent_mean"] = feat_df.get("news_sent_mean", 0).fillna(0).astype(float)

    # Optional GLOBAL news
    if not news_daily.empty and "GLOBAL" in set(news_daily["Symbol"].unique()):
        global_news = news_daily[news_daily["Symbol"] == "GLOBAL"].drop(columns=["Symbol"])
        feat_df = feat_df.merge(global_news, on="Date", how="left", suffixes=("", "_global"))
        feat_df["news_count_global"] = feat_df["news_count_global"].fillna(0).astype(float)
        feat_df["news_sent_mean_global"] = feat_df["news_sent_mean_global"].fillna(0).astype(float)
    else:
        feat_df["news_count_global"] = 0.0
        feat_df["news_sent_mean_global"] = 0.0

    # Lag news to avoid leakage
    lag = int(cfg["news"].get("lag_days", 1))
    for c in ["news_count", "news_sent_mean", "news_count_global", "news_sent_mean_global"]:
        feat_df[c] = feat_df.groupby("Symbol")[c].shift(lag)
    feat_df[["news_count", "news_sent_mean", "news_count_global", "news_sent_mean_global"]] = (
        feat_df[["news_count", "news_sent_mean", "news_count_global", "news_sent_mean_global"]].fillna(0)
    )

    # Add to features
    feature_cols = feature_cols + ["news_count", "news_sent_mean", "news_count_global", "news_sent_mean_global"]

    # Quick verification print
    check_day = feat_df["Date"].max()
    print("News check on", check_day)
    print(
        feat_df[feat_df["Date"] == check_day][["Symbol", "news_count", "news_sent_mean"]]
        .sort_values("Symbol")
        .head(20)
    )

    # Walk-forward
    wf = cfg["walk_forward"]
    results = []

    for train_df, test_df, (train_end, test_end) in walk_forward_splits(
        feat_df, wf["train_days"], wf["test_days"], wf["step_days"]
    ):
        with mlflow.start_run(run_name=f"wf_{train_end.date()}"):
            mlflow.log_params({
                "task": cfg["task"],
                "horizon_days": cfg["horizon_days"],
                "train_end": str(train_end.date()),
                "test_end": str(test_end.date()),
                "n_train": len(train_df),
                "n_test": len(test_df),
                "n_symbols": int(feat_df["Symbol"].nunique()),
            })

            # ---- Baseline ----
            base_model, base_metrics, base_p = train_baseline(train_df, test_df, feature_cols, cfg)
            base_out = {f"base_{k}": v for k, v in base_metrics.items()}

            # ---- TCN ---- (only for classification)
            if cfg["task"] == "classification":
                lookback = int(cfg["deep"]["lookback"])
                Xtr, ytr, meta_tr, Xte, yte, meta_te = make_seq_train_test(train_df, test_df, feature_cols, lookback)
                tcn_model, tcn_metrics, tcn_p = train_tcn(Xtr, ytr, Xte, yte, cfg)

                tcn_thr = float(tcn_metrics.get("best_threshold", 0.5))
                tcn_strat = strat_metrics_from_probs(meta_te, tcn_p, tcn_thr, int(cfg["horizon_days"]))
                tcn_out = {f"tcn_{k}": v for k, v in {**tcn_metrics, **tcn_strat}.items()}
            else:
                tcn_out = {"tcn_skipped": 1.0}

            mlflow.log_metrics({**base_out, **tcn_out})
            results.append({
                "train_end": train_end.date(),
                "test_end": test_end.date(),
                **base_out,
                **tcn_out,
            })

    if not results:
        raise RuntimeError("No walk-forward splits produced. Reduce train_days/test_days or increase data history.")

    out = pd.DataFrame(results)
    Path("artifacts").mkdir(exist_ok=True)
    out.to_csv("artifacts/walk_forward_results.csv", index=False)
    print(out.tail())
    print("Saved:", "artifacts/walk_forward_results.csv")


if __name__ == "__main__":
    run()