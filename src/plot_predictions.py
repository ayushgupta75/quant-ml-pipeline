import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.ingest import ingest
from src.news import daily_news_signals
from src.features import build_features
from src.split import walk_forward_splits
from src.train_baseline import train_baseline


def plot_one_split(cfg_path="config.yaml", split_idx=0, symbol=None):
    cfg = yaml.safe_load(open(cfg_path))

    # Ingest + features
    prices, _ = ingest(cfg["symbols"], cfg["start_date"], cfg["end_date"])
    feat_df, feature_cols = build_features(prices, cfg)

    # Merge news signals
    news_daily = daily_news_signals(cfg["news"]["rss_feeds"])
    feat_df = feat_df.merge(news_daily, on="Date", how="left")
    feat_df["news_count"] = feat_df["news_count"].fillna(0).astype(float)
    feat_df["news_sent_mean"] = feat_df["news_sent_mean"].fillna(0).astype(float)
    feature_cols = feature_cols + ["news_count", "news_sent_mean"]

    # Walk-forward split selection
    wf = cfg["walk_forward"]
    splits = list(walk_forward_splits(feat_df, wf["train_days"], wf["test_days"], wf["step_days"]))
    train_df, test_df, (train_end, test_end) = splits[split_idx]

    # Train + predict
    model, metrics, p = train_baseline(train_df, test_df, feature_cols, cfg)
    print("Metrics:", metrics)

    # For regression mode: p is predicted future_ret
    plot_df = test_df[["Date", "Symbol", "future_ret"]].copy()
    plot_df["pred_future_ret"] = p

    # Choose symbol to plot
    if symbol is None:
        symbol = cfg["symbols"][0]

    sdf = plot_df[plot_df["Symbol"] == symbol].sort_values("Date")
    if sdf.empty:
        raise ValueError(f"No rows found for symbol={symbol}. Available: {plot_df['Symbol'].unique()}")

    # Correlation (helpful diagnostic)
    corr = float(np.corrcoef(sdf["pred_future_ret"].values, sdf["future_ret"].values)[0, 1])
    print(f"{symbol} pred-vs-actual correlation: {corr:.4f}")

    # Plot predicted vs actual future returns
    plt.figure()
    plt.plot(sdf["Date"], sdf["pred_future_ret"], label=f"Predicted future_ret ({symbol})")
    plt.plot(sdf["Date"], sdf["future_ret"], label=f"Actual future_ret ({symbol})")
    plt.title(f"{symbol}: Predicted vs Actual future_ret (test window {train_end.date()} → {test_end.date()})")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_one_split()