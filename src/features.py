# src/features.py
import numpy as np
import pandas as pd

def build_features(prices: pd.DataFrame, cfg: dict):
    df = prices.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    g_close = df.groupby("Symbol")["Close"]

    # Returns (skip 1-day here; we compute ret_1d separately)
    for w in cfg["features"]["returns_windows"]:
        if w == 1:
            continue
        df[f"ret_{w}d"] = g_close.transform(lambda s: s.pct_change(periods=w, fill_method=None))

    # 1-day return for volatility + optional modeling
    df["ret_1d"] = g_close.transform(lambda s: s.pct_change(fill_method=None))

    # Rolling volatility of returns
    for w in cfg["features"]["vol_windows"]:
        df[f"vol_{w}d"] = df.groupby("Symbol")["ret_1d"].transform(lambda s: s.rolling(w).std())

    # Moving averages + ratios
    for w in cfg["features"]["ma_windows"]:
        ma = g_close.transform(lambda s: s.rolling(w).mean())
        df[f"close_over_ma_{w}"] = df["Close"] / (ma + 1e-9)

    # RSI
    def rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta.clip(upper=0)).rolling(window).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    df["rsi"] = g_close.transform(lambda s: rsi(s, cfg["features"]["rsi_window"]))

    # Label + future return
    h = int(cfg["horizon_days"])
    future_ret = g_close.transform(lambda s: s.shift(-h) / s - 1.0)
    df["future_ret"] = future_ret

    if cfg["task"] == "classification":
        df["y"] = (future_ret > 0).astype(int)
    else:
        df["y"] = future_ret

    # Base features
    feature_cols = [c for c in df.columns if c.startswith(("ret_", "vol_", "close_over_")) or c == "rsi"]

    # --- Cross-sectional rank features (per Date across Symbols) ---
    rank_base = []
    for col in feature_cols:
        if col.startswith(("ret_", "vol_", "close_over_ma_")) or col == "rsi":
            rank_base.append(col)

    for col in rank_base:
        df[f"rank_{col}"] = df.groupby("Date")[col].rank(pct=True)

    rank_cols = [f"rank_{c}" for c in rank_base]

    # Lag rank features by 1 day per symbol
    for c in rank_cols:
        df[c] = df.groupby("Symbol")[c].shift(1)

    feature_cols = feature_cols + rank_cols

    # Drop rows with any missing features/labels/returns
    df = df.dropna(subset=feature_cols + ["y", "future_ret"]).reset_index(drop=True)

    return df, feature_cols