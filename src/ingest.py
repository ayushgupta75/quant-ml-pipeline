import pandas as pd
import yfinance as yf
from pathlib import Path

def fetch_ohlcv(symbol: str, start: str, end: str | None) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)

    # ✅ If yfinance returns MultiIndex columns, flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]  # keep price field names only

    # ✅ Drop duplicate column names (this is the real fix)
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]

    df = df.reset_index()
    df["Symbol"] = symbol
    return df

def fetch_fundamentals(symbol: str) -> dict:
    t = yf.Ticker(symbol)
    info = t.info or {}
    # Keep only stable, small fields (avoid huge noisy blobs)
    keys = ["sector", "industry", "marketCap", "trailingPE", "forwardPE", "dividendYield"]
    return {k: info.get(k) for k in keys}

def save_raw(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

def ingest(symbols: list[str], start: str, end: str | None, out_dir: str = "data/raw"):
    out = Path(out_dir)
    fundamentals_rows = []

    all_df = []
    for s in symbols:
        ohlcv = fetch_ohlcv(s, start, end)
        all_df.append(ohlcv)

        f = fetch_fundamentals(s)
        f["Symbol"] = s
        fundamentals_rows.append(f)

    prices = pd.concat(all_df, ignore_index=True)
    fundamentals = pd.DataFrame(fundamentals_rows)

    save_raw(prices, out / "ohlcv.parquet")
    save_raw(fundamentals, out / "fundamentals.parquet")
    return prices, fundamentals
