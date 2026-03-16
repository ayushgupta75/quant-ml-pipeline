import pandas as pd
from datetime import timedelta

def walk_forward_splits(df: pd.DataFrame, train_days: int, test_days: int, step_days: int):
    df = df.sort_values("Date")
    start = df["Date"].min()
    end = df["Date"].max()

    train_delta = timedelta(days=train_days)
    test_delta = timedelta(days=test_days)
    step_delta = timedelta(days=step_days)

    t0 = start + train_delta
    while t0 + test_delta <= end:
        train_end = t0
        test_end = t0 + test_delta

        train_mask = (df["Date"] < train_end) & (df["Date"] >= train_end - train_delta)
        test_mask = (df["Date"] >= train_end) & (df["Date"] < test_end)

        yield df[train_mask].copy(), df[test_mask].copy(), (train_end, test_end)
        t0 = t0 + step_delta
