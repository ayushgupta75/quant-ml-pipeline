# src/datasets.py
import numpy as np
import pandas as pd

def make_seq_train_test(train_df: pd.DataFrame,
                        test_df: pd.DataFrame,
                        feature_cols: list[str],
                        lookback: int):
    """
    Returns:
      Xtr, ytr, meta_tr
      Xte, yte, meta_te  (test sequences may include lookback context from end of train)
    """
    # Ensure sort
    train_df = train_df.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    test_df = test_df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    Xtr_list, ytr_list, meta_tr = [], [], []
    Xte_list, yte_list, meta_te = [], [], []

    for sym in sorted(set(train_df["Symbol"]).union(set(test_df["Symbol"]))):
        tr = train_df[train_df["Symbol"] == sym].sort_values("Date").reset_index(drop=True)
        te = test_df[test_df["Symbol"] == sym].sort_values("Date").reset_index(drop=True)

        # --- Train sequences (purely within train) ---
        if len(tr) > lookback:
            feats_tr = tr[feature_cols].values.astype(np.float32)
            y_tr = tr["y"].values.astype(np.int64)
            for i in range(lookback, len(tr)):
                Xtr_list.append(feats_tr[i - lookback:i])
                ytr_list.append(y_tr[i])
                meta_tr.append({
                    "Date": tr.loc[i, "Date"],
                    "Symbol": sym,
                    "future_ret": float(tr.loc[i, "future_ret"]),
                })

        # --- Test sequences (allow lookback context from end of train) ---
        if len(te) == 0:
            continue

        ctx = tr.tail(lookback) if len(tr) >= lookback else tr
        comb = pd.concat([ctx, te], ignore_index=True).sort_values("Date").reset_index(drop=True)

        feats = comb[feature_cols].values.astype(np.float32)
        y_all = comb["y"].values.astype(np.int64)

        # Only evaluate on rows that belong to test window
        test_dates = set(te["Date"].tolist())

        for i in range(lookback, len(comb)):
            d = comb.loc[i, "Date"]
            if d not in test_dates:
                continue
            Xte_list.append(feats[i - lookback:i])
            yte_list.append(y_all[i])
            meta_te.append({
                "Date": d,
                "Symbol": sym,
                "future_ret": float(comb.loc[i, "future_ret"]),
            })

    Xtr = np.stack(Xtr_list) if Xtr_list else np.empty((0, lookback, len(feature_cols)), dtype=np.float32)
    ytr = np.array(ytr_list, dtype=np.int64)
    Xte = np.stack(Xte_list) if Xte_list else np.empty((0, lookback, len(feature_cols)), dtype=np.float32)
    yte = np.array(yte_list, dtype=np.int64)

    return Xtr, ytr, pd.DataFrame(meta_tr), Xte, yte, pd.DataFrame(meta_te)