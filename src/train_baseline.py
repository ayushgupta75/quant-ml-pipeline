# src/train_baseline.py
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    mean_squared_error,
    balanced_accuracy_score,
    log_loss,
)
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_baseline(train_df, test_df, feature_cols, cfg):
    """
    Returns:
      model, metrics, p (probabilities for classification, predictions for regression)
    Safe against single-class test windows.
    """
    Xtr, ytr = train_df[feature_cols].values, train_df["y"].values
    Xte, yte = test_df[feature_cols].values, test_df["y"].values

    if cfg["task"] == "classification":
        model = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            max_iter=400,
            random_state=42
        )
        model.fit(Xtr, ytr)

        # Threshold tuning on TRAIN using balanced accuracy
        ptr = model.predict_proba(Xtr)[:, 1]
        best_t, best_score = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 37):
            score = balanced_accuracy_score(ytr, (ptr >= t).astype(int))
            if score > best_score:
                best_score, best_t = score, t

        # Evaluate on TEST
        p = model.predict_proba(Xte)[:, 1]
        preds = (p >= best_t).astype(int)

        pos_rate = float(np.mean(yte))
        dummy_acc = float(max(pos_rate, 1 - pos_rate))

        # Strategy: long if p >= threshold else flat
        test_rets = test_df["future_ret"].values.astype(float)
        positions = (p >= best_t).astype(int)
        strat_rets = positions * test_rets

        mean_ret = float(np.mean(strat_rets))
        std_ret = float(np.std(strat_rets) + 1e-12)
        ann_factor = float(np.sqrt(252.0 / max(float(cfg["horizon_days"]), 1.0)))
        sharpe = float((mean_ret / std_ret) * ann_factor)

        trade_rate = float(np.mean(positions))
        avg_trade_ret = float(np.mean(strat_rets[positions == 1]) if positions.sum() > 0 else 0.0)
        hit_rate = float(np.mean((strat_rets[positions == 1] > 0).astype(int)) if positions.sum() > 0 else 0.0)

        # SAFE AUC/logloss when only one class in test
        classes = np.unique(yte.astype(int))
        single_class_test = 1.0 if len(classes) < 2 else 0.0
        auc = float("nan") if len(classes) < 2 else float(roc_auc_score(yte, p))

        metrics = {
            "auc": auc,
            "acc": float(accuracy_score(yte, preds)),
            "dummy_acc": dummy_acc,
            "pos_rate": pos_rate,
            "best_threshold": float(best_t),
            "bal_acc": float(balanced_accuracy_score(yte, preds)),
            "logloss": float(log_loss(yte, p, labels=[0, 1])),
            "single_class_test": float(single_class_test),
            # strategy metrics
            "mean_strat_ret": mean_ret,
            "sharpe": sharpe,
            "trade_rate": trade_rate,
            "avg_trade_ret": avg_trade_ret,
            "hit_rate": hit_rate,
        }

    else:
        # Regression baseline
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0))
        ])
        model.fit(Xtr, ytr)
        p = model.predict(Xte)

        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(yte, p))),
        }

    return model, metrics, p