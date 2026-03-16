# src/train_deep.py
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, log_loss

from src.models import TCN

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def train_tcn(Xtr, ytr, Xte, yte, cfg):
    """
    Returns:
      model, metrics, p_te
    Safe against single-class test windows (AUC undefined, logloss needs labels=[0,1]).
    """
    if Xtr.shape[0] == 0 or Xte.shape[0] == 0:
        return None, {"tcn_error": 1.0}, np.zeros(len(yte), dtype=float)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ✅ Standardize features (fit on TRAIN only) — important for deep models
    mu = Xtr.mean(axis=(0, 1), keepdims=True)
    sd = Xtr.std(axis=(0, 1), keepdims=True) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)

    yte_np = yte.astype(int)

    train_loader = DataLoader(
        TensorDataset(Xtr_t, ytr_t),
        batch_size=int(cfg["deep"]["batch_size"]),
        shuffle=True
    )

    model = TCN(
        n_features=Xtr.shape[-1],
        hidden=int(cfg["deep"].get("hidden", 64)),
        kernel=int(cfg["deep"].get("kernel", 3)),
        dropout=float(cfg["deep"].get("dropout", 0.1)),
    ).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["deep"]["lr"]),
        weight_decay=float(cfg["deep"].get("weight_decay", 0.0))
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(int(cfg["deep"]["epochs"])):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    # Train probs for threshold tuning + test probs
    model.eval()
    with torch.no_grad():
        tr_logits = model(Xtr_t.to(device)).cpu().numpy()
        te_logits = model(Xte_t.to(device)).cpu().numpy()

    p_tr = _sigmoid(tr_logits)
    p_te = _sigmoid(te_logits)

    # Threshold tuning on TRAIN using balanced accuracy
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 37):
        score = balanced_accuracy_score(ytr, (p_tr >= t).astype(int))
        if score > best_score:
            best_score, best_t = score, t

    preds = (p_te >= best_t).astype(int)

    # ---- SAFE metrics: handle single-class test windows ----
    classes = np.unique(yte_np)
    single_class_test = 1.0 if len(classes) < 2 else 0.0

    if len(classes) < 2:
        auc = float("nan")  # undefined
    else:
        auc = float(roc_auc_score(yte_np, p_te))

    metrics = {
        "auc": auc,
        "acc": float(accuracy_score(yte_np, preds)),
        "bal_acc": float(balanced_accuracy_score(yte_np, preds)),
        "logloss": float(log_loss(yte_np, p_te, labels=[0, 1])),
        "best_threshold": float(best_t),
        "single_class_test": float(single_class_test),
    }

    return model, metrics, p_te