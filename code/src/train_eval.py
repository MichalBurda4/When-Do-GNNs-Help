"""Shared training / evaluation helpers."""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, bundle, mask: torch.Tensor) -> dict:
    model.eval()
    logits = model(bundle.x, bundle.edge_index)
    pred = logits[mask].argmax(dim=-1).cpu().numpy()
    y_true = bundle.y[mask].cpu().numpy()
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }


def train_torch_model(
    model,
    bundle,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> dict:
    set_seed(seed)
    device = torch.device("cpu")
    model = model.to(device)
    bundle_x = bundle.x.to(device)
    edge_index = bundle.edge_index.to(device)
    y = bundle.y.to(device)
    train_mask = bundle.train_mask.to(device)
    val_mask = bundle.val_mask.to(device)
    test_mask = bundle.test_mask.to(device)

    # lightweight stand-in object with tensors on device
    class _B:
        pass

    b = _B()
    b.x, b.edge_index, b.y = bundle_x, edge_index, y
    b.train_mask, b.val_mask, b.test_mask = train_mask, val_mask, test_mask

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = -1.0
    best_state = None

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(b.x, b.edge_index)
        loss = F.cross_entropy(logits[b.train_mask], b.y[b.train_mask])
        loss.backward()
        opt.step()

        val = evaluate(model, b, b.val_mask)["accuracy"]
        if val >= best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    train_m = evaluate(model, b, b.train_mask)
    val_m = evaluate(model, b, b.val_mask)
    test_m = evaluate(model, b, b.test_mask)
    return {
        "train_accuracy": train_m["accuracy"],
        "val_accuracy": val_m["accuracy"],
        "test_accuracy": test_m["accuracy"],
        "test_macro_f1": test_m["macro_f1"],
        "best_val_accuracy": best_val,
    }


def train_xgboost(bundle, *, seed: int, n_estimators: int = 200, max_depth: int = 6) -> dict:
    from xgboost import XGBClassifier

    set_seed(seed)
    x = bundle.x.numpy()
    y = bundle.y.numpy()
    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=bundle.num_classes,
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=2,
    )
    clf.fit(x[bundle.train_mask.numpy()], y[bundle.train_mask.numpy()])

    def _scores(mask: torch.Tensor) -> dict:
        pred = clf.predict(x[mask.numpy()])
        y_true = y[mask.numpy()]
        return {
            "accuracy": float(accuracy_score(y_true, pred)),
            "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        }

    train_m = _scores(bundle.train_mask)
    val_m = _scores(bundle.val_mask)
    test_m = _scores(bundle.test_mask)
    return {
        "train_accuracy": train_m["accuracy"],
        "val_accuracy": val_m["accuracy"],
        "test_accuracy": test_m["accuracy"],
        "test_macro_f1": test_m["macro_f1"],
        "best_val_accuracy": val_m["accuracy"],
    }
