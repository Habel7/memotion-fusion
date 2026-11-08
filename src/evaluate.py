"""Metrics, statistical tests, subgroup (bias) analysis and leakage checks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)

from .models import safe_log_loss


def compute_metrics(y, proba, classes) -> dict:
    pred = proba.argmax(1)
    k = len(classes)
    m = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "weighted_f1": f1_score(y, pred, average="weighted"),
        "cohen_kappa": cohen_kappa_score(y, pred),
        "log_loss": safe_log_loss(y, proba, k),
    }
    try:
        m["macro_auc"] = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=list(range(k)))
    except ValueError:
        m["macro_auc"] = np.nan
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=list(range(k)), zero_division=0)
    for i, c in enumerate(classes):
        m[f"precision_{c}"], m[f"recall_{c}"], m[f"f1_{c}"] = p[i], r[i], f[i]
    return {key: float(v) for key, v in m.items()}


def summarise_runs(runs: dict[str, list[dict]], split: str = "test") -> pd.DataFrame:
    """runs: model -> list of per-seed metric dicts. Returns mean and std per metric."""
    rows = []
    for name, seed_metrics in runs.items():
        df = pd.DataFrame([s[split] for s in seed_metrics])
        row = {"model": name, "n_seeds": len(df)}
        for col in df.columns:
            row[f"{col}_mean"] = df[col].mean()
            row[f"{col}_std"] = df[col].std(ddof=1) if len(df) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def bootstrap_ci(y, pred, n_boot: int = 1000, seed: int = 0, metric=None) -> tuple[float, float, float]:
    metric = metric or (lambda a, b: f1_score(a, b, average="macro"))
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = [metric(y[idx], pred[idx]) for idx in (rng.integers(0, n, n) for _ in range(n_boot))]
    return float(metric(y, pred)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def mcnemar(y, pred_a, pred_b) -> dict:
    """Exact McNemar test on paired correctness of two classifiers."""
    a_ok, b_ok = pred_a == y, pred_b == y
    b = int((a_ok & ~b_ok).sum())  # A right, B wrong
    c = int((~a_ok & b_ok).sum())  # A wrong, B right
    p = binomtest(b, b + c, 0.5).pvalue if (b + c) > 0 else 1.0
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "p_value": float(p)}


def subgroup_table(df_test: pd.DataFrame, preds: dict[str, np.ndarray], group_cols: dict) -> pd.DataFrame:
    """Accuracy and macro-F1 of each model inside each subgroup of the test set."""
    rows = []
    y = df_test["label"].values
    for col, order in group_cols.items():
        if col not in df_test.columns:
            continue
        for level in order:
            mask = (df_test[col] == level).values
            if mask.sum() == 0:
                continue
            row = {"attribute": col, "level": level, "n": int(mask.sum()),
                   "pct_negative": float((y[mask] == 0).mean() * 100)}
            for name, p in preds.items():
                row[f"{name}_acc"] = float((p[mask] == y[mask]).mean())
                row[f"{name}_macro_f1"] = float(f1_score(y[mask], p[mask], average="macro",
                                                         labels=[0, 1, 2], zero_division=0))
            rows.append(row)
    return pd.DataFrame(rows)


def near_duplicates(train_emb, test_emb, y_train, y_test, threshold: float):
    """Find test memes whose CLIP image embedding nearly matches a training meme.

    These are re-posted templates or identical images. High overlap can inflate
    test scores (leakage); label disagreement between near-duplicates measures
    annotation subjectivity.
    """
    sims = test_emb @ train_emb.T
    nn_idx = sims.argmax(1)
    nn_sim = sims.max(1)
    dup = nn_sim >= threshold
    return {
        "mask": dup,
        "nn_idx": nn_idx,
        "nn_sim": nn_sim,
        "n_test_near_dup": int(dup.sum()),
        "pct_test_near_dup": float(dup.mean() * 100),
        "label_agreement_pct": float((y_train[nn_idx[dup]] == y_test[dup]).mean() * 100) if dup.any() else np.nan,
    }


def confusion(y, pred, k=3, normalize=None):
    return confusion_matrix(y, pred, labels=list(range(k)), normalize=normalize)


def uncertainty_report(y, proba, uncertainty, coverages=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5)) -> dict:
    """How useful is the uncertainty? (selective prediction / risk-coverage)

    * error_auroc: ROC-AUC of uncertainty for flagging wrong predictions (0.5 = useless)
    * at each coverage c, keep the c most confident memes and report accuracy / macro-F1
      (the rest would be deferred to a human reviewer)
    """
    pred = proba.argmax(1)
    wrong = (pred != y).astype(int)
    try:
        err_auc = roc_auc_score(wrong, uncertainty) if 0 < wrong.sum() < len(wrong) else np.nan
    except ValueError:
        err_auc = np.nan
    order = np.argsort(uncertainty)  # most confident first
    rows = []
    for c in coverages:
        k = max(1, int(round(c * len(y))))
        keep = order[:k]
        rows.append({"coverage": c, "n": k,
                     "accuracy": float((pred[keep] == y[keep]).mean()),
                     "macro_f1": float(f1_score(y[keep], pred[keep], average="macro", labels=[0, 1, 2], zero_division=0))})
    return {"error_auroc": float(err_auc), "curve": pd.DataFrame(rows)}
