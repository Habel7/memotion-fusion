"""Is the task learnable? Pipeline sanity checks and label-reliability analysis.

Three questions:
1. Are features and rows aligned? (CLIP should match each meme image to its own text.)
2. Do the frozen features predict *any* Memotion label? (cross-validated linear probes)
3. Do near-identical memes receive the same label? (annotation consistency)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def retrieval_check(img: np.ndarray, txt: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    """Image-to-text retrieval among n memes. Chance top-1 = 1/n."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(img), size=min(n, len(img)), replace=False)
    sims = img[idx] @ txt[idx].T
    ranks = (-sims).argsort(1)
    hit1 = (ranks[:, 0] == np.arange(len(idx))).mean()
    hit5 = (ranks[:, :5] == np.arange(len(idx))[:, None]).any(1).mean()
    own = (img * txt).sum(1)
    shuffled = (img * txt[rng.permutation(len(txt))]).sum(1)
    return {"n": int(len(idx)), "top1": float(hit1), "top5": float(hit5), "chance_top1": 1 / len(idx),
            "cos_own_pair": float(own.mean()), "cos_shuffled_pair": float(shuffled.mean())}


def probe_targets(df: pd.DataFrame) -> dict:
    """Binary targets: (row mask, positive label)."""
    s5 = df["sentiment_5"].values
    allrows = np.ones(len(df), bool)
    return {
        "Sentiment: negative vs rest": (allrows, np.isin(s5, ["negative", "very_negative"])),
        "Sentiment: positive vs rest": (allrows, np.isin(s5, ["positive", "very_positive"])),
        "Sentiment: very pos. vs very neg.": (np.isin(s5, ["very_positive", "very_negative"]), s5 == "very_positive"),
        "Motivational": (allrows, (df["motivational"] == "motivational").values),
        "Offensive (very/hateful)": (allrows, df["offensive"].isin(["very_offensive", "hateful_offensive"]).values),
        "Humour: not funny": (allrows, (df["humour"] == "not_funny").values),
        "Sarcasm: not sarcastic": (allrows, (df["sarcasm"] == "not_sarcastic").values),
    }


def linear_probe_auc(features: dict, df: pd.DataFrame, n_repeats: int = 2, folds: int = 5) -> pd.DataFrame:
    """Repeated stratified k-fold ROC-AUC of a logistic-regression probe. 0.5 = no signal."""
    rows = []
    for target, (mask, y) in probe_targets(df).items():
        for fname, X in features.items():
            aucs = []
            for r in range(n_repeats):
                skf = StratifiedKFold(folds, shuffle=True, random_state=r)
                clf = make_pipeline(StandardScaler(),
                                    LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))
                p = cross_val_predict(clf, X[mask], y[mask], cv=skf, method="predict_proba")[:, 1]
                aucs.append(roc_auc_score(y[mask], p))
            rows.append({"target": target, "features": fname, "n": int(mask.sum()),
                         "pos_rate": float(y[mask].mean()), "auc_mean": float(np.mean(aucs)),
                         "auc_std": float(np.std(aucs))})
    return pd.DataFrame(rows)


def label_agreement(img: np.ndarray, df: pd.DataFrame, thresholds=(0.98, 0.95, 0.90)) -> pd.DataFrame:
    """Share of near-identical meme pairs that carry the same 3-class label."""
    lab = df["sentiment"].values
    p = df["sentiment"].value_counts(normalize=True).values
    chance = float((p ** 2).sum())
    S = img @ img.T
    np.fill_diagonal(S, -1)
    rows = []
    for t in thresholds:
        i, j = np.where(np.triu(S >= t, 1))
        rows.append({"pair_type": f"Image cosine ≥ {t}", "n_pairs": int(len(i)),
                     "same_label_pct": float((lab[i] == lab[j]).mean() * 100) if len(i) else np.nan})
    real = df[df["text_source"] != "placeholder"]
    same = pairs = 0
    for _, g in real.groupby("text"):
        l = g["sentiment"].values
        for a in range(len(l)):
            for b in range(a + 1, len(l)):
                pairs += 1
                same += l[a] == l[b]
    rows.append({"pair_type": "Identical text", "n_pairs": pairs,
                 "same_label_pct": same / pairs * 100 if pairs else np.nan})
    out = pd.DataFrame(rows)
    out["chance_pct"] = chance * 100
    return out


def plot_reliability(probe: pd.DataFrame, agree: pd.DataFrame, out):
    import matplotlib.pyplot as plt

    from .plots import SERIES, TEXT, MUTED, _save

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.7, 1]})
    ax = axes[0]
    targets = list(dict.fromkeys(probe["target"]))
    feats = list(dict.fromkeys(probe["features"]))
    x = np.arange(len(targets))
    w = 0.8 / len(feats)
    for k, f in enumerate(feats):
        sub = probe[probe["features"] == f].set_index("target").loc[targets]
        ax.bar(x + k * w - 0.4 + w / 2, sub["auc_mean"], w * 0.92, yerr=sub["auc_std"], color=SERIES[k], label=f,
               error_kw={"ecolor": TEXT, "elinewidth": 1, "capsize": 2})
    ax.axhline(0.5, color=SERIES[7], lw=1.5, ls="--", label="chance (AUC 0.5)")
    import textwrap
    ax.set_xticks(x, ["\n".join(textwrap.wrap(t, 14)) for t in targets], fontsize=8)
    ax.set_ylim(0.4, 0.65)
    ax.set_ylabel("Cross-validated ROC-AUC")
    ax.set_title("(a) Linear probes: can frozen features predict each label?")
    ax.legend(ncol=5, fontsize=8, loc="upper left")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    y = np.arange(len(agree))
    ax.barh(y, agree["same_label_pct"], color=SERIES[0], height=0.6)
    for yi, (v, n) in enumerate(zip(agree["same_label_pct"], agree["n_pairs"])):
        ax.text(v + 1, yi, f"{v:.0f}%  (n={n})", va="center", fontsize=8, color=TEXT)
    ax.axvline(agree["chance_pct"].iloc[0], color=SERIES[7], lw=1.5, ls="--")
    ax.text(agree["chance_pct"].iloc[0] + 1, -0.55, f"chance agreement {agree['chance_pct'].iloc[0]:.0f}%",
            fontsize=8, color=MUTED, va="bottom")
    ax.set_ylim(-0.7, len(agree) - 0.5)
    ax.set_yticks(y, agree["pair_type"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Pairs with the same 3-class sentiment label (%)")
    ax.set_title("(b) Label agreement for near-identical memes")
    ax.grid(axis="y", visible=False)
    return _save(fig, out)
