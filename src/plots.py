"""Report-ready figures. Every function saves a PNG (200 dpi) and returns the figure."""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Validated categorical palette (fixed order) and a single-hue sequential ramp
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

FAMILY_COLOURS = {
    "Baseline": "#8a8984",
    "Image only": SERIES[0],
    "Text only": SERIES[1],
    "Early fusion": SERIES[2],
    "Late fusion": SERIES[6],
    "Attention fusion": SERIES[3],
}
CLASS_COLOURS = {"negative": SERIES[0], "neutral": SERIES[1], "positive": SERIES[2]}


def set_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT, "axes.titlecolor": TEXT,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 9, "font.size": 10,
        "lines.linewidth": 2,
    })


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------
# Data exploration
# --------------------------------------------------------------------------
def plot_label_distribution(df, out: Path, classes):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    order5 = ["very_negative", "negative", "neutral", "positive", "very_positive"]
    c5 = df["sentiment_5"].value_counts().reindex(order5).fillna(0)
    axes[0].bar(range(5), c5.values, color=SERIES[0], width=0.65)
    axes[0].set_xticks(range(5), [o.replace("_", "\n") for o in order5])
    axes[0].set_title("(a) Original 5-point labels")
    axes[0].set_ylabel("Number of memes")
    c3 = df["sentiment"].value_counts().reindex(classes)
    bars = axes[1].bar(classes, c3.values, color=[CLASS_COLOURS[c] for c in classes], width=0.6)
    for b, v in zip(bars, c3.values):
        axes[1].text(b.get_x() + b.get_width() / 2, v, f"{v}\n({v / len(df):.1%})",
                     ha="center", va="bottom", fontsize=9, color=TEXT)
    axes[1].set_title("(b) Merged 3-class labels")
    axes[1].set_ylim(0, c3.max() * 1.25)
    for ax in axes:
        ax.grid(axis="x", visible=False)
    return _save(fig, out)


def plot_data_quality(df, out: Path, classes):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    ok = df[df["readable"]]
    for c in classes:
        sub = df[df["sentiment"] == c]
        axes[0].hist(sub["n_words"].clip(upper=80), bins=40, alpha=0.55, label=c, color=CLASS_COLOURS[c])
    axes[0].set_title("(a) Words per meme (clipped at 80)")
    axes[0].set_xlabel("Words in cleaned text"); axes[0].set_ylabel("Number of memes")
    axes[0].legend()

    axes[1].scatter(ok["width"], ok["height"], s=6, alpha=0.35, color=SERIES[0], edgecolors="none")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].axvline(224, color=SERIES[7], lw=1, ls="--"); axes[1].axhline(224, color=SERIES[7], lw=1, ls="--")
    axes[1].set_title("(b) Image resolution (224 px dashed)")
    axes[1].set_xlabel("Width (px, log)"); axes[1].set_ylabel("Height (px, log)")

    src = pd.crosstab(df["text_source"], df["sentiment"], normalize="columns").reindex(
        index=["corrected", "ocr", "placeholder"], columns=classes).fillna(0) * 100
    x = np.arange(len(classes))
    bottom = np.zeros(len(classes))
    for i, s in enumerate(src.index):
        axes[2].bar(x, src.loc[s].values, bottom=bottom, color=SERIES[[2, 3, 7][i]], label=s,
                    width=0.6, edgecolor=SURFACE, linewidth=2)
        bottom += src.loc[s].values
    axes[2].set_xticks(x, classes); axes[2].set_ylabel("% of memes in class")
    axes[2].set_title("(c) Text source used per class"); axes[2].legend(loc="lower right")
    axes[2].grid(axis="x", visible=False)
    return _save(fig, out)


def plot_aux_vs_sentiment(df, out: Path, aux_orders, classes):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, (col, order) in zip(axes, aux_orders.items()):
        tab = pd.crosstab(df[col], df["sentiment"], normalize="index").reindex(index=order, columns=classes) * 100
        bottom = np.zeros(len(order))
        for c in classes:
            ax.bar(range(len(order)), tab[c].values, bottom=bottom, color=CLASS_COLOURS[c], label=c,
                   width=0.65, edgecolor=SURFACE, linewidth=2)
            bottom += tab[c].values
        ax.set_xticks(range(len(order)), [o.replace("_", "\n") for o in order], fontsize=8)
        ax.set_title(col.capitalize()); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("% of memes"); axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    return _save(fig, out)


# --------------------------------------------------------------------------
# Training and evaluation
# --------------------------------------------------------------------------
def plot_learning_curves(histories: dict, out: Path):
    n = len(histories)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 6), squeeze=False)
    for j, (name, h) in enumerate(histories.items()):
        ep = np.arange(1, len(h["train_loss"]) + 1)
        best = int(np.argmax(h["val_f1"])) + 1
        axes[0, j].plot(ep, h["train_loss"], color=SERIES[0], label="train")
        axes[0, j].plot(ep, h["val_loss"], color=SERIES[1], label="validation")
        axes[1, j].plot(ep, h["train_f1"], color=SERIES[0], label="train")
        axes[1, j].plot(ep, h["val_f1"], color=SERIES[1], label="validation")
        for i in range(2):
            axes[i, j].axvline(best, color=MUTED, lw=1, ls=":")
        axes[0, j].set_title(name, fontsize=10)
        axes[1, j].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Weighted CE loss"); axes[1, 0].set_ylabel("Macro-F1")
    axes[0, 0].legend()
    fig.suptitle("Learning curves (seed 0). Dotted line = epoch kept by early stopping",
                 fontsize=11, color=TEXT)
    return _save(fig, out)


def plot_model_comparison(summary: pd.DataFrame, families: dict, out: Path, metric="macro_f1"):
    s = summary.sort_values(f"{metric}_mean")
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(s) + 1.2))
    y = np.arange(len(s))
    cols = [FAMILY_COLOURS[families.get(m, "Baseline")] for m in s.index]
    ax.barh(y, s[f"{metric}_mean"], xerr=s[f"{metric}_std"], color=cols, height=0.62,
            error_kw={"ecolor": TEXT, "elinewidth": 1, "capsize": 3})
    for yi, (m, v) in enumerate(s[f"{metric}_mean"].items()):
        ax.text(v + s[f"{metric}_std"].iloc[yi] + 0.005, yi, f"{v:.3f}", va="center", fontsize=8, color=TEXT)
    ax.set_yticks(y, s.index)
    ax.set_xlabel(f"Test {metric.replace('_', '-')} (mean ± std over seeds)")
    ax.set_xlim(0, min(1.0, (s[f"{metric}_mean"] + s[f"{metric}_std"]).max() + 0.08))
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for f, c in FAMILY_COLOURS.items()
               if f in set(families.get(m, "Baseline") for m in s.index)]
    labels = [f for f in FAMILY_COLOURS if f in set(families.get(m, "Baseline") for m in s.index)]
    ax.legend(handles, labels, loc="lower right")
    return _save(fig, out)


def plot_per_class_f1(summary: pd.DataFrame, models: list, classes, out: Path):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(classes))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        vals = [summary.loc[m, f"f1_{c}_mean"] for c in classes]
        errs = [summary.loc[m, f"f1_{c}_std"] for c in classes]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w * 0.92, yerr=errs, color=SERIES[i], label=m,
               error_kw={"ecolor": TEXT, "elinewidth": 1, "capsize": 2})
    ax.set_xticks(x, classes); ax.set_ylabel("Test F1 (mean ± std)")
    ax.set_title("Per-class F1"); ax.legend(ncol=2, fontsize=8); ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 1)
    return _save(fig, out)


def plot_confusions(cms: dict, classes, out: Path):
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.6))
    axes = np.atleast_1d(axes)
    for ax, (name, cm) in zip(axes, cms.items()):
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.55 else TEXT)
        ax.set_xticks(range(len(classes)), classes); ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("Predicted"); ax.set_title(name, fontsize=10); ax.grid(False)
    axes[0].set_ylabel("True")
    fig.suptitle("Row-normalised confusion matrices on the test set (seed 0)", fontsize=11)
    return _save(fig, out)


def plot_late_fusion_curve(grid, curves: dict, out: Path):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for i, (name, c) in enumerate(curves.items()):
        ax.plot(grid, c, color=SERIES[i], label=name)
        k = int(np.argmax(c))
        ax.scatter([grid[k]], [c[k]], color=SERIES[i], s=40, zorder=3, edgecolors=SURFACE, linewidths=2)
    ax.set_xlabel("Image weight w  (p = w·p_image + (1−w)·p_text)")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Late fusion: choosing the modality weight")
    ax.legend()
    return _save(fig, out)


def plot_subgroups(sub: pd.DataFrame, models: list, out: Path, metric="acc"):
    attrs = sub["attribute"].unique()
    fig, axes = plt.subplots(1, len(attrs), figsize=(4 * len(attrs), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, a in zip(axes, attrs):
        s = sub[sub["attribute"] == a]
        x = np.arange(len(s))
        w = 0.8 / len(models)
        for i, m in enumerate(models):
            ax.bar(x + i * w - 0.4 + w / 2, s[f"{m}_{metric}"], w * 0.92, color=SERIES[i], label=m)
        ax.set_xticks(x, [f"{l.replace('_', ' ')}\n(n={n})" for l, n in zip(s["level"], s["n"])], fontsize=7)
        ax.set_title(a.capitalize()); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(f"Test {'accuracy' if metric == 'acc' else 'macro-F1'}")
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    return _save(fig, out)


def plot_tsne(embs: dict, y, classes, out: Path, seed=0):
    from sklearn.manifold import TSNE

    fig, axes = plt.subplots(1, len(embs), figsize=(5 * len(embs), 4.4))
    axes = np.atleast_1d(axes)
    for ax, (name, X) in zip(axes, embs.items()):
        # method="exact" avoids scikit-learn's OpenMP Barnes-Hut code, which segfaults
        # when PyTorch's OpenMP runtime is already loaded (macOS + Anaconda).
        from sklearn.decomposition import PCA
        Xr = PCA(n_components=min(50, X.shape[1]), random_state=seed).fit_transform(X)
        Z = TSNE(n_components=2, perplexity=30, init="pca", method="exact",
                 random_state=seed).fit_transform(Xr)
        for k, c in enumerate(classes):
            m = y == k
            ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.6, color=CLASS_COLOURS[c], label=c, edgecolors="none")
        ax.set_title(name, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(markerscale=2)
    fig.suptitle("t-SNE of test-set representations, coloured by true sentiment", fontsize=11)
    return _save(fig, out)


def plot_ablation(df_abl: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(df_abl) + 1.2))
    y = np.arange(len(df_abl))
    ax.barh(y, df_abl["macro_f1_mean"], xerr=df_abl["macro_f1_std"], color=SERIES[2], height=0.6,
            error_kw={"ecolor": TEXT, "elinewidth": 1, "capsize": 3})
    for yi, v in enumerate(df_abl["macro_f1_mean"]):
        ax.text(v + df_abl["macro_f1_std"].iloc[yi] + 0.005, yi, f"{v:.3f}", va="center", fontsize=8)
    ax.set_yticks(y, df_abl.index); ax.set_xlabel("Test macro-F1 (mean ± std)")
    ax.set_title("Early-fusion ablations"); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, (df_abl["macro_f1_mean"] + df_abl["macro_f1_std"]).max() + 0.08)
    return _save(fig, out)


def plot_failure_cases(df_cases: pd.DataFrame, load_image, classes, out: Path, n=8):
    rows = df_cases.head(n)
    cols = 4
    r = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(r, cols, figsize=(3.4 * cols, 3.9 * r), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, row) in zip(axes.ravel(), rows.iterrows()):
        try:
            img = load_image(row["image_path"])
            img.thumbnail((300, 300))
            ax.imshow(img)
        except Exception:  # noqa: BLE001
            ax.text(0.5, 0.5, "image unavailable", ha="center")
        txt = textwrap.shorten(row["text"], 70)
        ax.set_title(f"true: {classes[row['label']]} | pred: {classes[row['pred']]}\n"
                     f"conf {row['conf']:.2f}", fontsize=8.5)
        ax.text(0.5, -0.08, "\n".join(textwrap.wrap(txt, 38)), transform=ax.transAxes,
                ha="center", va="top", fontsize=7, color=MUTED)
    fig.suptitle("Most confident errors of the best fusion model", fontsize=11)
    return _save(fig, out)


def plot_risk_coverage(curves: dict, out, metric="accuracy"):
    """Selective prediction: performance on the most-confident fraction of memes."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, (name, c) in enumerate(curves.items()):
        ax.plot(c["coverage"] * 100, c[metric], marker="o", ms=4, color=SERIES[i], label=name)
    ax.invert_xaxis()
    ax.set_xlabel("Coverage: % of test memes the model keeps (rest deferred to a human)")
    ax.set_ylabel(f"Test {'accuracy' if metric == 'accuracy' else 'macro-F1'} on kept memes")
    ax.set_title("MC-dropout uncertainty: does deferring uncertain memes help?")
    ax.legend(fontsize=8)
    return _save(fig, out)


def plot_attention_weights(weights_by_class: dict, out):
    """Distribution of the per-meme image weight learned by attention-gated fusion."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    names = list(weights_by_class)
    data = [weights_by_class[k] for k in names]
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.55, showfliers=False)
    for patch, k in zip(bp["boxes"], names):
        patch.set_facecolor(CLASS_COLOURS.get(k, SERIES[0])); patch.set_alpha(0.75); patch.set_edgecolor(TEXT)
    for med in bp["medians"]:
        med.set_color(TEXT)
    ax.set_yticks(range(1, len(names) + 1), names)
    ax.axvline(0.5, color=MUTED, lw=1, ls="--")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Attention weight on the image (1 - weight on the text)")
    ax.set_title("Attention-gated fusion: per-meme modality weights (test, seed 0)")
    ax.grid(axis="y", visible=False)
    return _save(fig, out)
