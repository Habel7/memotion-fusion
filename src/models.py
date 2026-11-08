"""Classifier heads and fusion strategies on top of frozen features.

Early fusion: modality features are combined *before* the classifier
             (concatenation, optionally with multiplicative/difference
             interaction terms), one MLP learns from the joint vector.
Late fusion: one MLP per modality, their class probabilities are combined
             *after* prediction with a weight tuned on the validation set.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.preprocessing import StandardScaler

from .config import set_seed


# --------------------------------------------------------------------------
# Feature preparation
# --------------------------------------------------------------------------
class PartScaler:
    """Standardise each modality separately using training statistics only."""

    def __init__(self):
        self.scalers = []

    def fit(self, train_parts):
        # train_parts: list of arrays (n, V, d) - fit on the original view (V=0)
        self.scalers = [StandardScaler().fit(p[:, 0, :]) for p in train_parts]
        return self

    def transform(self, parts):
        out = []
        for s, p in zip(self.scalers, parts):
            if p.ndim == 3:
                n, v, d = p.shape
                out.append(s.transform(p.reshape(n * v, d)).reshape(n, v, d).astype(np.float32))
            else:
                out.append(s.transform(p).astype(np.float32))
        return out


def combine(parts, mode: str = "concat"):
    """Join modality vectors. parts are 2-D tensors/arrays of shape (n, d_k)."""
    lib = torch if isinstance(parts[0], torch.Tensor) else np
    cat = (lambda xs: torch.cat(xs, dim=1)) if lib is torch else (lambda xs: np.concatenate(xs, axis=1))
    if mode in ("concat", "gated") or len(parts) == 1:
        return cat(list(parts))
    if mode == "interact":
        a, b = parts
        return cat([a, b, a * b, lib.abs(a - b)])
    raise ValueError(mode)


# --------------------------------------------------------------------------
# MLP head
# --------------------------------------------------------------------------
class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int, dropout: float):
        super().__init__()
        self.body = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, x, return_hidden: bool = False):
        h = self.body(x)
        return (self.out(h), h) if return_hidden else self.out(h)


class GatedFusionHead(nn.Module):
    """Attention-gated fusion (course weeks 4-5: attention-based fusion).

    Each modality is projected to a common hidden space; a small attention
    network scores every modality *per meme* and the softmax weights decide how
    much each modality contributes before the classifier:
        h_m = GELU(W_m x_m),  a = softmax_m(v . tanh(U h_m)),  z = sum_m a_m h_m
    Unlike late fusion (one global weight w) the weights adapt to each input,
    and unlike plain concatenation they are directly interpretable.
    """

    def __init__(self, dims, hidden: int, n_classes: int, dropout: float):
        super().__init__()
        self.dims = list(dims)
        self.proj = nn.ModuleList([nn.Sequential(nn.Dropout(0.1), nn.Linear(d, hidden), nn.GELU())
                                   for d in self.dims])
        self.att = nn.Sequential(nn.Linear(hidden, 64), nn.Tanh(), nn.Linear(64, 1))
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, n_classes)
        self.last_weights = None

    def forward(self, x, return_hidden: bool = False):
        parts = torch.split(x, self.dims, dim=1)
        h = torch.stack([p(xm) for p, xm in zip(self.proj, parts)], dim=1)   # (n, M, hidden)
        a = torch.softmax(self.att(h).squeeze(-1), dim=1)                    # (n, M)
        self.last_weights = a.detach()
        z = self.drop((a.unsqueeze(-1) * h).sum(1))
        return (self.out(z), z) if return_hidden else self.out(z)


@dataclass
class TrainResult:
    model: nn.Module
    history: dict
    best_epoch: int
    train_seconds: float
    n_params: int
    extra: dict = field(default_factory=dict)


def class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    return (len(y) / (n_classes * counts)).astype(np.float32)


def _eval_epoch(model, X, y, loss_fn):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss = loss_fn(logits, y).item()
        pred = logits.argmax(1).cpu().numpy()
    return loss, f1_score(y.cpu().numpy(), pred, average="macro")


def train_mlp(train_parts, y_train, val_parts, y_val, cfg, seed: int, mode: str = "concat",
              weighted: bool = True, use_aug: bool = True) -> TrainResult:
    """Train one MLP head with early stopping on validation macro-F1.

    train_parts: list of arrays (n, V, d) - V views per sample (V=1 if no augmentation)
    val_parts:   list of arrays (n, d)
    Each epoch draws one random view per sample and modality, so augmented
    image views act as cheap, cached data augmentation.
    """
    set_seed(seed)
    dev = cfg.train_device
    n_classes = len(cfg.classes)
    Xtr = [torch.tensor(p if use_aug else p[:, :1], device=dev) for p in train_parts]
    Xva = combine([torch.tensor(p, device=dev) for p in val_parts], mode)
    ytr = torch.tensor(y_train, dtype=torch.long, device=dev)
    yva = torch.tensor(y_val, dtype=torch.long, device=dev)

    in_dim = combine([p[:2, 0] for p in Xtr], mode).shape[1]
    if mode == "gated":
        model = GatedFusionHead([p.shape[2] for p in Xtr], cfg.hidden_dim, n_classes, cfg.dropout).to(dev)
    else:
        model = MLPHead(in_dim, cfg.hidden_dim, n_classes, cfg.dropout).to(dev)
    w = torch.tensor(class_weights(y_train, n_classes) if weighted else np.ones(n_classes, np.float32), device=dev)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    eval_loss = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    hist = {k: [] for k in ("train_loss", "val_loss", "train_f1", "val_f1")}
    best, best_state, best_epoch, wait = -1.0, None, 0, 0
    n = len(ytr)
    t0 = time.perf_counter()
    for epoch in range(cfg.max_epochs):
        model.train()
        # one random view per sample for this epoch
        views = [torch.randint(p.shape[1], (n,), generator=gen).to(dev) for p in Xtr]
        X_epoch = combine([p[torch.arange(n, device=dev), v] for p, v in zip(Xtr, views)], mode)
        perm = torch.randperm(n, generator=gen).to(dev)
        for i in range(0, n, cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X_epoch[b]), ytr[b])
            loss.backward()
            opt.step()

        Xtr0 = combine([p[:, 0] for p in Xtr], mode)  # original views for monitoring
        tr_loss, tr_f1 = _eval_epoch(model, Xtr0, ytr, eval_loss)
        va_loss, va_f1 = _eval_epoch(model, Xva, yva, eval_loss)
        for k, v in zip(hist, (tr_loss, va_loss, tr_f1, va_f1)):
            hist[k].append(v)
        if va_f1 > best + 1e-4:
            best, best_state, best_epoch, wait = va_f1, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    secs = time.perf_counter() - t0
    model.load_state_dict(best_state)
    model.eval()
    return TrainResult(model, hist, best_epoch, secs, sum(p.numel() for p in model.parameters()))


@torch.no_grad()
def predict_proba(model, parts, cfg, mode: str = "concat") -> np.ndarray:
    X = combine([torch.tensor(p, device=cfg.train_device) for p in parts], mode)
    return torch.softmax(model(X), dim=1).cpu().numpy()


@torch.no_grad()
def hidden_repr(model, parts, cfg, mode: str = "concat") -> np.ndarray:
    X = combine([torch.tensor(p, device=cfg.train_device) for p in parts], mode)
    return model(X, return_hidden=True)[1].cpu().numpy()


@torch.no_grad()
def attention_weights(model, parts, cfg) -> np.ndarray:
    """Per-meme modality weights of a GatedFusionHead (columns follow `parts`)."""
    X = combine([torch.tensor(p, device=cfg.train_device) for p in parts], "gated")
    model.eval()
    model(X)
    return model.last_weights.cpu().numpy()


@torch.no_grad()
def mc_dropout_proba(model, parts, cfg, mode: str = "concat", n_samples: int = 30, seed: int = 0):
    """Monte Carlo dropout (course week 5: uncertainty-aware fusion).

    Dropout stays active at test time; averaging n_samples stochastic passes
    approximates Bayesian model averaging (Gal & Ghahramani, 2016).
    Returns the mean probabilities and the predictive entropy per meme.
    """
    torch.manual_seed(seed)
    X = combine([torch.tensor(p, device=cfg.train_device) for p in parts], mode)
    model.train()  # keeps dropout on
    probs = torch.stack([torch.softmax(model(X), dim=1) for _ in range(n_samples)]).mean(0)
    model.eval()
    p = probs.cpu().numpy()
    entropy = -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)
    return p, entropy


# --------------------------------------------------------------------------
# Linear probe (classical baseline)
# --------------------------------------------------------------------------
def train_logreg(X_train, y_train, X_val, y_val, Cs=(0.01, 0.1, 1.0, 10.0)):
    best = None
    t0 = time.perf_counter()
    for C in Cs:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
        clf.fit(X_train, y_train)
        f1 = f1_score(y_val, clf.predict(X_val), average="macro")
        if best is None or f1 > best[0]:
            best = (f1, C, clf)
    return best[2], {"C": best[1], "val_f1": best[0], "train_seconds": time.perf_counter() - t0}


# --------------------------------------------------------------------------
# Late fusion
# --------------------------------------------------------------------------
def tune_late_fusion(p_a_val, p_b_val, y_val, grid):
    """Weighted average of class probabilities: p = w * p_a + (1 - w) * p_b.

    w is chosen on validation macro-F1 (w=1 -> modality A only, w=0 -> B only).
    Returns best w and the full validation curve for plotting.
    """
    curve = []
    for w in grid:
        p = w * p_a_val + (1 - w) * p_b_val
        curve.append(f1_score(y_val, p.argmax(1), average="macro"))
    best_w = float(grid[int(np.argmax(curve))])
    return best_w, np.array(curve)


def late_fuse(p_a, p_b, w):
    return w * p_a + (1 - w) * p_b


# --------------------------------------------------------------------------
# Zero-shot CLIP
# --------------------------------------------------------------------------
def zero_shot_proba(emb, prompt_emb, temperature: float = 100.0):
    logits = temperature * emb @ prompt_emb.T
    logits = logits - logits.max(1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(1, keepdims=True)


def safe_log_loss(y, p, n_classes):
    return float(log_loss(y, np.clip(p, 1e-7, 1), labels=list(range(n_classes))))
