"""Central configuration for the Memotion multimodal sentiment project.

Every experiment setting lives here so that runs are reproducible and the
report can quote exact values (see README, "Configuration").
"""
from __future__ import annotations

import os
import platform
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Config:
    # ---------------- paths ----------------
    # Point this at the Kaggle download: either the zip file itself
    # (e.g. data/archive.zip) or the folder you unzipped it into.
    data_path: str = "data"
    output_dir: str = "outputs"

    # ---------------- task ----------------
    # Five-point sentiment is merged into three classes (see data.SENTIMENT_MAP).
    classes: tuple = ("negative", "neutral", "positive")
    split_ratios: tuple = (0.70, 0.15, 0.15)  # train / val / test, stratified
    split_seed: int = 42
    max_samples: int | None = None     # e.g. 1500 for a quick dry run; None = full dataset

    # ---------------- feature extraction ----------------
    # "hf" uses the real pretrained encoders from Hugging Face.
    # "mock" is only for quick smoke tests without downloading models.
    encoder_backend: str = "hf"
    clip_model: str = "openai/clip-vit-base-patch32"
    text_model: str = "roberta-base"
    clip_max_tokens: int = 77          # CLIP context length
    roberta_max_tokens: int = 128
    n_aug_views: int = 2               # extra augmented image views per training meme
    feature_batch_size: int = 32
    num_workers: int = 0               # 0 is safest in notebooks on macOS

    # ---------------- classifier heads ----------------
    hidden_dim: int = 256
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 80
    patience: int = 10                 # early stopping on validation macro-F1
    seeds: tuple = (0, 1, 2, 3, 4)     # every trained model is repeated over these seeds
    train_device: str = "cpu"          # small MLP heads train fastest on CPU

    # ---------------- evaluation ----------------
    n_bootstrap: int = 1000
    near_dup_threshold: float = 0.95   # CLIP image cosine similarity
    late_fusion_grid: tuple = tuple(np.round(np.linspace(0, 1, 21), 2))

    extra: dict = field(default_factory=dict)

    # ---------------- helpers ----------------
    @property
    def out(self) -> Path:
        return Path(self.output_dir)

    @property
    def fig_dir(self) -> Path:
        return self.out / "figures"

    @property
    def table_dir(self) -> Path:
        return self.out / "tables"

    @property
    def cache_dir(self) -> Path:
        return self.out / "cache"

    def make_dirs(self) -> None:
        for d in (self.out, self.fig_dir, self.table_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["late_fusion_grid"] = [float(x) for x in self.late_fusion_grid]
        return d


def get_device(prefer: str | None = None) -> str:
    """Pick the fastest available accelerator: Apple MPS, then CUDA, then CPU."""
    import torch

    if prefer:
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def environment_info() -> dict:
    import sklearn
    import torch

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "accelerator": get_device(),
    }
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    return info
