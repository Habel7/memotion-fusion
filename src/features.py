"""Frozen-encoder feature extraction (CLIP ViT-B/32 image + text, RoBERTa text).

Features are computed once and cached to disk, so classifier experiments
can be rerun in seconds without touching the encoders again.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm  # plain text bars: no notebook widgets

from .data import load_image

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# --------------------------------------------------------------------------
# Image pipeline
# --------------------------------------------------------------------------
def clip_image_transform(augment: bool):
    """224x224 CLIP preprocessing; augmentation is for training views only.

    No horizontal flips: they would mirror the caption burned into the meme.
    """
    from torchvision import transforms as T

    norm = [T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)]
    if augment:
        return T.Compose([
            T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1),
                                interpolation=T.InterpolationMode.BICUBIC),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            *norm,
        ])
    return T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        *norm,
    ])


class ImageDataset(Dataset):
    def __init__(self, paths, augment: bool):
        self.paths = list(paths)
        self.tf = clip_image_transform(augment)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(load_image(self.paths[i]))


# --------------------------------------------------------------------------
# Encoders
# --------------------------------------------------------------------------
def _pooled(out):
    """Return the pooled tensor whatever the transformers version returns."""
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    return out[1]


class ClipEncoder:
    def __init__(self, model_name: str, device: str):
        from transformers import CLIPModel, CLIPTokenizer

        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.tok = CLIPTokenizer.from_pretrained(model_name)
        self.n_params = sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.model.vision_model(pixel_values=pixel_values.to(self.device))
        emb = self.model.visual_projection(_pooled(out))
        return torch.nn.functional.normalize(emb, dim=-1)

    @torch.no_grad()
    def texts(self, texts: list[str], max_tokens: int = 77) -> torch.Tensor:
        enc = self.tok(texts, padding=True, truncation=True, max_length=max_tokens, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model.text_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        emb = self.model.text_projection(_pooled(out))
        return torch.nn.functional.normalize(emb, dim=-1)


class RobertaEncoder:
    def __init__(self, model_name: str, device: str):
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.n_params = sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def texts(self, texts: list[str], max_tokens: int = 128) -> torch.Tensor:
        enc = self.tok(texts, padding=True, truncation=True, max_length=max_tokens, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        h = self.model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        return (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)  # mean pooling


class MockEncoder:
    """Deterministic stand-in used only for smoke tests (no downloads).

    Image features come from colour statistics of the picture, text features
    from hashed words, so the pipeline sees *some* real signal.
    """

    def __init__(self, dim: int = 512, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.proj = torch.tensor(rng.normal(size=(3 * 4 * 4, dim)), dtype=torch.float32)
        self.dim = dim
        self.n_params = 0
        self.device = "cpu"

    @torch.no_grad()
    def images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pooled = torch.nn.functional.adaptive_avg_pool2d(pixel_values, 4).flatten(1)
        return torch.nn.functional.normalize(pooled @ self.proj, dim=-1)

    @torch.no_grad()
    def texts(self, texts: list[str], max_tokens: int = 77) -> torch.Tensor:
        out = torch.zeros(len(texts), self.dim)
        for i, t in enumerate(texts):
            for w in t.split()[:max_tokens]:
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return torch.nn.functional.normalize(out + 1e-6, dim=-1)


# --------------------------------------------------------------------------
# Extraction with caching
# --------------------------------------------------------------------------
def _cache(path: Path, fn):
    if path.exists():
        data = np.load(path)
        return data["x"], float(data["seconds"])
    t0 = time.perf_counter()
    x = fn()
    secs = time.perf_counter() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x, seconds=secs)
    return x, secs


def encode_images(encoder, paths, augment: bool, batch_size: int, num_workers: int, seed: int) -> np.ndarray:
    ds = ImageDataset(paths, augment)
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, generator=g)
    feats = []
    for px in tqdm(dl, desc=f"images{' (aug)' if augment else ''}", leave=False):
        feats.append(encoder.images(px).float().cpu().numpy())
    return np.concatenate(feats).astype(np.float32)


def encode_texts(encoder, texts, batch_size: int, max_tokens: int) -> np.ndarray:
    feats = []
    texts = list(texts)
    for i in tqdm(range(0, len(texts), batch_size), desc="texts", leave=False):
        feats.append(encoder.texts(texts[i:i + batch_size], max_tokens).float().cpu().numpy())
    return np.concatenate(feats).astype(np.float32)


def extract_all(df, train_mask, cfg, device: str) -> dict:
    """Return a dict of feature arrays plus timing info.

    clip_img      (N, 512)        original view for every meme
    clip_img_aug  (N_train, V, 512) augmented views for training memes only
    clip_txt      (N, 512)
    roberta_txt   (N, 768)
    """
    cache = cfg.cache_dir / cfg.encoder_backend
    key = hashlib.md5("|".join(df["image_name"]).encode()).hexdigest()[:8]
    timings, params = {}, {}

    if cfg.encoder_backend == "mock":
        clip = MockEncoder(512, seed=0)
        rob = MockEncoder(768, seed=1)
    else:
        print(f"Loading {cfg.clip_model} and {cfg.text_model} on {device} ...")
        clip = ClipEncoder(cfg.clip_model, device)
        rob = RobertaEncoder(cfg.text_model, device)
    params["clip"], params["roberta"] = clip.n_params, rob.n_params

    paths = df["image_path"].tolist()
    texts = df["text"].tolist()
    train_paths = df.loc[train_mask, "image_path"].tolist()

    out = {}
    out["clip_img"], timings["clip_img"] = _cache(
        cache / f"clip_img_{key}.npz",
        lambda: encode_images(clip, paths, False, cfg.feature_batch_size, cfg.num_workers, 0))

    views = []
    for v in range(cfg.n_aug_views):
        x, t = _cache(cache / f"clip_img_aug{v}_{key}.npz",
                      lambda v=v: encode_images(clip, train_paths, True, cfg.feature_batch_size,
                                                cfg.num_workers, 1000 + v))
        views.append(x)
        timings[f"clip_img_aug{v}"] = t
    orig_train = out["clip_img"][np.asarray(train_mask)]
    out["clip_img_aug"] = np.stack([orig_train, *views], axis=1)

    out["clip_txt"], timings["clip_txt"] = _cache(
        cache / f"clip_txt_{key}.npz",
        lambda: encode_texts(clip, texts, cfg.feature_batch_size, cfg.clip_max_tokens))
    out["roberta_txt"], timings["roberta_txt"] = _cache(
        cache / f"roberta_txt_{key}.npz",
        lambda: encode_texts(rob, texts, cfg.feature_batch_size, cfg.roberta_max_tokens))

    out["timings_sec"] = timings
    out["encoder_params"] = params
    out["encoders"] = (clip, rob)
    return out


def zero_shot_prompt_embeddings(clip_encoder, classes, cfg) -> np.ndarray:
    """Averaged CLIP text embeddings for a few prompt templates per class."""
    templates = [
        "a {} meme.",
        "an internet meme with {} sentiment.",
        "a meme that feels {}.",
    ]
    words = {"negative": ["negative", "sad", "angry"],
             "neutral": ["neutral", "plain", "ordinary"],
             "positive": ["positive", "happy", "funny"]}
    embs = []
    for c in classes:
        prompts = [t.format(w) for t in templates for w in words[c]]
        e = clip_encoder.texts(prompts, cfg.clip_max_tokens).float().cpu().numpy()
        e = e.mean(0)
        embs.append(e / np.linalg.norm(e))
    return np.stack(embs)
