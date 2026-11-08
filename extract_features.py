"""Compute and cache all encoder features from Terminal (same steps as the notebook).

The notebook then finds the cached files in outputs/cache/ and skips extraction.
Usage (inside the memotion-fusion folder):
    /opt/anaconda3/bin/python -X faulthandler extract_features.py            # full dataset
    /opt/anaconda3/bin/python -X faulthandler extract_features.py --device cpu   # if the GPU crashes
"""
import sys, time
import src  # noqa: F401  (macOS fixes)
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import Config, get_device
from src import data as D, features as F

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("max_samples", nargs="?", type=int, default=None)
ap.add_argument("--device", default=None, help="mps, cpu or cuda (default: auto)")
ap.add_argument("--batch", type=int, default=None, help="images/texts per batch (default 32)")
args = ap.parse_args()
max_samples = args.max_samples
import kagglehub
path = kagglehub.dataset_download("williamscott701/memotion-dataset-7k")
cfg = Config(data_path=path, max_samples=max_samples)
if args.batch:
    cfg.feature_batch_size = args.batch
cfg.make_dirs()
device = args.device or get_device()
print(f"device={device} batch={cfg.feature_batch_size}", flush=True)

labels_csv, images_dir = D.locate_dataset(cfg.data_path)
df = D.load_labels(labels_csv, images_dir)
if cfg.max_samples:
    keep, _ = train_test_split(np.arange(len(df)), train_size=cfg.max_samples, stratify=df["label"], random_state=0)
    df = df.iloc[np.sort(keep)].reset_index(drop=True)
df = D.inspect_images(df)
df = df[df["readable"]].reset_index(drop=True)
df["split"] = D.make_splits(df, cfg.split_ratios, cfg.split_seed)
print(f"{len(df)} memes, train {int((df.split == 'train').sum())}", flush=True)

t0 = time.time()
feats = F.extract_all(df, (df["split"] == "train").values, cfg, device)
for k, v in feats.items():
    if isinstance(v, np.ndarray):
        print(f"{k:14s} {v.shape}")
print(f"Done in {time.time() - t0:.0f}s. Cached in {cfg.cache_dir}. Now Restart & Run All in the notebook.")
