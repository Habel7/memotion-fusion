"""Loading, quality assessment, text cleaning and splitting for Memotion 7k.

Expected input: the Kaggle download "williamscott701/memotion-dataset-7k",
either as the zip file or already unzipped. The loader searches for
`labels.csv` and the `images/` folder, so the exact folder layout does not matter.
"""
from __future__ import annotations

import hashlib
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split

Image.MAX_IMAGE_PIXELS = None          # some memes are very large
ImageFile.LOAD_TRUNCATED_IMAGES = False  # we want truncated files reported, not silently patched

# Five-point labels -> three classes (proposal, Section 2)
SENTIMENT_MAP = {
    "very_positive": "positive",
    "positive": "positive",
    "neutral": "neutral",
    "negative": "negative",
    "very_negative": "negative",
}
CLASSES = ["negative", "neutral", "positive"]
PLACEHOLDER = "[no text]"

# Ordinal orders for the auxiliary labels, used for subgroup (bias) analysis
AUX_ORDERS = {
    "humour": ["not_funny", "funny", "very_funny", "hilarious"],
    "sarcasm": ["not_sarcastic", "general", "twisted_meaning", "very_twisted"],
    "offensive": ["not_offensive", "slight", "very_offensive", "hateful_offensive"],
    "motivational": ["not_motivational", "motivational"],
}


# --------------------------------------------------------------------------
# Locating the dataset
# --------------------------------------------------------------------------
def locate_dataset(data_path: str | Path) -> tuple[Path, Path]:
    """Return (labels_csv, images_dir) from a zip file or any parent folder."""
    p = Path(data_path).expanduser()
    if not p.exists():
        raise FileNotFoundError(
            f"'{p}' does not exist. Download the dataset from "
            "https://www.kaggle.com/datasets/williamscott701/memotion-dataset-7k "
            "and set Config.data_path to the zip or the unzipped folder."
        )

    # If given a zip (or a folder that only contains the zip), extract it once.
    zips = [p] if p.suffix == ".zip" else list(p.glob("*.zip"))
    if zips and not list((p if p.is_dir() else p.parent).rglob("labels.csv")):
        z = zips[0]
        target = z.with_suffix("")
        if not target.exists():
            print(f"Extracting {z.name} -> {target} ...")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(target)
        p = target

    search_root = p if p.is_dir() else p.parent
    labels = sorted(search_root.rglob("labels.csv"), key=lambda x: len(x.parts))
    labels = [l for l in labels if "__MACOSX" not in l.parts]
    if not labels:
        raise FileNotFoundError(f"No labels.csv found under {search_root}")
    labels_csv = labels[0]

    images_dir = labels_csv.parent / "images"
    if not images_dir.is_dir():
        candidates = [d for d in search_root.rglob("images") if d.is_dir() and "__MACOSX" not in d.parts]
        if not candidates:
            raise FileNotFoundError(f"No images/ folder found under {search_root}")
        images_dir = candidates[0]
    return labels_csv, images_dir


# --------------------------------------------------------------------------
# Text cleaning
# --------------------------------------------------------------------------
_URL = re.compile(r"(https?://\S+|www\.\S+)")
# meme-generator watermarks such as "memegenerator.net", "imgflip.com", "quickmeme.com"
_WATERMARK = re.compile(r"\b[\w-]+\.(com|net|org|co|in|us|io|ly)\b")
_NON_TEXT = re.compile(r"[^a-z0-9\s'!?.,]")
_SPACES = re.compile(r"\s+")


def _valid(s) -> bool:
    return isinstance(s, str) and s.strip() != "" and s.strip().lower() not in {"nan", "none"}


def clean_text(s: str) -> str:
    s = s.lower()
    s = _URL.sub(" ", s)
    s = _WATERMARK.sub(" ", s)
    s = _NON_TEXT.sub(" ", s)
    s = _SPACES.sub(" ", s).strip()
    return s


def build_text(corrected, ocr) -> tuple[str, str]:
    """Corrected OCR first, raw OCR second, placeholder last.

    Returns (clean_text, source) where source is 'corrected', 'ocr' or 'placeholder'.
    """
    for raw, src in ((corrected, "corrected"), (ocr, "ocr")):
        if _valid(raw):
            t = clean_text(raw)
            if t:
                return t, src
    return PLACEHOLDER, "placeholder"


# --------------------------------------------------------------------------
# Loading labels
# --------------------------------------------------------------------------
def load_labels(labels_csv: Path, images_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    df = df.loc[:, ~df.columns.str.lower().str.startswith("unnamed")]
    df.columns = [c.strip().lower() for c in df.columns]
    for c in ["image_name", "overall_sentiment", *AUX_ORDERS]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    df["sentiment_5"] = df["overall_sentiment"].str.lower()
    df["sentiment"] = df["sentiment_5"].map(SENTIMENT_MAP)
    unknown = df["sentiment"].isna().sum()
    if unknown:
        print(f"Warning: {unknown} rows have an unrecognised sentiment label and are dropped.")
        df = df[df["sentiment"].notna()].copy()
    df["label"] = df["sentiment"].map({c: i for i, c in enumerate(CLASSES)}).astype(int)

    built = [build_text(c, o) for c, o in zip(df.get("text_corrected"), df.get("text_ocr"))]
    df["text"] = [b[0] for b in built]
    df["text_source"] = [b[1] for b in built]
    df["ocr_missing"] = ~df["text_ocr"].apply(_valid)
    df["corrected_missing"] = ~df["text_corrected"].apply(_valid)
    df["n_words"] = df["text"].apply(lambda t: 0 if t == PLACEHOLDER else len(t.split()))

    df["image_path"] = df["image_name"].apply(lambda n: str(images_dir / n))
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------
def load_image(path: str | Path) -> Image.Image:
    """Open any meme image as RGB. Animated GIFs use their first frame;
    transparent images are composited onto white so text stays visible."""
    img = Image.open(path)
    if getattr(img, "is_animated", False):
        img.seek(0)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def _inspect_image(path: str) -> dict:
    rec = {"readable": False, "width": np.nan, "height": np.nan, "mode": None,
           "format": None, "n_frames": 1, "md5": None, "error": None}
    try:
        p = Path(path)
        if not p.exists():
            rec["error"] = "missing file"
            return rec
        raw = p.read_bytes()
        rec["md5"] = hashlib.md5(raw).hexdigest()
        with Image.open(p) as im:
            rec["format"], rec["mode"] = im.format, im.mode
            rec["width"], rec["height"] = im.size
            rec["n_frames"] = getattr(im, "n_frames", 1)
            im.load()  # forces a full decode, catches truncated files
        rec["readable"] = True
    except Exception as e:  # noqa: BLE001 - we want every failure reason
        rec["error"] = f"{type(e).__name__}: {str(e)[:80]}"
    return rec


def inspect_images(df: pd.DataFrame, n_threads: int = 8) -> pd.DataFrame:
    with ThreadPoolExecutor(n_threads) as ex:
        recs = list(ex.map(_inspect_image, df["image_path"]))
    info = pd.DataFrame(recs, index=df.index)
    return pd.concat([df, info], axis=1)


# --------------------------------------------------------------------------
# Quality summary
# --------------------------------------------------------------------------
def quality_summary(df: pd.DataFrame) -> dict:
    n = len(df)
    readable = df["readable"]
    dup_hash = df.loc[readable, "md5"].duplicated(keep=False)
    real_text = df["text"] != PLACEHOLDER
    dup_text = df.loc[real_text, "text"].duplicated(keep=False)
    s = {
        "n_rows": int(n),
        "n_unique_image_names": int(df["image_name"].nunique()),
        "sentiment_5_counts": df["sentiment_5"].value_counts().to_dict(),
        "sentiment_3_counts": df["sentiment"].value_counts().reindex(CLASSES).to_dict(),
        "sentiment_3_pct": (df["sentiment"].value_counts(normalize=True).reindex(CLASSES) * 100).round(1).to_dict(),
        "ocr_missing": int(df["ocr_missing"].sum()),
        "corrected_missing": int(df["corrected_missing"].sum()),
        "text_source_counts": df["text_source"].value_counts().to_dict(),
        "unreadable_images": int((~readable).sum()),
        "unreadable_examples": df.loc[~readable, ["image_name", "error"]].head(10).to_dict("records"),
        "image_formats": df["format"].value_counts(dropna=False).astype(int).to_dict(),
        "image_modes": df["mode"].value_counts(dropna=False).astype(int).to_dict(),
        "animated_images": int((df["n_frames"] > 1).sum()),
        "exact_duplicate_images": int(dup_hash.sum()),
        "duplicate_texts": int(dup_text.sum()),
        "width_median": float(df["width"].median()),
        "height_median": float(df["height"].median()),
        "width_min": float(df["width"].min()),
        "height_min": float(df["height"].min()),
        "small_images_lt_224": int(((df["width"] < 224) | (df["height"] < 224)).sum()),
        "words_median": float(df["n_words"].median()),
        "words_p95": float(df["n_words"].quantile(0.95)),
    }
    # Exact duplicate images whose labels disagree: direct evidence of annotation subjectivity
    groups = df.loc[readable].groupby("md5")["sentiment"].nunique()
    s["duplicate_image_groups"] = int((df.loc[readable, "md5"].value_counts() > 1).sum())
    s["duplicate_groups_with_conflicting_labels"] = int((groups > 1).sum())
    return s


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def make_splits(df: pd.DataFrame, ratios=(0.7, 0.15, 0.15), seed: int = 42) -> pd.Series:
    """Stratified train/val/test split on the 3-class label."""
    idx = np.arange(len(df))
    y = df["label"].values
    train_idx, rest = train_test_split(idx, test_size=1 - ratios[0], stratify=y, random_state=seed)
    val_share = ratios[1] / (ratios[1] + ratios[2])
    val_idx, test_idx = train_test_split(rest, test_size=1 - val_share, stratify=y[rest], random_state=seed)
    split = pd.Series("train", index=df.index)
    split.iloc[val_idx] = "val"
    split.iloc[test_idx] = "test"
    return split
