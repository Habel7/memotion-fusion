# Multimodal Sentiment Classification of Internet Memes (Memotion 7k)

COMP1013 Multi-Modal Data Analysis, Project 2.2. Group Fusion Force (Aksa, Sona, Raynould, Habel Benny).

We classify memes as **negative / neutral / positive** from the meme **image** and its **text**, and compare
**early fusion** (feature-level), **late fusion** (decision-level) and **attention-gated fusion** (per-meme modality
weights) against image-only, text-only, zero-shot and linear baselines, with Monte Carlo dropout uncertainty. A second notebook tests whether the task is learnable at all.

**Main finding:** every model stays close to chance (macro-F1 0.32-0.35, macro ROC-AUC about 0.50), and fusion does not
beat text alone. Sanity checks show the pipeline is correct (CLIP matches 72.6% of memes to their own caption among 1,000)
while the Memotion labels are barely predictable and inconsistent (near-identical memes share a label only 60% of the time,
against 46% by chance).

## Repository structure

```
memotion-fusion/
├── memotion_fusion.ipynb        # main notebook: data quality, preprocessing, features, all models, evaluation, figures
├── 02_label_reliability.ipynb   # pipeline sanity check + label-reliability analysis (run after the main notebook)
├── src/
│   ├── __init__.py              # macOS/Anaconda fixes (import order, file limits) - loaded automatically
│   ├── config.py                # every setting in one dataclass (Config)
│   ├── data.py                  # dataset discovery, text cleaning, image loading, quality checks, stratified split
│   ├── features.py              # frozen CLIP ViT-B/32 and RoBERTa encoders, augmentation, feature cache
│   ├── models.py                # MLP heads, early / late / attention-gated fusion, MC dropout, logistic regression, zero-shot CLIP
│   ├── evaluate.py              # metrics, bootstrap CIs, McNemar test, subgroup analysis, near-duplicate check
│   ├── reliability.py           # retrieval check, cross-validated linear probes, label agreement
│   └── plots.py                 # all report figures
├── extract_features.py          # optional: compute/cache features from Terminal instead of the notebook
├── diagnose.py                  # optional: isolate crashes on macOS (see Troubleshooting)
├── requirements.txt
└── data/                        # put the dataset here (not committed)
```

## 1. Setup

Python 3.10+ (tested on Python 3.14, PyTorch 2.14, transformers 5, Apple M5).

**Terminal / conda**
```bash
conda create -n memotion python=3.11 -y
conda activate memotion
pip install -r requirements.txt
```

**Anaconda Navigator:** Environments > Create (`memotion`, Python 3.11) > ▶ Open Terminal >
`pip install -r requirements.txt`, then Home > select `memotion` > Launch Jupyter Notebook.
Alternatively run `%pip install -r requirements.txt` in the first notebook cell and restart the kernel.

## 2. Get the data

Either option works; the loader searches the given path recursively for `labels.csv` and `images/`.

* **kagglehub (recommended):** in the first code cell of the notebook set
  ```python
  import kagglehub
  DATA_PATH = kagglehub.dataset_download("williamscott701/memotion-dataset-7k")
  ```
* **Manual:** download the zip from <https://www.kaggle.com/datasets/williamscott701/memotion-dataset-7k>,
  put it (zipped or unzipped) in `data/` and keep `DATA_PATH = "data"`.

## 3. Run

1. Open `memotion_fusion.ipynb` and **Run All**. It auto-selects Apple MPS, CUDA or CPU.
   * First run downloads CLIP and RoBERTa (about 1 GB) and extracts features (about 5-15 min on an M5).
     Features are cached in `outputs/cache/`, so later runs take a few minutes.
   * Quick dry run: `cfg = Config(data_path=DATA_PATH, max_samples=1500)`. Delete `outputs/` before the full run.
2. Open `02_label_reliability.ipynb` and **Run All** (about 1 minute, no GPU needed).

Outputs:

| Path | Contents |
|---|---|
| `outputs/figures/` | fig01-fig13 used in the report |
| `outputs/tables/` | main results, all metrics, overfitting, ablations, CIs, McNemar, subgroups, probes |
| `outputs/results_summary.json` | every number in the report, plus configuration and environment |
| `outputs_for_report.zip` | figures + tables + summary bundled |

## 4. Configuration (src/config.py)

| Setting | Value |
|---|---|
| Labels | 5-point sentiment merged to negative / neutral / positive |
| Split | stratified 70 / 15 / 15, seed 42 |
| Text | corrected OCR > raw OCR > `[no text]`; lower-case, URLs/watermarks removed; 77 CLIP / 128 RoBERTa tokens |
| Images | first GIF frame, alpha on white, RGB, 224×224, CLIP normalisation; 2 augmented views per training meme (no flips) |
| Encoders | `openai/clip-vit-base-patch32` (frozen), `roberta-base` mean-pooled (frozen) |
| Head | Dropout 0.1 → Linear(256) → GELU → Dropout 0.3 → Linear(3) |
| Training | AdamW lr 1e-3, wd 1e-4, batch 128, class-weighted CE, ≤80 epochs, early stopping on val macro-F1 (patience 10) |
| Seeds | 0-4 (every model repeated 5 times) |
| Late fusion | `p = w·p_image + (1-w)·p_text`, w ∈ {0, 0.05, …, 1} chosen on validation |
| Attention fusion | per modality Linear(256)+GELU; attention Linear(64)-tanh-Linear(1), softmax over modalities per meme |
| MC dropout | 30 stochastic passes at test time, predictive entropy for selective prediction |

## 5. Results (test set, mean ± std over 5 seeds)

| Model | Macro-F1 | Balanced acc. | Macro AUC |
|---|---|---|---|
| Majority class | 0.249 | 0.333 | 0.500 |
| Image only (CLIP) | 0.327 ± 0.006 | 0.329 ± 0.005 | 0.494 ± 0.008 |
| Text only (CLIP) | **0.345 ± 0.015** | 0.346 ± 0.014 | 0.513 ± 0.009 |
| Early fusion (CLIP img + RoBERTa) | 0.337 ± 0.018 | 0.337 ± 0.019 | 0.501 ± 0.006 |
| Late fusion (CLIP img + RoBERTa) | 0.340 ± 0.007 | 0.343 ± 0.007 | 0.509 ± 0.008 |
| Attention fusion (CLIP img + CLIP txt) | 0.340 ± 0.010 | 0.340 ± 0.010 | 0.494 ± 0.006 |

Full tables are written to `outputs/tables/`.

## 6. Reproducibility

* Fixed split seed and model seeds; classifier heads train on CPU, so reruns from the cache give identical numbers.
* `results_summary.json` records library versions and the accelerator used.

## Troubleshooting (macOS + Anaconda)

* **Kernel dies / segmentation fault:** Anaconda's scikit-learn and pip's PyTorch each ship an OpenMP runtime; if
  scikit-learn's loads first, PyTorch crashes. `src/__init__.py` imports PyTorch first and sets
  `KMP_DUPLICATE_LIB_OK`; always import `src` before scikit-learn. t-SNE uses `method="exact"` for the same reason.
  `crash_log.txt` records the Python stack of any hard crash; `python diagnose.py` isolates the failing step.
* **"Too many open files" while downloading models:** handled in `src/__init__.py` (raises the limit, disables
  Hugging Face progress bars).
* **`No module named ...`:** Jupyter is using another environment; run `%pip install -r requirements.txt` in a cell.

## Data, licence and ethics

Memotion 7k (Sharma et al., 2020, SemEval-2020 Task 8) is distributed for research; we use it for non-commercial
academic work only and do not redistribute it. It contains offensive and hateful content and images of public figures.
The models here are **not suitable for content moderation** (see the report's ethics section).

## References

* Sharma, C. et al. (2020). SemEval-2020 Task 8: Memotion analysis - the visuo-lingual metaphor! SemEval. https://aclanthology.org/2020.semeval-1.99
* Radford, A. et al. (2021). Learning transferable visual models from natural language supervision (CLIP). ICML. https://arxiv.org/abs/2103.00020
* Liu, Y. et al. (2019). RoBERTa: A robustly optimized BERT pretraining approach. https://arxiv.org/abs/1907.11692
