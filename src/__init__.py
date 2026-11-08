"""Package setup: runs before any Hugging Face import.

macOS allows only 256 open files per process by default. Model downloads
with notebook progress bars can exceed that ("Too many open files"),
so we raise the limit and switch the download progress bars off.
"""
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Anaconda and PyTorch each ship an OpenMP runtime; loading both kills the
# kernel silently on macOS ("OMP: Error #15"). Allow the duplicate.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Let PyTorch fall back to CPU for any op the Apple GPU (MPS) lacks.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

try:
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = 10240 if hard == resource.RLIM_INFINITY else min(10240, hard)
    if soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
except (ImportError, ValueError, OSError):
    pass  # Windows or a locked-down system: keep the default

# IMPORTANT (macOS + Anaconda): PyTorch must be imported before scikit-learn.
# Both bundle an OpenMP runtime; if scikit-learn's loads first, the first
# PyTorch computation segfaults ("kernel died"). Importing torch here, the
# first thing any `from src import ...` runs, fixes the load order.
import torch  # noqa: E402,F401

# If the kernel dies on a hard crash (segfault/abort), write the Python stack
# to crash_log.txt so the cause is visible instead of "kernel died".
try:
    import faulthandler
    from pathlib import Path as _P

    _CRASH_LOG = open(_P(__file__).resolve().parent.parent / "crash_log.txt", "w")
    faulthandler.enable(file=_CRASH_LOG, all_threads=True)
except Exception:  # noqa: BLE001
    pass
