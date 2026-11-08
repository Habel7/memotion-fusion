"""Find which combination segfaults. Each test runs in its own process.

Usage: /opt/anaconda3/bin/python diagnose.py
"""
import subprocess, sys, textwrap

COMMON = textwrap.dedent('''
    import src, sys, numpy as np, torch
    from src import data as D, features as F
    import kagglehub
    path = kagglehub.dataset_download("williamscott701/memotion-dataset-7k")
    lab, img = D.locate_dataset(path)
    df = D.load_labels(lab, img)
''')

TESTS = {
    "1 mps, no image inspection, 32 imgs": "df = df.head(32); dev='mps'",
    "2 mps, after full image inspection (threads), 32 imgs": "df = D.inspect_images(df); df = df[df.readable].head(32); dev='mps'",
    "3 mps, after image inspection (1 thread), 32 imgs": "df = D.inspect_images(df, n_threads=1); df = df[df.readable].head(32); dev='mps'",
    "4 cpu, after full image inspection (threads), 32 imgs": "df = D.inspect_images(df); df = df[df.readable].head(32); dev='cpu'",
    "5 mps, no inspection, 640 imgs": "df = df.head(640); dev='mps'",
}
RUN = textwrap.dedent('''
    clip = F.ClipEncoder("openai/clip-vit-base-patch32", dev)
    x = F.encode_images(clip, df.image_path, False, 32, 0, 0)
    print("RESULT_OK", x.shape)
''')

for name, setup in TESTS.items():
    code = COMMON + setup + "\n" + RUN
    p = subprocess.run([sys.executable, "-X", "faulthandler", "-c", code],
                       capture_output=True, text=True, timeout=900)
    ok = "RESULT_OK" in p.stdout
    status = "OK" if ok else f"FAILED (exit code {p.returncode})"
    print(f"{name:58s} {status}", flush=True)
    if not ok:
        tail = (p.stderr or "").strip().splitlines()[-12:]
        print("    " + "\n    ".join(tail))
