"""Extract 1,024-dimensional OrganLens liver embeddings for a cohort.

Required locations are configured through environment variables rather than
study-machine absolute paths. See the package README before execution.
"""
import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = os.environ["FATTYLIVER_DATA_ROOT"]
IMAGE_DIR = Path(os.environ.get("FATTYLIVER_IMAGE_DIR", str(Path(ROOT) / "Img_nii")))
OUT = os.environ.get("FATTYLIVER_OUTPUT_DIR", str(PACKAGE_ROOT / "results"))
MODEL_CODE = Path(os.environ.get("FATTYLIVER_ORGANLENS_CODE", str(PACKAGE_ROOT / "local_model_code" / "organlens")))
WEIGHTS = os.environ["ORGANLENS_WEIGHTS"]
os.makedirs(OUT, exist_ok=True)
os.environ["XFORMERS_DISABLED"] = "1"
sys.path.insert(0, str(MODEL_CODE))

man = pd.read_csv(
    os.environ["FATTYLIVER_MANIFEST"],
    dtype={"case_id": str}).set_index("case_id")
cases = list(man.index)
print(f"[info] {len(cases)} cases", flush=True)

from organlens.inference import OrganLensEmbedder
teacher = OrganLensEmbedder(
    WEIGHTS,
    device="cpu", amp="none")
print("[info] model ready", flush=True)

done = set()
feat_path = os.path.join(OUT, "organlens_liver_features.csv")
if os.path.exists(feat_path):
    done = set(pd.read_csv(feat_path, dtype={"case_id": str}).case_id)
    print(f"[info] resuming; {len(done)} cases already complete", flush=True)

rows = {}
t0 = time.time()
for i, cid in enumerate(cases):
    if cid in done:
        continue
    try:
        payload = teacher.embed_volume(
            str(IMAGE_DIR / f"CRLM_CT_{cid}.nii.gz"),
            organs=["liver"], dataset="ctrate")
        rows[cid] = payload["embeddings"]["liver"].numpy()
    except Exception as e:
        print(f"[warn] {cid} failed: {e}", flush=True)
        continue
    n = len(rows) + len(done)
    if n % 10 == 0 or n == len(cases):
        el = time.time() - t0
        rate = el / max(1, len(rows))
        print(f"[{time.strftime('%H:%M:%S')}] {n}/{len(cases)}, {el:.0f}s, estimated remaining {rate*(len(cases)-n):.0f}s", flush=True)
        F = pd.DataFrame(rows).T
        F.index.name = "case_id"
        F.columns = [f"ol_{j}" for j in range(F.shape[1])]
        if done:
            F = pd.concat([pd.read_csv(feat_path, dtype={"case_id": str}, index_col=0), F])
        F.to_csv(feat_path)

print("ORGANLENS_FULL_DONE", flush=True)
