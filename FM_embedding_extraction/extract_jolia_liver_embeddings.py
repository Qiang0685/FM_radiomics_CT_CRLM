"""Extract global and 576-dimensional liver Jolia embeddings for a cohort.

For the present study only the liver embedding is used in modelling. Paths are
provided through environment variables; see the package README.
"""
import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = os.environ["FATTYLIVER_DATA_ROOT"]
IMAGE_DIR = Path(os.environ.get("FATTYLIVER_IMAGE_DIR", str(Path(ROOT) / "Img_nii")))
OUT = os.environ.get("FATTYLIVER_OUTPUT_DIR", str(PACKAGE_ROOT / "results"))
REPO = Path(os.environ.get("FATTYLIVER_JOLIA_CODE", str(PACKAGE_ROOT / "local_model_code" / "jolia")))
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, str(REPO))

man = pd.read_csv(
    os.environ["FATTYLIVER_MANIFEST"],
    dtype={"case_id": str}).set_index("case_id")
cases = list(man.index)
print(f"[info] {len(cases)} cases", flush=True)

from preprocessing_jolia import JoliaPreprocessor
from transformers import AutoModel
pre = JoliaPreprocessor()
model = AutoModel.from_pretrained(REPO, trust_remote_code=True).eval()
print(f"[info] model ready; {len(model.organ_slot_names)} organ slots; liver available: {'liver' in model.organ_slot_names}", flush=True)

g_path = os.path.join(OUT, "jolia_global_features.csv")
l_path = os.path.join(OUT, "jolia_liver_features.csv")
done = set()
if os.path.exists(g_path):
    done = set(pd.read_csv(g_path, dtype={"case_id": str}).case_id)
    print(f"[info] resuming; {len(done)} cases already complete", flush=True)

G, L = {}, {}
t0 = time.time()
with torch.no_grad():
    for i, cid in enumerate(cases):
        if cid in done:
            continue
        try:
            img = sitk.ReadImage(str(IMAGE_DIR / f"CRLM_CT_{cid}.nii.gz"))
            vol = np.transpose(sitk.GetArrayFromImage(img).astype(np.float32), (1, 2, 0))
            sx, sy, sz = img.GetSpacing()
            image = pre(vol, resolution=(sy, sx, sz)).unsqueeze(0)
            G[cid] = model(image).pooler_output[0].numpy()
            org = model.encode_organs(image, organs=["liver"], normalize=True)
            L[cid] = org["liver"][0].numpy()
        except Exception as e:
            print(f"[warn] {cid} failed: {e}", flush=True)
            continue
        n = len(G) + len(done)
        if n % 10 == 0 or n == len(cases):
            el = time.time() - t0
            rate = el / max(1, len(G))
            print(f"[{time.strftime('%H:%M:%S')}] {n}/{len(cases)}, {el:.0f}s, estimated remaining {rate*(len(cases)-n):.0f}s", flush=True)
            for path, d in [(g_path, G), (l_path, L)]:
                F = pd.DataFrame(d).T
                F.index.name = "case_id"
                prefix = "jg" if "global" in path else "jl"
                F.columns = [f"{prefix}_{j}" for j in range(F.shape[1])]
                if done:
                    old = pd.read_csv(path, dtype={"case_id": str}, index_col=0)
                    F = pd.concat([old, F])
                F.to_csv(path)
print("JOLIA_FULL_DONE", flush=True)
