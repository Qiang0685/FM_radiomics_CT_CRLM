"""Extract 2,048-dimensional Merlin image embeddings.

Usage:
    python extract_merlin_embeddings.py --case 1001
    python extract_merlin_embeddings.py --all
    python extract_merlin_embeddings.py --all --resume

The study configuration is ``Merlin(ImageEmbedding=True)`` with the default
image-embedding checkpoint recorded in ``MODEL_VERSION_RECORD.md``. This is an
embedding extractor, not a zero-shot analysis.
"""
import os
import sys
import argparse
import warnings
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_CODE = os.environ.get("FATTYLIVER_MERLIN_CODE", os.path.join(PACKAGE_ROOT, "local_model_code", "merlin"))
sys.path.insert(0, PACKAGE_ROOT)
sys.path.insert(0, MODEL_CODE)
from data_loader import list_cases, get_paths, build_manifest

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.environ.get("FATTYLIVER_CACHE_DIR", os.path.join(CODE_DIR, "cache"))
FEATURES_CSV = os.environ.get(
    "FATTYLIVER_OUTPUT_CSV", os.path.join(CODE_DIR, "merlin_image_embeddings.csv")
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_model():
    from merlin import Merlin
    model = Merlin(ImageEmbedding=True)
    model.eval()
    model.to(DEVICE)
    return model


def extract_single(model, case_id: str) -> np.ndarray:
    """Extract a 2,048-dimensional NumPy vector for one case."""
    from merlin.data import DataLoader

    paths = get_paths(case_id)
    datalist = [{"image": paths["ct"]}]
    dl = DataLoader(
        datalist=datalist,
        cache_dir=CACHE_DIR,
        batchsize=1,
        shuffle=False,
        num_workers=0,
    )
    for batch in dl:
        img = batch["image"].to(DEVICE)
        with torch.no_grad():
            out = model(img)
        vec = out[0].squeeze(0).cpu().numpy()
        assert vec.shape == (2048,), f"Unexpected output shape: {vec.shape}"
        assert np.all(np.isfinite(vec)), "Output contains NaN or Inf"
        return vec
    raise RuntimeError(f"DataLoader did not yield a batch for case {case_id}")


def run_single(case_id: str):
    print(f"[single-case validation] case {case_id}")
    print(f"Device: {DEVICE}")
    model = load_model()
    t0 = time.time()
    vec = extract_single(model, case_id)
    dt = time.time() - t0
    print(f"Output shape: {vec.shape}")
    print(f"First 10 values: {vec[:10]}")
    print(f"Summary: min={vec.min():.4f} max={vec.max():.4f} mean={vec.mean():.4f} std={vec.std():.4f}")
    print(f"Elapsed: {dt:.1f}s")
    print("✓ Single-case validation passed")


def run_all(resume: bool = False):
    manifest = build_manifest()
    case_ids = manifest["case_id"].tolist()
    print(f"{len(case_ids)} cases queued for extraction")
    print(f"Device: {DEVICE}")

    # Resume support
    done = set()
    if resume and os.path.isfile(FEATURES_CSV):
        old = pd.read_csv(FEATURES_CSV, index_col=0)
        done = set(old.index.astype(str))
        print(f"Resuming with {len(done)} completed cases")

    model = load_model()
    rows = {}
    t_start = time.time()
    todo = [c for c in case_ids if c not in done]
    pbar = tqdm(todo, desc="Extracting embeddings", unit="case")
    for cid in pbar:
        try:
            vec = extract_single(model, cid)
            rows[cid] = vec
            elapsed = time.time() - t_start
            eta = elapsed / (pbar.n + 1) * (len(todo) - pbar.n - 1) if pbar.n > 0 else 0
            pbar.set_postfix_str(f"elapsed {elapsed/60:.1f} min; ETA {eta/60:.1f} min")
        except Exception as e:
            pbar.write(f"{cid} ✗ {e}")
            rows[cid] = np.full(2048, np.nan)
    pbar.close()

    # Merge previously completed cases
    if done and resume:
        old = pd.read_csv(FEATURES_CSV, index_col=0)
        for cid in done:
            if cid not in rows:
                rows[cid] = old.loc[cid].values

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "case_id"
    df.columns = [f"merlin_{j:04d}" for j in range(2048)]
    df.to_csv(FEATURES_CSV)
    print(f"\nEmbeddings saved: {FEATURES_CSV} ({df.shape[0]} rows × {df.shape[1]} columns)")

    # Quality report
    n_nan = df.isna().any(axis=1).sum()
    print(f"Cases containing NaN: {n_nan}")
    if n_nan:
        print("  ->", df[df.isna().any(axis=1)].index.tolist())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=str, help="single case for validation, e.g. 1001")
    ap.add_argument("--all", action="store_true", help="extract all cases")
    ap.add_argument("--resume", action="store_true", help="resume a previous extraction")
    args = ap.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)

    if args.case:
        run_single(args.case)
    elif args.all:
        run_all(resume=args.resume)
    else:
        print("Usage: python extract_merlin_embeddings.py --case 1001 or --all [--resume]")
