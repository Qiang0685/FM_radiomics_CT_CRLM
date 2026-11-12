"""
CRLM cohort data loader.

Responsibilities: identifier normalisation, NIfTI path pairing, clinical-table
joining, and pairing validation. Set ``FATTYLIVER_DATA_ROOT`` to override the
package-relative default data root.
"""
import os
import re
import warnings
import nibabel as nib
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

DATA_ROOT = os.environ.get("FATTYLIVER_DATA_ROOT")
if not DATA_ROOT:
    raise RuntimeError("Set FATTYLIVER_DATA_ROOT before importing data_loader")

DIRS = {
    "ct": ("Img_nii", "CRLM_CT_{}.nii.gz"),
    "mask_liver": ("Msk_liver_lcc_TS", "CRLM-CT-{}.nii.gz"),
}

def extract_id(filename: str) -> str | None:
    """Extract a four-digit case identifier from an arbitrary filename."""
    m = re.search(r"(\d{4})", filename)
    return m.group(1) if m else None


def list_cases() -> list[str]:
    """List unique four-digit identifiers in the CT directory in sorted order."""
    ids = set()
    for f in os.listdir(os.path.join(DATA_ROOT, DIRS["ct"][0])):
        cid = extract_id(f)
        if cid:
            ids.add(cid)
    return sorted(ids)


def get_paths(case_id: str) -> dict[str, str]:
    """Return absolute paths to the four expected files for one case."""
    out = {}
    for key, (d, pat) in DIRS.items():
        out[key] = os.path.join(DATA_ROOT, d, pat.format(case_id))
    return out


def verify_case(case_id: str) -> dict:
    """
    Validate one paired case:
    - CT and liver-mask files exist;
    - CT and the liver mask have matching shape and affine;
    - masks are non-empty; and
    - CT intensity range is compatible with Hounsfield units.

    Returns a dictionary with all checks and errors.
    """
    paths = get_paths(case_id)
    result = {"case_id": case_id, "ok": True, "errors": []}

    # File existence
    for k, p in paths.items():
        if not os.path.isfile(p):
            result["errors"].append(f"missing:{k}")
            result["ok"] = False
    if not result["ok"]:
        return result

    # Image loading
    imgs = {}
    for k, p in paths.items():
        try:
            imgs[k] = nib.load(p)
        except Exception as e:
            result["errors"].append(f"load_fail:{k}:{e}")
            result["ok"] = False
    if not result["ok"]:
        return result

    ct = imgs["ct"]
    ct_shape = ct.shape
    ct_affine = ct.affine
    ct_data = ct.get_fdata()

    result["ct_shape"] = ct_shape
    result["ct_spacing"] = tuple(round(float(z), 3) for z in ct.header.get_zooms()[:3])
    result["ct_min"] = float(ct_data.min())
    result["ct_max"] = float(ct_data.max())

    # Mask geometric agreement and non-emptiness
    for mk in ["mask_liver"]:
        m = imgs[mk]
        if m.shape != ct_shape:
            result["errors"].append(f"shape_mismatch:{mk}:{m.shape}vs{ct_shape}")
            result["ok"] = False
        if not np.allclose(m.affine, ct_affine, atol=1e-3):
            result["errors"].append(f"affine_mismatch:{mk}")
            result["ok"] = False
        nvox = int((m.get_fdata() > 0).sum())
        result[f"{mk}_nvox"] = nvox
        if nvox == 0:
            result["errors"].append(f"empty_mask:{mk}")
            result["ok"] = False

    # Plausible Hounsfield-unit range
    if result["ct_max"] < 500:
        result["errors"].append(f"suspicious_ct_max:{result['ct_max']:.0f}")
        result["ok"] = False
    if result["ct_min"] > -500:
        result["errors"].append(f"suspicious_ct_min:{result['ct_min']:.0f}")
        result["ok"] = False

    return result


def verify_all(verbose: bool = True) -> pd.DataFrame:
    """Validate all available cases and return a summary DataFrame."""
    cases = list_cases()
    rows = []
    for cid in cases:
        r = verify_case(cid)
        rows.append(r)
        if verbose and not r["ok"]:
            print(f"[FAIL] {cid}: {r['errors']}")
    df = pd.DataFrame(rows)
    n_ok = int(df["ok"].sum())
    print(f"\nValidation completed: {n_ok}/{len(df)} cases passed")
    if n_ok < len(df):
        print("Failed cases:", df.loc[~df["ok"], "case_id"].tolist())
    return df


def build_manifest() -> pd.DataFrame:
    """Create a case manifest from the available CT and liver-mask files."""
    cases = list_cases()
    rows = []
    for cid in cases:
        p = get_paths(cid)
        row = {"case_id": cid, **{f"{k}_path": v for k, v in p.items()}}
        rows.append(row)
    manifest = pd.DataFrame(rows)
    out = os.environ.get("FATTYLIVER_MANIFEST_OUTPUT", os.path.join(DATA_ROOT, "manifest.csv"))
    manifest.to_csv(out, index=False)
    print(f"Manifest saved: {out} ({len(manifest)} rows × {len(manifest.columns)} columns)")
    return manifest


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="validate all paired cases")
    ap.add_argument("--manifest", action="store_true", help="generate manifest.csv")
    args = ap.parse_args()

    if args.verify:
        verify_all()
    if args.manifest:
        build_manifest()
    if not args.verify and not args.manifest:
        print("Usage: python data_loader.py --verify and/or --manifest")
