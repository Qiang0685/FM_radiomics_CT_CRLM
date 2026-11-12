#!/usr/bin/env python3
"""Compute and summarize paired liver-mask agreement.

Outputs per-case Dice/surface-distance metrics, summary statistics, and a run
manifest in one step.
"""
import argparse
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree


def get_cases(folder: Path):
    result = {}
    for path in sorted(folder.glob("*.nii*")):
        if path.name.startswith("._"):
            continue
        match = re.search(r"(\d{4})", path.name)
        if not match:
            continue
        case = match.group(1)
        if case in result:
            raise ValueError(f"Duplicate case ID {case} in {folder}")
        result[case] = path
    return result


def binary_surface(mask):
    structure = ndimage.generate_binary_structure(3, 1)
    return mask & ~ndimage.binary_erosion(mask, structure=structure, border_value=0)


def voxel_points_to_world(coords, affine):
    return coords @ affine[:3, :3].T + affine[:3, 3]


def compute_case_metrics(mask_a_dir: Path, mask_b_dir: Path) -> pd.DataFrame:
    a_cases, b_cases = get_cases(mask_a_dir), get_cases(mask_b_dir)
    if set(a_cases) != set(b_cases):
        raise ValueError("Mask cohorts do not have identical case IDs")
    rows = []
    for case in sorted(a_cases):
        img_a, img_b = nib.load(a_cases[case]), nib.load(b_cases[case])
        if img_a.shape != img_b.shape or not np.allclose(img_a.affine, img_b.affine, atol=1e-5, rtol=0):
            raise ValueError(f"Geometry mismatch for case {case}")
        mask_a = np.asanyarray(img_a.dataobj) > 0
        mask_b = np.asanyarray(img_b.dataobj) > 0
        n_a, n_b = int(mask_a.sum()), int(mask_b.sum())
        if n_a == 0 or n_b == 0:
            raise ValueError(f"Empty mask for case {case}")
        inter = int(np.count_nonzero(mask_a & mask_b))
        coords_a = np.argwhere(binary_surface(mask_a))
        coords_b = np.argwhere(binary_surface(mask_b))
        world_a = voxel_points_to_world(coords_a, img_a.affine)
        world_b = voxel_points_to_world(coords_b, img_b.affine)
        dist_a = cKDTree(world_b).query(world_a, k=1, workers=2)[0]
        dist_b = cKDTree(world_a).query(world_b, k=1, workers=2)[0]
        both = np.concatenate((dist_a, dist_b))
        rows.append({
            "case_id": case,
            "TotalSegmentator_liver_voxels": n_a,
            "TCIA_liver_voxels": n_b,
            "dice": 2.0 * inter / (n_a + n_b),
            "mean_surface_distance_mm": float(both.mean()),
            "hd95_mm": float(np.percentile(both, 95)),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-a", type=Path)
    ap.add_argument("--mask-b", type=Path)
    ap.add_argument("--per-case-input", type=Path, help="Precomputed per-case metrics CSV; avoids requiring NIfTI masks.")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--per-case-output", type=Path, help="Optional CSV path for per-case metrics.")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.per_case_input:
        df = pd.read_csv(args.per_case_input, dtype={"case_id": str})
        aliases = {"voxels_mask_a": "TotalSegmentator_liver_voxels",
                   "voxels_mask_b": "TCIA_liver_voxels"}
        df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})
        required = {"case_id", "TotalSegmentator_liver_voxels", "TCIA_liver_voxels", "dice",
                    "mean_surface_distance_mm", "hd95_mm"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Per-case CSV is missing columns: {sorted(missing)}")
    else:
        if not args.mask_a or not args.mask_b:
            ap.error("Provide --per-case-input or both --mask-a and --mask-b")
        df = compute_case_metrics(args.mask_a, args.mask_b)
    if len(df) != 197 or not df.case_id.is_unique:
        raise ValueError(f"Expected 197 unique cases, found {len(df)}")
    keep_columns = ["case_id", "TotalSegmentator_liver_voxels", "TCIA_liver_voxels",
                    "dice", "mean_surface_distance_mm", "hd95_mm"]
    df = df[keep_columns].copy()
    if args.per_case_output:
        args.per_case_output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.per_case_output, index=False, float_format="%.10g")

    metrics = ["dice", "mean_surface_distance_mm", "hd95_mm"]
    summary = pd.DataFrame([{
        "Metric": m, "N": len(df), "Mean": df[m].mean(), "SD": df[m].std(ddof=1),
        "Median": df[m].median(), "Q1": df[m].quantile(.25), "Q3": df[m].quantile(.75),
        "Minimum": df[m].min(), "Maximum": df[m].max()
    } for m in metrics])
    summary.to_csv(args.output_dir / "summary_metrics.csv", index=False, float_format="%.10g")

    manifest = {
        "case_count": len(df), "mask_a": str(args.mask_a) if args.mask_a else None,
        "mask_b": str(args.mask_b) if args.mask_b else None,
        "per_case_input": str(args.per_case_input) if args.per_case_input else None,
        "geometry": "Paired masks required identical shape and affine; no resampling.",
        "foreground": "Binary voxel value > 0.",
        "dice": "2*intersection/(n_voxels_A+n_voxels_B)",
        "surface_distance": "Bidirectional nearest boundary-voxel-center distances in millimetres.",
        "note": "Pairwise agreement does not establish anatomical ground truth."
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved mask agreement results to {args.output_dir}")


if __name__ == "__main__":
    main()
