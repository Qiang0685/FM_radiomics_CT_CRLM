#!/usr/bin/env python3
"""Extract the two prespecified imaging baselines used in this study.

The liver metric is mean CT attenuation within the resected tumour-free liver
mask.  The spleen metric is mask volume in mL.  Source NIfTI files are read
only; no resampling, clipping, filtering, or interpolation is performed.
"""

from __future__ import annotations

import argparse
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


CASE_RE = re.compile(r"(\d{4})\.nii\.gz$")


def index_cases(directory: Path) -> dict[str, Path]:
    cases: dict[str, Path] = {}
    for path in sorted(directory.glob("*.nii.gz")):
        if path.name.startswith("._"):
            continue
        match = CASE_RE.search(path.name)
        if match:
            cases[match.group(1)] = path
    return cases


def read_mask(mask: Path, ct: nib.spatialimages.SpatialImage) -> tuple[np.ndarray, float]:
    roi = nib.load(str(mask))
    if ct.shape != roi.shape or not np.allclose(ct.affine, roi.affine, atol=1e-5, rtol=0):
        raise ValueError(f"geometry mismatch: {mask.name}")
    labels = np.unique(np.asanyarray(roi.dataobj))
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError(f"mask is not binary: {mask.name}; labels={labels.tolist()}")
    voxel_volume_ml = abs(float(np.linalg.det(roi.affine[:3, :3]))) / 1000.0
    return np.asanyarray(roi.dataobj) > 0, voxel_volume_ml


def extract_case(case_id: str, image: str, liver_mask: str, spleen_mask: str,
                 excluded_spleen_cases: set[str]) -> dict[str, object]:
    ct_image = nib.load(image)
    ct = np.asanyarray(ct_image.dataobj)
    liver, liver_voxel_ml = read_mask(Path(liver_mask), ct_image)
    if not liver.any():
        raise ValueError(f"empty resected liver mask: {case_id}")
    spleen, spleen_voxel_ml = read_mask(Path(spleen_mask), ct_image)
    spleen_voxels = int(spleen.sum())
    if case_id in excluded_spleen_cases:
        spleen_volume = np.nan
        spleen_status = "excluded_after_qc"
    elif spleen_voxels == 0:
        spleen_volume = np.nan
        spleen_status = "empty_mask"
    else:
        spleen_volume = float(spleen_voxels * spleen_voxel_ml)
        spleen_status = "ok"
    return {
        "case_id": case_id,
        "mean_liver_resected_hu": float(np.mean(ct[liver])),
        "spleen_volume_ml": spleen_volume,
        "spleen_volume_status": spleen_status,
        "liver_mask_voxels": int(liver.sum()),
        "spleen_mask_voxels": spleen_voxels,
    }


def extract_case_star(job: tuple[str, str, str, str, set[str]]) -> dict[str, object]:
    return extract_case(*job)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--liver-mask-dir", type=Path, required=True,
                        help="Resected tumour-free liver masks")
    parser.add_argument("--spleen-mask-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-spleen-cases", nargs="*", default=[],
                        help="Case IDs whose spleen volumes should be set to missing after QC")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")

    images = index_cases(args.image_dir)
    liver_masks = index_cases(args.liver_mask_dir)
    spleen_masks = index_cases(args.spleen_mask_dir)
    if len(images) != 197 or set(images) != set(liver_masks) or set(images) != set(spleen_masks):
        raise ValueError(
            f"Expected 197 matched cases; images={len(images)}, "
            f"liver_masks={len(liver_masks)}, spleen_masks={len(spleen_masks)}"
        )

    excluded = {str(case).zfill(4) for case in args.exclude_spleen_cases}
    jobs = [(case_id, str(images[case_id]), str(liver_masks[case_id]),
             str(spleen_masks[case_id]), excluded) for case_id in sorted(images)]
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        rows = list(pool.map(extract_case_star, jobs))

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, float_format="%.17g")
    print(f"Saved {len(result)} cases to {args.output}")
    print(f"Missing spleen volumes: {int(result.spleen_volume_ml.isna().sum())}")


if __name__ == "__main__":
    main()
