#!/usr/bin/env python3
"""Portable PyRadiomics extraction for the three CRLM liver mask sets.

The extraction settings and quality-control gates are defined explicitly below;
all input and output paths are supplied through command-line arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing
import os
import platform
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import nibabel as nib
import numpy as np
import pandas as pd
import radiomics
from radiomics import featureextractor
import SimpleITK as sitk


MASK_SETS = {
    "whole_noVes": "Msk_Liver_whole_noVes",
    "whole_noVesTumor": "Msk_Liver_whole_noVesTumor",
    "resected_noVesTumor": "Msk_Liver_resected_noVesTumor",
}
CASE_RE = re.compile(r"(\d{4})\.nii\.gz$")
PARAMETERS = {
    "imageType": {"Original": {}, "Wavelet": {}, "LoG": {"sigma": [1.0, 2.0, 3.0]}},
    "featureClass": {
        "firstorder": [], "shape": [], "glcm": [], "glrlm": [],
        "glszm": [], "gldm": [], "ngtdm": [],
    },
    "setting": {
        "binWidth": 25,
        "interpolator": "sitkBSpline",
        "resampledPixelSpacing": [1, 1, 1],
        "normalize": False,
    },
}
EXPECTED_FEATURES = 1130


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_directory(directory: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(directory.glob("*.nii.gz")):
        if path.name.startswith("._"):
            continue
        match = CASE_RE.search(path.name)
        if match:
            found[match.group(1)] = path
    return found


def make_extractor() -> featureextractor.RadiomicsFeatureExtractor:
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    radiomics.logger.setLevel(logging.ERROR)
    return featureextractor.RadiomicsFeatureExtractor(PARAMETERS)


def extract_one(case_id: str, image: str, mask: str, expected_names: tuple[str, ...] | None):
    ct, roi = nib.load(image), nib.load(mask)
    if ct.shape != roi.shape or not np.allclose(ct.affine, roi.affine, atol=1e-5, rtol=0):
        raise ValueError(f"geometry mismatch: {case_id}")
    array = np.asanyarray(roi.dataobj)
    if not np.array_equal(np.unique(array), [0, 1]) or not array.any():
        raise ValueError(f"invalid binary mask: {case_id}")
    result = make_extractor().execute(image, mask)
    features = {key: float(value) for key, value in result.items() if not key.startswith("diagnostics_")}
    if len(features) != EXPECTED_FEATURES or not np.isfinite(list(features.values())).all():
        raise ValueError(f"invalid feature set: {case_id}, n={len(features)}")
    if expected_names is not None and tuple(features) != expected_names:
        raise ValueError(f"feature ordering mismatch: {case_id}")
    qc = {
        "case_id": case_id,
        "mask_voxels": int(array.sum()),
        "mask_volume_mm3": float(array.sum() * np.prod(roi.header.get_zooms()[:3])),
        "feature_count": len(features),
    }
    return case_id, features, qc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", choices=sorted(MASK_SETS), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke-case", help="Run exactly one case without writing output files")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")

    image_dir = args.image_dir.resolve()
    mask_dir = (args.mask_root / MASK_SETS[args.label]).resolve()
    output_dir = args.output_dir.resolve()
    images = index_directory(image_dir)
    masks = index_directory(mask_dir)
    if len(images) != 197 or len(masks) != 197 or set(images) != set(masks):
        raise RuntimeError(f"cohort mismatch: images={len(images)}, masks={len(masks)}")
    cases = sorted(images)
    if args.smoke_case:
        if args.smoke_case not in images:
            raise ValueError(f"unknown smoke case: {args.smoke_case}")
        case_id, features, qc = extract_one(args.smoke_case, str(images[args.smoke_case]), str(masks[args.smoke_case]), None)
        print(json.dumps({"smoke_passed": True, "case_id": case_id, "feature_count": len(features), "qc": qc}, indent=2), flush=True)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "label": args.label,
        "image_dir": str(image_dir),
        "mask_dir": str(mask_dir),
        "parameters": PARAMETERS,
        "pyradiomics": radiomics.__version__,
        "simpleitk": sitk.Version_VersionString(),
        "nibabel": nib.__version__,
        "pandas": pd.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "script_sha256": sha256(Path(__file__)),
    }
    (output_dir / f"radiomics_{args.label}_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    rows, qc_rows, failures = [], [], []
    expected_names: tuple[str, ...] | None = None
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs = {pool.submit(extract_one, case_id, str(images[case_id]), str(masks[case_id]), None): case_id for case_id in cases}
        for number, future in enumerate(as_completed(jobs), start=1):
            case_id = jobs[future]
            try:
                case_id, features, qc = future.result()
                if expected_names is None:
                    expected_names = tuple(features)
                elif tuple(features) != expected_names:
                    raise ValueError("feature ordering mismatch")
                rows.append({"case_id": case_id, **features})
                qc_rows.append(qc)
                if number % 20 == 0 or number == len(cases):
                    print(f"[{args.label} {number}/{len(cases)}] {case_id}", flush=True)
            except Exception as exc:
                failures.append({"case_id": case_id, "error": str(exc)})
                print(f"FAILED {args.label} {case_id}: {exc}", flush=True)
    if failures:
        (output_dir / f"radiomics_{args.label}_failures.json").write_text(json.dumps(failures, indent=2) + "\n")
        raise RuntimeError(f"{args.label}: {len(failures)} failed cases")

    frame = pd.DataFrame(rows).sort_values("case_id")
    if frame.shape != (197, EXPECTED_FEATURES + 1) or not np.isfinite(frame.iloc[:, 1:].to_numpy()).all():
        raise RuntimeError(f"invalid final matrix: {frame.shape}")
    feature_path = output_dir / f"radiomics_{args.label}_197x1130.csv"
    qc_path = output_dir / f"radiomics_{args.label}_QC.csv"
    frame.to_csv(feature_path, index=False, float_format="%.17g")
    pd.DataFrame(qc_rows).sort_values("case_id").to_csv(qc_path, index=False)
    summary = {
        "label": args.label, "mask_dir": str(mask_dir),
        "image_dir": str(image_dir), "cases": 197, "features": EXPECTED_FEATURES,
        "workers": args.workers, "elapsed_seconds": time.monotonic() - started,
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "feature_csv": feature_path.name, "qc_csv": qc_path.name,
    }
    (output_dir / f"radiomics_{args.label}_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"complete": args.label, **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
