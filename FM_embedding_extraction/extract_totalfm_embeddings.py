"""Extract 768-dimensional TotalFM liver embeddings for a cohort.

This entry point deliberately excludes zero-shot similarity scoring. It requires
the CT image directory, binary liver masks, a case manifest, and the local
TotalFM checkpoint.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk


def make_label5_mask(source_mask: Path, target_mask: Path) -> Path:
    """Map a binary liver mask to the TotalSegmentator liver label (5)."""
    if not target_mask.exists():
        image = sitk.ReadImage(str(source_mask))
        labels = (sitk.GetArrayFromImage(image) > 0).astype(np.uint8) * 5
        output = sitk.GetImageFromArray(labels)
        output.CopyInformation(image)
        sitk.WriteImage(output, str(target_mask))
    return target_mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--liver-mask-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--model-code", type=Path,
                        default=Path(__file__).resolve().parent / "local_model_code" / "totalfm")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--temporary-mask-dir", type=Path, required=True)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(args.model_code))
    from totalfm import TotalFM

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.temporary_mask_dir.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(args.manifest, dtype={"case_id": str})["case_id"].astype(str).tolist()
    model = TotalFM(device="cpu", weights_path=str(args.weights))
    model._model = model._model.float()
    rows: dict[str, np.ndarray] = {}
    for number, case_id in enumerate(cases, start=1):
        ct_path = args.image_dir / f"CRLM_CT_{case_id}.nii.gz"
        liver_mask = args.liver_mask_dir / f"CRLM-CT-{case_id}.nii.gz"
        label5_mask = make_label5_mask(liver_mask, args.temporary_mask_dir / f"seg5_{case_id}.nii.gz")
        rows[case_id] = np.asarray(model.encode_image(str(ct_path), str(label5_mask))["liver"])
        print(f"[{number}/{len(cases)}] extracted case {case_id}", flush=True)

    output = pd.DataFrame.from_dict(rows, orient="index")
    output.index.name = "case_id"
    output.columns = [f"totalfm_{index:04d}" for index in range(output.shape[1])]
    output.to_csv(args.output_csv)
    print(f"Saved {output.shape[0]} cases × {output.shape[1]} features to {args.output_csv}")


if __name__ == "__main__":
    main()
