#!/usr/bin/env python3
"""Subgroup AUC stability by chemotherapy status for six current model arms.

Analysis: Recompute Figure 08 style subgroup AUCs using current 6-arm, 3-task OOF predictions.
Date: 2026-11-02
Random seed: 42 (task/model/stratum-specific bootstrap streams)
Python: 3.13.9
Key packages: numpy==2.3.5, pandas==2.3.3, scipy==1.16.3, scikit-learn==1.7.2, matplotlib
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import roc_auc_score

np.random.seed(42)

MODELS = ["Merlin", "RADAR", "TotalFM", "Jolia", "OrganLens", "Radiomics"]
TASKS = [
    ("Steatosis", "steatosis", "Steatosis_yesno"),
    ("NAS ≥ 3", "nas3", "NASH_score"),
    ("Sinusoidal dilatation", "sinusoidal_dilatation", "Presence_sinusoidal_dilata"),
]
STRATA = [(1, "Chemotherapy"), (0, "No chemotherapy")]


def case_key(values: pd.Series) -> pd.Series:
    return values.astype(str).str.extract(r"(\d{4})", expand=False)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bootstrap_auc_ci(y: np.ndarray, score: np.ndarray, n_boot: int, seed: int):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if not len(pos) or not len(neg):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos), replace=True),
                    rng.choice(neg, len(neg), replace=True)]
        boot[b] = roc_auc_score(y[idx], score[idx])
    return tuple(np.quantile(boot, [0.025, 0.975]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-root", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--oof-dir", type=Path, default=None,
                        help="Use one fresh six-arm OOF directory instead of archived hybrid OOF paths.")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rev, manifest_path, out = args.revision_root.resolve(), args.cohort_manifest.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if manifest_path.suffix.lower() in {".xlsx", ".xls"}:
        baseline = pd.read_excel(manifest_path, sheet_name="TCIA_n=197")
        baseline = baseline.rename(columns={"Patient-ID": "case_id"})
    else:
        baseline = pd.read_csv(manifest_path, dtype={"case_id": str})
    if "Chemo_before_liver_resection" not in baseline:
        if "chemo_before_liver_resection" in baseline:
            baseline["Chemo_before_liver_resection"] = baseline["chemo_before_liver_resection"]
        else:
            raise ValueError("Chemotherapy grouping column missing from cohort manifest")
    baseline["case_key"] = case_key(baseline.case_id)
    if baseline.case_key.isna().any() or not baseline.case_key.is_unique:
        raise ValueError("Manifest case IDs are missing or duplicated")
    if baseline.Chemo_before_liver_resection.isna().any():
        raise ValueError("Unexpected missing chemotherapy status; original script uses fillna(0)")
    chemo = baseline.set_index("case_key").Chemo_before_liver_resection.astype(int)
    if not set(chemo.unique()).issubset({0, 1}):
        raise ValueError(f"Expected binary chemotherapy coding, found {sorted(chemo.unique())}")

    rows = []
    source_paths = {"cohort_manifest": str(manifest_path)}
    for ti, (task, slug, label_col) in enumerate(TASKS):
        if args.oof_dir is not None:
            fm_path = args.oof_dir / f"oof_{slug}.csv"
            rad_path = fm_path
        else:
            fm_path = rev / "totalfm_resected_compare" / f"oof_{slug}.csv"
            rad_path = rev / "radiomics_sets_main" / "resected_noVesTumor" / f"oof_{slug}.csv"
        fm = pd.read_csv(fm_path, dtype={"case_id": str})
        rad = pd.read_csv(rad_path, dtype={"case_id": str})
        fm["case_key"], rad["case_key"] = case_key(fm.case_id), case_key(rad.case_id)
        if fm.case_key.isna().any() or rad.case_key.isna().any():
            raise ValueError(f"Unparseable case ID for {task}")
        if not fm.case_key.is_unique or not rad.case_key.is_unique:
            raise ValueError(f"Duplicate case ID for {task}")
        if set(fm.case_key) != set(rad.case_key) or set(fm.case_key) != set(chemo.index):
            raise ValueError(f"Cohort mismatch for {task}")
        rad = rad.set_index("case_key").loc[fm.case_key]
        if not np.array_equal(fm.y_true.to_numpy(int), rad.y_true.to_numpy(int)):
            raise ValueError(f"Outcome labels differ between OOF sources for {task}")
        source_col = next((c for c in baseline.columns if c.lower() == label_col.lower()), None)
        if source_col is None:
            raise ValueError(f"Outcome column missing from cohort manifest: {label_col}")
        if label_col == "NASH_score":
            source_y = (pd.to_numeric(baseline.set_index("case_key").loc[fm.case_key, source_col], errors="coerce")
                        .fillna(0).to_numpy() >= 3).astype(int)
        else:
            source_y = (pd.to_numeric(baseline.set_index("case_key").loc[fm.case_key, source_col], errors="coerce")
                        .fillna(0).astype(int).to_numpy())
        y = fm.y_true.to_numpy(int)
        if not np.array_equal(y, source_y):
            raise ValueError(f"OOF outcomes do not match cohort-manifest definition for {task}")
        group_values = fm.case_key.map(chemo).to_numpy(int)
        n_events_all = int(y.sum())
        for mi, model in enumerate(MODELS):
            scores = (rad.Radiomics.to_numpy(float) if model == "Radiomics"
                      else fm[model].to_numpy(float))
            for gi, (group_code, group_name) in enumerate(STRATA):
                mask = group_values == group_code
                yg, sg = y[mask], scores[mask]
                n, events = len(yg), int(yg.sum())
                auc = float(roc_auc_score(yg, sg)) if len(np.unique(yg)) == 2 else np.nan
                if np.isfinite(auc):
                    lo, hi = bootstrap_auc_ci(yg, sg, args.bootstrap,
                                              args.seed + 1000 * ti + 100 * mi + gi)
                else:
                    lo, hi = np.nan, np.nan
                rows.append({
                    "Task": task, "Task_slug": slug, "Model": model,
                    "Chemo_status": group_name, "Chemo_code": group_code,
                    "N": n, "Events": events, "Non_events": n - events,
                    "Prevalence": events / n if n else np.nan,
                    "AUC": auc, "CI_low": lo, "CI_high": hi,
                })
        source_paths[f"{task}_FM_OOF"] = str(fm_path)
        source_paths[f"{task}_Radiomics_OOF"] = str(rad_path)

    result = pd.DataFrame(rows)
    task_order = [x[0] for x in TASKS]
    group_order = [x[1] for x in STRATA]
    result["Task"] = pd.Categorical(result.Task, categories=task_order, ordered=True)
    result["Model"] = pd.Categorical(result.Model, categories=MODELS, ordered=True)
    result["Chemo_status"] = pd.Categorical(result.Chemo_status, categories=group_order, ordered=True)
    result = result.sort_values(["Task", "Model", "Chemo_status"]).reset_index(drop=True)
    result.to_csv(out / "subgroup_auc_raw.csv", index=False, float_format="%.17g")

    manifest = {
        "analysis": "Descriptive chemotherapy-stratified AUC stability for six model arms and three tasks",
        "selected_current_oof_set": "Hybrid current main analysis: all five FMs from resected TotalFM run; Radiomics from resected_noVesTumor run",
        "models_order": MODELS,
        "tasks_order": task_order,
        "groups": {"Chemotherapy": 1, "No chemotherapy": 0},
        "grouping_variable": "Chemo_before_liver_resection; same recoding as Figure08 script (fill missing as 0 then integer); no missing values were present",
        "cohort_N": int(len(baseline)),
        "bootstrap": {"resamples": args.bootstrap, "method": "class-stratified patient bootstrap within subgroup using fixed averaged OOF probabilities", "seed_base": args.seed},
        "inference": "Subgroup AUC estimates are descriptive; no p value or interaction test compares chemotherapy strata.",
        "label_validation": "Each OOF y_true was checked against the cohort manifest task definition.",
        "source_files": source_paths,
        "source_sha256": {k: sha256(Path(v)) for k, v in source_paths.items()},
        "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        "bootstrap_CI_caveat": "Intervals condition on fixed seed-averaged OOF predictions; they do not include model refitting, fold, or seed assignment uncertainty.",
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nSubgroup AUC:")
    print(result[["Task", "Model", "Chemo_status", "N", "Events", "AUC", "CI_low", "CI_high"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
