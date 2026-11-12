#!/usr/bin/env python3
"""Post-hoc, representation-focused comparison using a unified linear probe.

Reproducibility: run with Python 3.13, NumPy, pandas, SciPy and scikit-learn.
All supervised fitting (including scaling and C selection) occurs within the
training data of each outer fold. Five outer seeds are averaged per patient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


SEEDS = (42, 0, 1, 7, 2024)
C_GRID = (0.01, 0.1, 1.0, 10.0)
TASKS = ("Steatosis", "NAS >= 3", "Sinusoidal dilatation")
MODELS = ("Merlin", "RADAR", "TotalFM", "Jolia", "OrganLens", "Radiomics")
MODEL_FILES = {
    "Merlin": "01_Merlin_197x2048.csv",
    "RADAR": "02_RADAR_liver_embeddings_197x256.csv",
    "TotalFM": "03_TotalFM_liver_TS_197x768.csv",
    "Jolia": "04_Jolia_liver_197x576.csv",
    "OrganLens": "05_OrganLens_liver_197x1024.csv",
    "Radiomics": "06_radiomics_resected_TumorFree_197x1130.csv",
}
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs(input_root: Path):
    excel = next(input_root.glob("*.xlsx"), None)
    if excel is None:
        raise FileNotFoundError(f"No cohort Excel file found in {input_root}")
    clinical = pd.read_excel(excel, sheet_name="TCIA_n=197")
    manifest = pd.DataFrame({
        "case_id": clinical["Patient-ID"].astype(str).str.extract(r"(\d{4})", expand=False),
        "Steatosis_yesno": pd.to_numeric(clinical["steatosis_yesno"], errors="raise").astype(int),
        "NASH_score": pd.to_numeric(clinical["NASH_score"], errors="raise"),
        "Presence_sinusoidal_dilata": pd.to_numeric(clinical["presence_sinusoidal_dilata"], errors="raise").astype(int),
    })
    for column in clinical.columns:
        if column not in manifest:
            manifest[column] = clinical[column].values
    matrix_root = input_root
    manifest_path = excel
    if manifest["case_id"].isna().any() or not manifest["case_id"].is_unique:
        raise ValueError("Manifest case IDs are missing or duplicated")
    case_ids = manifest["case_id"].astype(str)
    required = ["Steatosis_yesno", "NASH_score", "Presence_sinusoidal_dilata"]
    if manifest[required].isna().any().any():
        raise ValueError("At least one task label is missing; no silent zero filling")
    labels = {
        "Steatosis": manifest["Steatosis_yesno"].to_numpy(dtype=int),
        "NAS >= 3": (manifest["NASH_score"].to_numpy(dtype=float) >= 3).astype(int),
        "Sinusoidal dilatation": manifest["Presence_sinusoidal_dilata"].to_numpy(dtype=int),
    }
    for task, y in labels.items():
        if not np.isin(y, [0, 1]).all() or np.bincount(y, minlength=2).min() < 5:
            raise ValueError(f"Invalid or insufficient binary labels for {task}")

    matrices = {}
    source_files = [manifest_path]
    for model in MODELS:
        path = matrix_root / MODEL_FILES[model]
        frame = pd.read_csv(path, dtype={"case_id": str})
        if "case_id" not in frame or frame["case_id"].isna().any():
            raise ValueError(f"Missing case IDs: {model}")
        extracted = frame["case_id"].str.extract(r"(\d{4})", expand=False)
        if extracted.isna().any() or not extracted.is_unique:
            raise ValueError(f"Unparseable or duplicated case IDs: {model}")
        frame = frame.drop(columns="case_id")
        frame.index = extracted
        if set(frame.index) != set(case_ids):
            raise ValueError(f"Case ID set does not match manifest: {model}")
        frame = frame.loc[case_ids]
        if model == "RADAR":
            frame = frame.loc[:, frame.columns.str.startswith("liver_")]
        if frame.shape[1] == 0 or frame.columns.duplicated().any():
            raise ValueError(f"No valid features or duplicate names: {model}")
        try:
            X = frame.to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Nonnumeric feature data: {model}") from exc
        if not np.isfinite(X).all():
            raise ValueError(f"Nonfinite feature values: {model}")
        matrices[model] = X
        source_files.append(path)
    return case_ids.to_numpy(), labels, matrices, source_files


def make_probe():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=5000, tol=1e-4,
        )),
    ])


def fit_outer_oof(X, y, task, model, seeds, inner_splits, return_seed_probs=False):
    seed_probs = []
    tuning_rows = []
    for seed in seeds:
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        probs = np.full(len(y), np.nan)
        for fold, (train, test) in enumerate(outer.split(X, y), start=1):
            inner = StratifiedKFold(
                n_splits=inner_splits, shuffle=True,
                random_state=seed * 100 + fold,
            )
            search = GridSearchCV(
                estimator=make_probe(),
                param_grid={"logistic__C": C_GRID},
                scoring="roc_auc", cv=inner, refit=True, n_jobs=1,
                error_score="raise",
            )
            search.fit(X[train], y[train])
            probs[test] = search.predict_proba(X[test])[:, 1]
            tuning_rows.append({
                "Task": task, "Model": model, "Seed": seed,
                "Outer_fold": fold, "Selected_C": search.best_params_["logistic__C"],
                "Inner_mean_AUC": search.best_score_,
                "N_train": len(train), "N_test": len(test),
                "Positive_train": int(y[train].sum()),
                "Positive_test": int(y[test].sum()),
            })
        if not np.isfinite(probs).all():
            raise RuntimeError(f"Incomplete OOF predictions: {task}, {model}, {seed}")
        seed_probs.append(probs)
    if return_seed_probs:
        return np.mean(seed_probs, axis=0), tuning_rows, seed_probs
    return np.mean(seed_probs, axis=0), tuning_rows


def bootstrap_metrics(y, scores, n_boot, seed):
    """Paired, class-stratified patient bootstrap of fixed averaged OOF scores."""
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    point_auc = {model: roc_auc_score(y, scores[model]) for model in MODELS}
    auc_draws = {model: np.empty(n_boot) for model in MODELS}
    brier_draws = {model: np.empty(n_boot) for model in MODELS}
    delta_draws = {model: np.empty(n_boot) for model in MODELS[:-1]}
    for b in range(n_boot):
        index = np.concatenate([
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ])
        y_b = y[index]
        for model in MODELS:
            p_b = scores[model][index]
            auc_draws[model][b] = roc_auc_score(y_b, p_b)
            brier_draws[model][b] = brier_score_loss(y_b, p_b)
        for model in MODELS[:-1]:
            delta_draws[model][b] = auc_draws[model][b] - auc_draws["Radiomics"][b]
    return point_auc, auc_draws, brier_draws, delta_draws


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--inner-splits", type=int, default=3)
    args = parser.parse_args()
    if args.bootstrap < 200 or args.inner_splits < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Require >=200 bootstrap draws, >=2 inner folds and unique seeds")
    np.random.seed(42)
    input_root = args.input_root.resolve()
    case_ids, labels, matrices, source_files = load_inputs(input_root)
    print(f"Verified {len(case_ids)} aligned cases; all feature values finite.", flush=True)
    for task in TASKS:
        print(f"  {task}: positive={labels[task].sum()}", flush=True)
    for model in MODELS:
        print(f"  {model}: {matrices[model].shape}", flush=True)
    if args.check_only:
        return

    output = args.output_dir.resolve()
    if output == input_root or input_root in output.parents:
        raise ValueError("Output must not be inside the source dataset directory")
    output.mkdir(parents=True, exist_ok=True)
    # macOS may create AppleDouble metadata files (._*) on mounted volumes;
    # they are not analysis outputs and should not make a fresh directory fail.
    if any(p for p in output.iterdir() if not p.name.startswith("._")):
        raise FileExistsError(f"Output directory is not empty: {output}")

    all_performance = []
    for task_index, task in enumerate(TASKS):
        y = labels[task]
        scores = {}
        for model in MODELS:
            print(f"Running {task} / {model}", flush=True)
            scores[model], _ = fit_outer_oof(
                matrices[model], y, task, model, args.seeds, args.inner_splits,
            )
        oof = pd.DataFrame({"case_id": case_ids, "y_true": y, **scores})
        task_slug = ("steatosis", "nas3", "sinusoidal_dilatation")[task_index]
        oof.to_csv(output / f"oof_{task_slug}.csv", index=False, float_format="%.17g")

        auc, auc_boot, brier_boot, delta_boot = bootstrap_metrics(
            y, scores, args.bootstrap, seed=4200 + task_index,
        )
        reference_brier = float(y.mean() * (1 - y.mean()))
        for model in MODELS:
            brier = brier_score_loss(y, scores[model])
            ci_auc = np.quantile(auc_boot[model], [0.025, 0.975])
            ci_brier = np.quantile(brier_boot[model], [0.025, 0.975])
            all_performance.append({
                "Task": task, "Model": model, "N": len(y), "Positive": int(y.sum()),
                "Features": matrices[model].shape[1],
                "OOF_AUC": auc[model], "AUC_CI_low": ci_auc[0], "AUC_CI_high": ci_auc[1],
                "Brier_score": brier,
                "Brier_CI_low": ci_brier[0], "Brier_CI_high": ci_brier[1],
                "Scaled_Brier": 1 - brier / reference_brier,
            })

    performance = pd.DataFrame(all_performance)
    performance.to_csv(output / "performance_raw.csv", index=False, float_format="%.17g")

    metadata = {
        "design": "post-hoc sensitivity analysis; unified linear probe",
        "input_root": str(input_root), "output_dir": str(output),
        "source_sha256": {str(path.relative_to(input_root)): sha256(path) for path in source_files},
        "previous_comparison": "Skipped because no prior-result file was present in the independent input view.",
        "seeds": args.seeds, "outer_cv": "StratifiedKFold(n_splits=5, shuffle=True)",
        "inner_cv": f"StratifiedKFold(n_splits={args.inner_splits}, shuffle=True)",
        "C_grid": list(C_GRID), "feature_selection": "none, including Radiomics",
        "scaling": "StandardScaler fitted inside inner/outer training folds",
        "classifier": "LogisticRegression(L2, lbfgs, max_iter=5000, tol=1e-4)",
        "bootstrap": f"{args.bootstrap} class-stratified paired case resamples of fixed seed-averaged OOF predictions",
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print("\nAUC (3 decimals):", flush=True)
    print(performance.pivot(index="Model", columns="Task", values="OOF_AUC")
          .reindex(index=MODELS, columns=TASKS).round(3).to_string(), flush=True)
    print(f"Saved outputs to {output}", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
