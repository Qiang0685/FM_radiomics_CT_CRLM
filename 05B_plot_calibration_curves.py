#!/usr/bin/env python3
"""Plot smoothed OOF calibration curves with patient-bootstrap confidence bands."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

MODELS = ["Merlin", "RADAR", "TotalFM", "Jolia", "OrganLens", "Radiomics"]
TASKS = [
    ("Steatosis", "steatosis"),
    ("NAS ≥ 3", "nas3"),
    ("Sinusoidal dilatation", "sinusoidal_dilatation"),
]
COLORS = {
    "Merlin": "#4C78A8", "RADAR": "#59A14F", "TotalFM": "#F28E2B",
    "Jolia": "#B279A2", "OrganLens": "#76B7B2", "Radiomics": "#7F7F7F",
}


def apply_figure_style(*, frame="boxed", sizes=(7.5, 6.3, 6.3), grid=False):
    base, secondary, tick = sizes
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.size": base,
        "axes.labelsize": base, "legend.fontsize": secondary,
        "xtick.labelsize": tick, "ytick.labelsize": tick,
        "axes.linewidth": 0.65,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.width": 0.65, "ytick.major.width": 0.65,
        "axes.spines.top": frame == "boxed",
        "axes.spines.right": frame == "boxed",
        "axes.spines.left": frame != "none",
        "axes.spines.bottom": frame != "none",
        "axes.grid": bool(grid), "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })


def smooth_curve(y: np.ndarray, p: np.ndarray, grid: np.ndarray, frac: float) -> np.ndarray:
    estimate = lowess(y, p, frac=frac, it=0, xvals=grid, return_sorted=False)
    return np.clip(np.asarray(estimate, dtype=float), 0.0, 1.0)


def curve_with_ci(
    y: np.ndarray,
    p: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
    frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # LOWESS is displayed only over the model-specific range supported by the data.
    low, high = np.quantile(p, [0.01, 0.99])
    if high - low < 0.05:
        low, high = float(p.min()), float(p.max())
    grid = np.linspace(max(0.0, low), min(1.0, high), 101)
    observed = smooth_curve(y, p, grid, frac)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    curves = np.empty((n_bootstrap, len(grid)), dtype=float)
    for replicate in range(n_bootstrap):
        ix = np.r_[rng.choice(pos, len(pos), replace=True),
                   rng.choice(neg, len(neg), replace=True)]
        curves[replicate] = smooth_curve(y[ix], p[ix], grid, frac)
    lower, upper = np.quantile(curves, [0.025, 0.975], axis=0)
    return grid, observed, lower, upper


def draw_one(
    oof_dir: Path,
    figure_dir: Path,
    kind: str,
    *,
    n_bootstrap: int,
    seed: int,
    frac: float,
) -> tuple[Path, Path]:
    apply_figure_style(frame="boxed", sizes=(7.5, 6.3, 6.3), grid=False)
    width_mm, height_mm = 180, 84
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor="white")
    x_mm = (10, 67, 124)
    y_mm, w_mm, h_mm = 14, 50, 50

    for panel, (task, slug) in enumerate(TASKS):
        if kind == "nested_Platt":
            # Current calibration analysis stores these directly. The nested
            # subdirectory is accepted only for historical runs.
            path = oof_dir / f"nested_predictions_{slug}.csv"
            if not path.exists():
                path = oof_dir / "nested_current_main" / f"nested_predictions_{slug}.csv"
        else:
            path = oof_dir / f"main_oof_{slug}.csv"
            if not path.exists():
                path = oof_dir / f"oof_{slug}.csv"
        data = pd.read_csv(path)
        if len(data) != 197 or data.case_id.duplicated().any():
            raise ValueError(f"Invalid OOF table: {path}")
        y = data.y_true.to_numpy(int)
        axis = fig.add_axes([
            x_mm[panel] / width_mm, y_mm / height_mm,
            w_mm / width_mm, h_mm / height_mm,
        ])
        for model_index, model in enumerate(MODELS):
            column = f"{model}_nested_Platt" if kind == "nested_Platt" else model
            p = data[column].to_numpy(float)
            if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
                raise ValueError(f"Invalid probabilities: {task}/{model}")
            grid, estimate, lower, upper = curve_with_ci(
                y, p, n_bootstrap=n_bootstrap,
                seed=seed + panel * 100 + model_index, frac=frac,
            )
            axis.fill_between(grid, lower, upper, color=COLORS[model], alpha=0.08,
                              linewidth=0, zorder=1)
            axis.plot(grid, estimate, color=COLORS[model], linewidth=1.35,
                      label=model, zorder=3)

        axis.plot([0, 1], [0, 1], color="#555555", linestyle=(0, (4, 3)),
                  linewidth=0.9, label="Ideal", zorder=2)
        axis.set(xlim=(0, 1), ylim=(0, 1),
                 xlabel="Predicted probability",
                 ylabel="Observed positive rate" if panel == 0 else "")
        axis.set_xticks(np.linspace(0, 1, 6))
        axis.set_yticks(np.linspace(0, 1, 6))
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, color="#E6E6E6", linestyle=(0, (1, 3)), linewidth=0.5)
        fig.text((x_mm[panel] - 7) / width_mm, (y_mm + h_mm + 2) / height_mm,
                 "ABC"[panel], fontsize=9.5, fontweight="bold", ha="left", va="center")

    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=7, loc="upper center", bbox_to_anchor=(0.52, 0.89),
               frameon=False, columnspacing=0.8, handlelength=1.6, handletextpad=0.3)
    name = "Figure_4_calibration_OOF_probabilities" if kind == "nested_Platt" else "FigS_calibration_raw_supplement"
    pdf, png = figure_dir / f"{name}.pdf", figure_dir / f"{name}.png"
    fig.savefig(pdf, format="pdf", bbox_inches=None, facecolor="white")
    fig.savefig(png, format="png", dpi=300, bbox_inches=None, facecolor="white")
    plt.close(fig)

    return pdf, png


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Figure-only output directory (PDF and PNG only).")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--lowess-frac", type=float, default=0.55)
    args = parser.parse_args()
    if args.bootstrap != 1000:
        raise ValueError("This study uses exactly 1,000 bootstrap resamples")
    if not 0 < args.lowess_frac <= 1:
        raise ValueError("LOWESS fraction must be in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Only the prespecified nested Platt-scaled calibration figure is part of
    # the current results. The former raw-probability supplement is archived
    # under Results/Others/unused_raw_calibration/.
    print(*draw_one(
        args.oof_dir, args.output_dir, "nested_Platt",
        n_bootstrap=args.bootstrap, seed=args.seed, frac=args.lowess_frac,
    ))


if __name__ == "__main__":
    main()
